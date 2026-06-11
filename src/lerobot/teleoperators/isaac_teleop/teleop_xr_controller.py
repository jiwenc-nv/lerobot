#!/usr/bin/env python

# Copyright 2024 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""XR (VR) controller device for NVIDIA Isaac Teleop, exposed to LeRobot.

``XRController`` is the first concrete :class:`IsaacTeleopTeleoperator` device
(see :mod:`lerobot.teleoperators.isaac_teleop.base` for the multi-device
pattern). It wires an Isaac Teleop retargeting pipeline (the three SO-101
retargeters) that turns a single XR controller into a clutch-rebased
end-effector pose, a wrist-roll angle, an absolute wrist-pitch angle, an analog
gripper closedness, and a clutch (``enabled``) signal. The output dict from
:meth:`XRController.get_action` is
designed to feed LeRobot's existing closed-loop IK pipeline (the same one phone
teleoperation uses) via :class:`MapXRControllerActionToRobotAction`.

The shared ``TeleopSession`` lifecycle and per-step health guard live on the
base class; this module only adds the controller-specific pipeline and action
unpacking. The ``isaacteleop`` package is an optional, separately distributed
NVIDIA dependency (the ``isaac-teleop`` extra); all imports of it are deferred
so this module can be imported — and the processor unit-tested — without it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.types import RobotAction

from .base import IsaacTeleopTeleoperator
from .config_isaac_teleop import XRControllerConfig

if TYPE_CHECKING:
    from isaacteleop.retargeting_engine.interface import ExecutionEvents, OutputCombiner

# Source-node name for the static base_T_anchor rebase input fed via
# ``TeleopSession.step(external_inputs=...)`` each frame.
_BASE_T_ANCHOR_INPUT = "base_T_anchor"


class XRController(IsaacTeleopTeleoperator):
    """XR controller -> EE pose + wrist roll + wrist pitch + gripper + clutch teleoperator.

    Wraps Isaac Teleop's three SO-101 retargeters (``SO101ClutchRetargeter``,
    ``SO101WristRetargeter``, ``SO101GripperRetargeter``) behind a single
    ``ControllersSource`` that is statically rebased into the robot base frame
    (``base_T_anchor``) by Isaac Teleop's native ``ControllerTransform``. The wrist
    retargeter emits both an engage-relative roll and an absolute world-elevation
    pitch (the latter recovered from the controller's AIM/pointer ray).

    Squeezing the controller grip past
    :attr:`XRControllerConfig.clutch_threshold` drives the session to ``RUNNING``;
    releasing it drives ``STOPPED`` and freezes the robot, exactly like the phone
    teleoperator's hold-to-enable button. The clutch retargeter latches its
    engage origin on the RUNNING edge (Play), so the arm does not teleport at
    engage — the lifecycle is owned by the retargeters, not re-derived here.
    """

    config_class = XRControllerConfig
    name = "isaac_teleop_controller"

    def __init__(self, config: XRControllerConfig):
        super().__init__(config)
        self.config: XRControllerConfig = config

        # Clutch state from the PREVIOUS frame: the execution events for frame N
        # are built from frame N-1's squeeze so RUNNING/STOPPED reflects the most
        # recent reading available when the session steps. The resulting one-frame
        # (~33 ms at 30 FPS) latency is a small lag, NOT a snap.
        self._enabled = False

        # Build the constant base_T_anchor input ONCE (a TensorGroup is a heavy,
        # isaacteleop-backed object), then reuse it every step. Constructed lazily
        # in connect() so this module imports without isaacteleop installed.
        self._external_inputs: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------

    def _build_pipeline(self) -> OutputCombiner:
        """Build the XR controller retargeting pipeline.

        Reuses the three Isaac Teleop SO-101 retargeters (5-DOF arm) behind a
        single ``ControllersSource`` that is statically rebased into the robot
        base frame by ``ControllerTransform`` (``controllers.transformed(...)``)
        before the retargeters consume it::

            ControllersSource ── .transformed(base_T_anchor) ─┬─ SO101ClutchRetargeter ── ee_pose (7D, base frame)
                                                              ├─ SO101WristRetargeter ─┬─ roll_command (rad)
                                                              │                        └─ pitch_command (rad)
                                                              └─ SO101GripperRetargeter ─ gripper_command (closedness [0,1])

        The transformed controller stream is also exposed as ``"controller"``;
        :meth:`get_action` reads ``SQUEEZE_VALUE`` off it (``ControllerTransform``
        copies the buttons/axes through verbatim, so no separate raw passthrough
        node is needed). The clutch is seeded with ``home_base_T_ee`` and latches
        its engage origin on the session RUNNING edge.
        """
        from isaacteleop.retargeters import (
            SO101ClutchRetargeter,
            SO101GripperRetargeter,
            SO101WristRetargeter,
        )
        from isaacteleop.retargeters.SO101.wrist_retargeter import (
            PITCH_COMMAND_KEY,
            ROLL_COMMAND_KEY,
        )
        from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
        from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
        from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

        side = self.config.hand_side
        controller_key = f"controller_{side}"

        controllers = ControllersSource(name="controllers")
        # Static base_T_anchor rebase fed via external_inputs each step.
        xform = ValueInput(_BASE_T_ANCHOR_INPUT, TransformMatrix())
        transformed = controllers.transformed(xform.output("value"))
        ctrl = transformed.output(controller_key)

        home_base_T_ee = None  # noqa: N806  (frameA_T_frameB transform-matrix convention)
        if self.config.home_base_T_ee is not None:
            home_base_T_ee = np.asarray(self.config.home_base_T_ee, dtype=np.float32)  # noqa: N806

        # Clutch-rebased absolute EE pose -> 7D ee_pose (output key "ee_pose").
        clutch = SO101ClutchRetargeter(
            name="ee_pose", input_device=controller_key, home_base_T_ee=home_base_T_ee
        )
        connected_clutch = clutch.connect({controller_key: ctrl})

        # Wrist retargeter (right hand): emits two scalar channels off the same
        # transformed controller stream --
        #   - roll [rad]: engage-relative swing-twist about the controller's LOCAL Z
        #     axis since engage (output group ROLL_COMMAND_KEY);
        #   - pitch [rad]: absolute world-elevation of the controller's AIM (pointer)
        #     ray above horizontal (output group PITCH_COMMAND_KEY). Measured in the
        #     base frame because the controller stream is already base-rebased.
        wrist = SO101WristRetargeter(name="wrist", input_device=controller_key)
        connected_wrist = wrist.connect({controller_key: ctrl})

        # Proportional jaw closedness [0,1] (output group "gripper_command").
        gripper = SO101GripperRetargeter(name="gripper", input_device=controller_key)
        connected_gripper = gripper.connect({controller_key: ctrl})

        return OutputCombiner(
            {
                "ee_pose": connected_clutch.output("ee_pose"),
                "roll": connected_wrist.output(ROLL_COMMAND_KEY),
                "pitch": connected_wrist.output(PITCH_COMMAND_KEY),
                "gripper": connected_gripper.output("gripper_command"),
                "controller": ctrl,
            }
        )

    def _build_external_inputs(self) -> dict[str, Any]:
        """Materialize the constant ``base_T_anchor`` external input (once, in connect)."""
        from isaacteleop.retargeting_engine.interface import TensorGroup
        from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

        tg = TensorGroup(TransformMatrix())
        tg[0] = np.asarray(self.config.base_T_anchor, dtype=np.float32)
        return {_BASE_T_ANCHOR_INPUT: {"value": tg}}

    def connect(self, calibrate: bool = True) -> None:
        super().connect(calibrate=calibrate)
        # Built after a successful connect so a failed connect leaves no half-state.
        self._external_inputs = self._build_external_inputs()

    def _execution_events_from_squeeze(self, *, enabled: bool) -> ExecutionEvents:
        """Build the session ``ExecutionEvents`` for this frame from the clutch.

        ``RUNNING`` while the clutch is engaged, ``STOPPED`` otherwise; the clutch
        retargeter latches its engage origin on the RUNNING edge. ``reset`` is left
        ``False`` here (per-episode reset wiring is deferred to the record loop).
        """
        from isaacteleop.retargeting_engine.interface import ExecutionEvents, ExecutionState

        state = ExecutionState.RUNNING if enabled else ExecutionState.STOPPED
        return ExecutionEvents(execution_state=state, reset=False)

    # ------------------------------------------------------------------
    # Action features
    # ------------------------------------------------------------------

    @property
    def action_features(self) -> dict:
        return {
            "ee_pose": {
                "dtype": "float32",
                "shape": (7,),
                "names": {"x": 0, "y": 1, "z": 2, "qx": 3, "qy": 4, "qz": 5, "qw": 6},
            },
            # ``get_action`` returns scalars for these three, so the advertised
            # shape is () (0-d) to stay consistent with the returned values.
            "wrist_roll": {
                "dtype": "float32",
                "shape": (),
                "names": None,
            },
            "wrist_pitch": {
                "dtype": "float32",
                "shape": (),
                "names": None,
            },
            "closedness": {
                "dtype": "float32",
                "shape": (),
                "names": None,
            },
            "enabled": {
                "dtype": "bool",
                "shape": (),
                "names": None,
            },
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    # ------------------------------------------------------------------
    # Action extraction
    # ------------------------------------------------------------------

    def get_action(self) -> RobotAction:
        """Step the Isaac Teleop session and return EE pose + roll + pitch + gripper + clutch.

        Drives the session lifecycle from the clutch: the ``ExecutionEvents`` for
        this step are built from the PREVIOUS frame's squeeze (the freshest reading
        available when the session steps), and the static ``base_T_anchor`` rebase
        is supplied as a constant external input. After stepping, ``enabled`` is
        recomputed from THIS frame's squeeze (for the device output and to drive
        next frame's lifecycle). The one-frame (~33 ms at 30 FPS) lag between
        squeeze and RUNNING is a small latency, not a snap. Because ``_enabled``
        starts ``False``, the very first step is always STOPPED regardless of the
        initial squeeze; engagement takes effect on the following step.

        Returns:
            A ``RobotAction`` dict with keys:

            - ``"ee_pose"``: ``np.ndarray`` shape ``(7,)`` — clutch-rebased
              absolute pose ``[x, y, z, qx, qy, qz, qw]`` in the robot base
              frame (position in metres).
            - ``"wrist_roll"``: ``float`` — wrist-roll angle in **radians**
              (swing-twist about the controller's local Z, measured since engage).
            - ``"wrist_pitch"``: ``float`` — wrist-pitch angle in **radians**
              (absolute world-elevation of the controller AIM ray above horizontal,
              in the robot base frame).
            - ``"closedness"``: ``float`` — jaw closedness in ``[0, 1]``
              (``0`` = open, ``1`` = closed).
            - ``"enabled"``: ``bool`` — clutch state (squeeze held past
              ``clutch_threshold``). Drives the session lifecycle; consumed by
              the downstream pipeline as the device->lifecycle signal.
        """
        # Steps the session and applies the shared staleness/worker-health
        # guard (see IsaacTeleopTeleoperator._step). The execution events come
        # from the previous frame's clutch state; the base_T_anchor rebase is a
        # constant external input.
        events = self._execution_events_from_squeeze(enabled=self._enabled)
        result = self._step(execution_events=events, external_inputs=self._external_inputs)

        # Retargeter outputs are batched (leading slot/batch dim); index [0]
        # selects the single tracked controller and drops that dim.
        # SO101ClutchRetargeter outputs a 7D array: [x, y, z, qx, qy, qz, qw].
        ee_pose = np.asarray(result["ee_pose"][0], dtype=np.float32)
        # SO101WristRetargeter emits two scalars: an engage-relative roll [rad] and
        # an absolute world-elevation pitch [rad] (from the AIM/pointer ray).
        wrist_roll = float(result["roll"][0])
        wrist_pitch = float(result["pitch"][0])
        # SO101GripperRetargeter emits a single closedness scalar in [0, 1].
        closedness = float(result["gripper"][0])

        # Transformed controller stream -> clutch from the squeeze analog
        # (ControllerTransform copies buttons/axes through verbatim). When the
        # controller is not tracked the optional group is None; treat that as
        # "not engaged" so the robot freezes safely.
        from isaacteleop.retargeting_engine.tensor_types.indices import ControllerInputIndex

        controller = result["controller"]
        # Defensive: a controller group may not be tracked every frame
        # (untracked/odd frame, missing squeeze axis, unexpected wrapper shape). Any
        # failure to read the squeeze is treated as "not engaged" (squeeze = 0.0) so
        # the arm freezes safely instead of crashing the teleop loop.
        if getattr(controller, "is_none", False):
            squeeze = 0.0
        else:
            try:
                squeeze = float(controller[ControllerInputIndex.SQUEEZE_VALUE])
            except (IndexError, KeyError, TypeError, ValueError):
                squeeze = 0.0
        enabled = bool(squeeze > self.config.clutch_threshold)
        # Store for next frame's execution events (one-frame delay, see docstring).
        self._enabled = enabled

        return {
            "ee_pose": ee_pose,
            "wrist_roll": wrist_roll,
            "wrist_pitch": wrist_pitch,
            "closedness": closedness,
            "enabled": enabled,
        }
