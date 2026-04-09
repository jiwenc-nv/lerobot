# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""XR controller teleoperator using Isaac Teleop.

Wraps Isaac Teleop's ``ControllersSource`` + ``Se3AbsRetargeter`` +
``GripperRetargeter`` pipeline to produce EE pose and gripper commands
from an XR (VR) controller.  The pipeline construction follows the same
pattern as Isaac Lab's ``_build_franka_stack_pipeline``.
"""

from __future__ import annotations

import numpy as np

from isaacteleop.retargeting_engine.deviceio_source_nodes import (
    ControllersSource,
    HandsSource,
)
from isaacteleop.retargeting_engine.interface import GraphExecutable, OutputCombiner
from isaacteleop.retargeters import (
    GripperRetargeter,
    GripperRetargeterConfig,
    Se3AbsRetargeter,
    Se3RetargeterConfig,
)

from lerobot.types import RobotAction

from ._base import _IsaacTeleopBase
from .config import IsaacTeleopControllerConfig


class IsaacTeleopController(_IsaacTeleopBase):
    """XR controller → EE pose + gripper teleoperator.

    Produces an absolute end-effector pose (position + quaternion) and a
    gripper command from a VR controller.  This is the LeRobot equivalent
    of Isaac Lab's ``IsaacTeleopDevice`` configured with an Se3Abs +
    Gripper pipeline.

    The output dict from ``get_action()`` is designed to feed directly
    into LeRobot's existing ``RobotProcessorPipeline`` (the same one
    used by phone teleoperation) for EE → joint conversion via Placo IK.
    """

    config_class = IsaacTeleopControllerConfig
    name = "isaac_teleop_controller"

    def __init__(self, config: IsaacTeleopControllerConfig):
        super().__init__(config)
        self.config: IsaacTeleopControllerConfig = config

    # ------------------------------------------------------------------
    # Pipeline construction (override point)
    # ------------------------------------------------------------------

    def _build_pipeline(self) -> GraphExecutable:
        """Build Se3Abs + Gripper pipeline from XR controller input.

        Pipeline graph::

            ControllersSource ──┬── Se3AbsRetargeter ── ee_pose (7D)
                                │
            HandsSource ────────┴── GripperRetargeter ── gripper_command (scalar)
                                │
                                └── OutputCombiner

        The ``HandsSource`` is included because ``GripperRetargeter``
        can fall back to pinch-distance detection when controller trigger
        data is unavailable (e.g. when using hand tracking via the
        synthetic hands plugin).
        """
        side = self.config.hand_side
        controllers = ControllersSource(name="controllers")
        hands = HandsSource(name="hands")

        # Se3 absolute pose retargeter — controller aim pose → 7D EE pose
        se3_cfg = Se3RetargeterConfig(
            input_device=f"controller_{side}",
            zero_out_xy_rotation=False,
            target_offset_roll=0.0,
        )
        se3 = Se3AbsRetargeter(se3_cfg, name="ee_pose")
        connected_se3 = se3.connect(
            {f"controller_{side}": controllers.output(f"controller_{side}")}
        )

        # Gripper retargeter — trigger/pinch → scalar command
        gripper_cfg = GripperRetargeterConfig(hand_side=side)
        gripper = GripperRetargeter(gripper_cfg, name="gripper")
        connected_gripper = gripper.connect(
            {
                f"controller_{side}": controllers.output(f"controller_{side}"),
                f"hand_{side}": hands.output(f"hand_{side}"),
            }
        )

        return OutputCombiner(
            {
                "ee_pose": connected_se3.output("ee_pose"),
                "gripper": connected_gripper.output("gripper_command"),
            }
        )

    # ------------------------------------------------------------------
    # Action features
    # ------------------------------------------------------------------

    @property
    def action_features(self) -> dict:
        return {
            "ee_pos": {
                "dtype": "float32",
                "shape": (3,),
                "names": {"x": 0, "y": 1, "z": 2},
            },
            "ee_quat": {
                "dtype": "float32",
                "shape": (4,),
                "names": {"qx": 0, "qy": 1, "qz": 2, "qw": 3},
            },
            "gripper": float,
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    # ------------------------------------------------------------------
    # Action extraction
    # ------------------------------------------------------------------

    def get_action(self) -> RobotAction:
        """Step the Isaac Teleop session and return EE pose + gripper.

        Returns:
            A ``RobotAction`` dict with keys:
            - ``"ee_pos"``: ``np.ndarray`` shape ``(3,)`` — position in meters
            - ``"ee_quat"``: ``np.ndarray`` shape ``(4,)`` — quaternion ``(x, y, z, w)``
            - ``"gripper"``: ``float`` — ``-1.0`` (closed) or ``1.0`` (open)
        """
        if self._session is None:
            raise RuntimeError("Not connected. Call connect() first.")

        result = self._session.step()

        # Se3AbsRetargeter outputs a 7D array: [x, y, z, qx, qy, qz, qw]
        ee_pose = result["ee_pose"][0]
        ee_pose_np = np.asarray(ee_pose, dtype=np.float32)

        # GripperRetargeter outputs a single scalar
        gripper_val = float(result["gripper"][0])

        return {
            "ee_pos": ee_pose_np[:3],
            "ee_quat": ee_pose_np[3:],
            "gripper": gripper_val,
        }
