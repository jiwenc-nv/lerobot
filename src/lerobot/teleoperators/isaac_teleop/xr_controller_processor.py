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

"""Processor step that maps XR controller actions to robot EE targets.

Analogous to ``MapPhoneActionToRobotAction`` in
``lerobot/teleoperators/phone/phone_processor.py``, this step bridges
:meth:`XRController.get_action` output to the input contract of the downstream
closed-loop IK pipeline (``EEBoundsAndSafety`` -> ``InverseKinematicsEEToJoints``).

This module is pure ``numpy`` and does **not** import ``isaacteleop``, so it
can be unit-tested without the XR runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import ProcessorStepRegistry, RobotActionProcessorStep
from lerobot.types import RobotAction

# Gripper closedness [0, 1] -> motor units [0, 100] (RANGE_0_100). The affine
# direction (and any polarity flip) lives here; there is no separate invert knob.
# The SO-101 follower used here calibrates its gripper with motor 100 = fully OPEN
# and 0 = fully CLOSED, the opposite of the retargeter's closedness convention
# (c=0 open, c=1 closed), so the mapping is INVERTED: ``gripper_pos = (1 - c) * 100``.
# TODO(verify-on-hardware): confirm open/close endpoints + polarity in motor
# units. A verifier watches for the jaw opening when it should close (inverted
# polarity) or not reaching full open/close (wrong travel range).
_GRIPPER_MOTOR_SCALE = 100.0


@ProcessorStepRegistry.register("map_xr_controller_action_to_robot_action")
@dataclass
class MapXRControllerActionToRobotAction(RobotActionProcessorStep):
    """Maps :class:`XRController` output to the closed-loop IK input contract.

    The XR controller reports an already clutch-rebased *absolute* 7D ``ee_pose``
    in the robot base frame — the clutch retargeter owns the engage latch and the
    no-teleport semantics on the session RUNNING edge, so this step is a pure,
    stateless per-frame mapping with no clutch logic of its own. Each frame it:

    - writes ``ee.x/y/z = ee_pose[:3]`` (the absolute base-frame position target);
    - writes ``ee.wx/wy/wz = 0`` — orientation is unconstrained; the IK runs
      position-only (``orientation_weight=0.0``). The terminal roll DOF is recovered
      separately post-IK via :class:`OverwriteWristRollFromAngle`; pitch and yaw are
      left free. All six ``ee.*`` components must be present for
      ``InverseKinematicsEEToJoints`` to accept the action;
    - writes ``ee.gripper_pos = (1 - closedness) * _GRIPPER_MOTOR_SCALE`` (absolute jaw
      target in motor units ``[0, 100]``, RANGE_0_100; the SO-101 calibrates 100=open,
      0=closed, so closedness is inverted here), passed straight through to
      ``gripper.pos`` by the IK step;
    - carries ``wrist_roll`` [rad] through for the post-IK overwrite step.

    Input keys (from :meth:`XRController.get_action`):
        - ``ee_pose``: ``np.ndarray`` shape ``(7,)`` — ``[x,y,z,qx,qy,qz,qw]`` (base frame).
        - ``wrist_roll``: ``float`` — wrist-roll angle [rad].
        - ``wrist_pitch``: ``float`` — dropped; not used with position-only IK.
        - ``closedness``: ``float`` — jaw closedness in ``[0, 1]`` (0=open, 1=closed).
        - ``enabled``: ``bool`` — clutch state (dropped here; lifecycle handled in device).

    Output keys:
        - ``ee.x``, ``ee.y``, ``ee.z``: ``float`` — absolute base-frame position target [m].
        - ``ee.wx``, ``ee.wy``, ``ee.wz``: ``float`` — zeros (orientation unconstrained).
        - ``ee.gripper_pos``: ``float`` — absolute jaw target in motor units ``[0, 100]``.
        - ``wrist_roll``: ``float`` — wrist-roll angle [rad], for :class:`OverwriteWristRollFromAngle`.
    """

    def action(self, action: RobotAction) -> RobotAction:
        ee_pose = action.pop("ee_pose")
        wrist_roll = float(action.pop("wrist_roll"))
        action.pop("wrist_pitch", None)
        closedness = float(action.pop("closedness"))
        action.pop("enabled", None)

        action["ee.x"] = float(ee_pose[0])
        action["ee.y"] = float(ee_pose[1])
        action["ee.z"] = float(ee_pose[2])
        # Orientation unconstrained: IK runs position-only (orientation_weight=0.0).
        action["ee.wx"] = 0.0
        action["ee.wy"] = 0.0
        action["ee.wz"] = 0.0
        # Inverted: closedness c=1 (closed) -> 0, c=0 (open) -> 100 (SO-101 calibration).
        action["ee.gripper_pos"] = (1.0 - closedness) * _GRIPPER_MOTOR_SCALE
        action["wrist_roll"] = wrist_roll
        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in ["ee_pose", "wrist_pitch", "closedness", "enabled"]:
            features[PipelineFeatureType.ACTION].pop(feat, None)

        for feat in [
            "ee.x",
            "ee.y",
            "ee.z",
            "ee.wx",
            "ee.wy",
            "ee.wz",
            "ee.gripper_pos",
            "wrist_roll",
        ]:
            features[PipelineFeatureType.ACTION][feat] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))

        return features
