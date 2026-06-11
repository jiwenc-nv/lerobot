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

import numpy as np

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import ProcessorStepRegistry, RobotActionProcessorStep
from lerobot.types import RobotAction

# Gripper closedness [0, 1] -> motor units [0, 100] (RANGE_0_100). The affine
# direction (and any polarity flip) lives here; there is no separate invert knob.
# The SO-101 follower used here calibrates its gripper with motor 100 = fully OPEN
# and 0 = fully CLOSED, the opposite of the retargeter's closedness convention
# (c=0 open, c=1 closed), so the mapping is INVERTED: ``gripper_pos = (1 - c) * 100``.
# (An earlier revision emitted ``c * 100``, which drove the jaw the wrong way -- the
# arm opened when the operator squeezed to grasp.)
# TODO(verify-on-hardware): confirm open/close endpoints + polarity in motor
# units. A verifier watches for the jaw opening when it should close (inverted
# polarity) or not reaching full open/close (wrong travel range).
_GRIPPER_MOTOR_SCALE = 100.0

# Wrist-pitch -> EE orientation target. SO101WristRetargeter emits an ABSOLUTE
# world-elevation pitch [rad] (the controller AIM-ray angle above horizontal, in the
# base frame). We encode it as a base-frame orientation target rotvec about a single
# horizontal axis and let the (orientation-weighted) IK solve position + pitch
# jointly -- the LeRobot analogue of Lab's reduced 3-pos + 1-pitch IK. The roll is
# recovered separately post-IK (OverwriteWristRollFromAngle) and yaw is left to the
# IK, so only the pitch DOF is driven here.
#
# Base frame is X=Forward, Y=Left, Z=Up. A rotation about base Y tilts a forward
# approach up/down (elevation), so the pitch axis is base Y. This intentionally
# ignores reach azimuth (it assumes a roughly forward-facing reach, true for the
# tabletop SO-101 workspace); a side reach would want the axis rotated into the
# reach plane (cf. Lab's azimuth-coupled pitch axis ``n``).
# TODO(verify-on-hardware): confirm the pitch axis and sign for the gripper-frame
# convention (flip _PITCH_SIGN / change _PITCH_AXIS if the gripper tilts the wrong
# way or about the wrong axis).
_PITCH_AXIS = np.array([0.0, 1.0, 0.0], dtype=np.float64)
_PITCH_SIGN = 1.0


@ProcessorStepRegistry.register("map_xr_controller_action_to_robot_action")
@dataclass
class MapXRControllerActionToRobotAction(RobotActionProcessorStep):
    """Maps :class:`XRController` output to the closed-loop IK input contract.

    The XR controller reports an already clutch-rebased *absolute* 7D ``ee_pose``
    in the robot base frame — the clutch retargeter owns the engage latch and the
    no-teleport semantics on the session RUNNING edge, so this step is a pure,
    stateless per-frame mapping with no clutch logic of its own. Each frame it:

    - writes ``ee.x/y/z = ee_pose[:3]`` (the absolute base-frame position target);
    - writes ``ee.wx/wy/wz`` = the rotvec of an absolute **pitch** orientation target
      (``_PITCH_SIGN * wrist_pitch`` about ``_PITCH_AXIS``, base Y). The SO-101 is a
      5-DOF arm, so the orientation is only soft-constrained: paired with a small
      ``orientation_weight`` on ``InverseKinematicsEEToJoints`` this lets the IK solve
      position + pitch jointly (the LeRobot analogue of Lab's reduced 3-pos + 1-pitch
      IK). The terminal roll is recovered separately via ``wrist_roll`` (post-IK
      :class:`OverwriteWristRollFromAngle`); yaw is left to the IK. All six ``ee.*``
      components MUST be present — ``InverseKinematicsEEToJoints`` raises otherwise;
    - writes ``ee.gripper_pos = (1 - closedness) * _GRIPPER_MOTOR_SCALE`` (absolute jaw
      target in motor units ``[0, 100]``, RANGE_0_100; the SO-101 calibrates 100=open,
      0=closed, so closedness is inverted here), passed straight through to
      ``gripper.pos`` by the IK step;
    - carries ``wrist_roll`` [rad] through for the post-IK overwrite step.

    Input keys (from :meth:`XRController.get_action`):
        - ``ee_pose``: ``np.ndarray`` shape ``(7,)`` — ``[x,y,z,qx,qy,qz,qw]`` (base frame).
        - ``wrist_roll``: ``float`` — wrist-roll angle [rad].
        - ``wrist_pitch``: ``float`` — absolute wrist-pitch (world-elevation) angle [rad].
        - ``closedness``: ``float`` — jaw closedness in ``[0, 1]`` (0=open, 1=closed).
        - ``enabled``: ``bool`` — clutch state (consumed/dropped here; the lifecycle
          it drives is handled inside the XR device, not in this mapper).

    Output keys:
        - ``ee.x``, ``ee.y``, ``ee.z``: ``float`` — absolute base-frame position target [m].
        - ``ee.wx``, ``ee.wy``, ``ee.wz``: ``float`` — orientation target rotvec encoding the
          absolute wrist pitch (about ``_PITCH_AXIS``); soft-tracked by the IK.
        - ``ee.gripper_pos``: ``float`` — absolute jaw target in motor units ``[0, 100]``.
        - ``wrist_roll``: ``float`` — wrist-roll angle [rad], for the post-IK
          :class:`OverwriteWristRollFromAngle` step.
    """

    def action(self, action: RobotAction) -> RobotAction:
        ee_pose = np.asarray(action.pop("ee_pose"), dtype=float)
        wrist_roll = float(action.pop("wrist_roll"))
        wrist_pitch = float(action.pop("wrist_pitch"))
        closedness = float(action.pop("closedness"))
        # ``enabled`` drives the session lifecycle inside the XR device; it has no
        # downstream sink in this pipeline, so it is dropped here.
        action.pop("enabled", None)

        ee_pos = ee_pose[:3]
        action["ee.x"] = float(ee_pos[0])
        action["ee.y"] = float(ee_pos[1])
        action["ee.z"] = float(ee_pos[2])
        # Absolute pitch orientation target as a base-frame rotvec about _PITCH_AXIS.
        # Soft-constrained by the IK orientation_weight so the 4 position-capable
        # joints solve position + pitch together; roll is overwritten post-IK.
        orientation_rotvec = (_PITCH_SIGN * wrist_pitch) * _PITCH_AXIS
        action["ee.wx"] = float(orientation_rotvec[0])
        action["ee.wy"] = float(orientation_rotvec[1])
        action["ee.wz"] = float(orientation_rotvec[2])
        # Absolute jaw target in motor units [0, 100]; IK forwards it to gripper.pos.
        # Inverted: closedness c=1 (closed) -> 0, c=0 (open) -> 100 (SO-101 calibration).
        action["ee.gripper_pos"] = (1.0 - closedness) * _GRIPPER_MOTOR_SCALE
        # Carried through for the post-IK OverwriteWristRollFromAngle step.
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
