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

"""Processor step that maps XR controller actions to robot EE targets.

Analogous to ``MapPhoneActionToRobotAction`` in
``lerobot/teleoperators/phone/phone_processor.py``, this step bridges
``IsaacTeleopController.get_action()`` output to the format expected by
LeRobot's ``EEReferenceAndDelta`` processor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import ProcessorStepRegistry, RobotActionProcessorStep
from lerobot.types import RobotAction


@ProcessorStepRegistry.register("map_xr_action_to_robot_action")
@dataclass
class MapXRActionToRobotAction(RobotActionProcessorStep):
    """Maps ``IsaacTeleopController`` output to the robot action format.

    The XR controller reports *absolute* world-space poses, but the
    downstream ``EEReferenceAndDelta`` processor expects *relative deltas*.
    On the first enabled frame this step captures the controller pose as
    the origin and outputs the displacement from that origin on every
    subsequent frame.

    Input keys (from ``IsaacTeleopController.get_action()``):
        - ``ee_pos``: ``np.ndarray`` shape ``(3,)`` — EE position (absolute)
        - ``ee_quat``: ``np.ndarray`` shape ``(4,)`` — EE quaternion ``(x,y,z,w)`` (absolute)
        - ``gripper``: ``float`` — ``-1.0`` (closed) or ``1.0`` (open)

    Output keys (for ``EEReferenceAndDelta``):
        - ``enabled``: ``bool`` — always ``True`` while connected
        - ``target_x``, ``target_y``, ``target_z``: ``float`` — EE position delta from origin
        - ``target_wx``, ``target_wy``, ``target_wz``: ``float`` — EE rotation delta as rotvec
        - ``gripper_vel``: ``float`` — gripper velocity command
    """

    _enabled_prev: bool = field(default=False, init=False, repr=False)
    _origin_pos: np.ndarray | None = field(default=None, init=False, repr=False)
    _origin_rot_inv: Rotation | None = field(default=None, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        ee_pos = action.pop("ee_pos")
        ee_quat = action.pop("ee_quat")
        gripper_cmd = action.pop("gripper")

        rot = Rotation.from_quat(ee_quat)

        # Capture origin on first frame
        if self._origin_pos is None:
            self._origin_pos = np.array(ee_pos, dtype=float)
            self._origin_rot_inv = rot.inv()

        # Compute deltas relative to captured origin
        delta_pos = np.asarray(ee_pos, dtype=float) - self._origin_pos
        delta_rot = self._origin_rot_inv * rot
        delta_rotvec = delta_rot.as_rotvec()

        # Map gripper scalar (-1.0 closed, 1.0 open) to velocity
        gripper_vel = float(gripper_cmd)

        action["enabled"] = True
        action["target_x"] = float(delta_pos[0])
        action["target_y"] = float(delta_pos[1])
        action["target_z"] = float(delta_pos[2])
        action["target_wx"] = float(delta_rotvec[0])
        action["target_wy"] = float(delta_rotvec[1])
        action["target_wz"] = float(delta_rotvec[2])
        action["gripper_vel"] = gripper_vel
        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in ["ee_pos", "ee_quat", "gripper"]:
            features[PipelineFeatureType.ACTION].pop(feat, None)

        for feat in [
            "enabled",
            "target_x",
            "target_y",
            "target_z",
            "target_wx",
            "target_wy",
            "target_wz",
            "gripper_vel",
        ]:
            features[PipelineFeatureType.ACTION][feat] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features
