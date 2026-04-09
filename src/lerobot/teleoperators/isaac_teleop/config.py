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

"""Configuration dataclasses for Isaac Teleop-based teleoperators."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.teleoperators.config import TeleoperatorConfig


@dataclass(kw_only=True)
class IsaacTeleopBaseConfig(TeleoperatorConfig):
    """Shared config fields for all Isaac Teleop-based teleoperators."""

    app_name: str = "LeTeleop"
    """Application name for the OpenXR / Isaac Teleop session."""

    plugins: list = field(default_factory=list)
    """List of ``isaacteleop.teleop_session_manager.PluginConfig`` instances.

    Plugins provide additional device support (e.g. synthetic hands from
    controller input, Manus gloves, foot pedals).
    """


@TeleoperatorConfig.register_subclass("isaac_teleop_controller")
@dataclass(kw_only=True)
class IsaacTeleopControllerConfig(IsaacTeleopBaseConfig):
    """Config for XR controller-based teleoperation.

    Produces EE pose (position + quaternion) and gripper command from
    a VR controller via Isaac Teleop's ``ControllersSource``,
    ``Se3AbsRetargeter``, and ``GripperRetargeter``.
    """

    hand_side: str = "right"
    """Which controller hand to use: ``"left"`` or ``"right"``."""

    gripper_source: str = "trigger"
    """Which controller input drives the gripper: ``"trigger"`` or ``"squeeze"``."""
