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

"""Isaac Teleop-based teleoperators for LeRobot.

Provides LeRobot ``Teleoperator`` subclasses that wrap NVIDIA Isaac
Teleop's ``TeleopSession`` to expose XR input devices (VR controllers,
hand tracking, full-body tracking) as first-class teleop backends.
"""

from .config import IsaacTeleopBaseConfig, IsaacTeleopControllerConfig
from .controller import IsaacTeleopController
