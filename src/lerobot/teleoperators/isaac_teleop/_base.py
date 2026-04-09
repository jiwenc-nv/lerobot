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

"""Base class for Isaac Teleop-based LeRobot teleoperators.

``_IsaacTeleopBase`` owns the ``TeleopSession`` lifecycle (connect /
disconnect) and delegates pipeline construction to subclasses via the
abstract ``_build_pipeline()`` method.  This mirrors how Isaac Lab's
``IsaacTeleopCfg`` uses a ``pipeline_builder`` callable — each env/task
provides its own pipeline graph and the device just executes it.  Here
each ``Teleoperator`` subclass plays that role.
"""

from __future__ import annotations

import abc
import logging
from typing import Any

from isaacteleop.retargeting_engine.interface import GraphExecutable
from isaacteleop.teleop_session_manager import TeleopSession, TeleopSessionConfig

from lerobot.teleoperators.teleoperator import Teleoperator

from .config import IsaacTeleopBaseConfig

logger = logging.getLogger(__name__)


class _IsaacTeleopBase(Teleoperator):
    """Shared lifecycle for all Isaac Teleop-based teleoperators.

    The base class owns the ``TeleopSession`` lifecycle entirely.
    Subclasses customize behavior through two override points:

    1. ``_build_pipeline() → GraphExecutable``
       Wires up the Isaac Teleop source/retargeter graph specific to the
       input device type (controllers, hands, full body, etc.).
       Called once by ``connect()``.

    2. ``get_action() → RobotAction``
       Steps the session and unpacks the pipeline output into a LeRobot
       ``RobotAction`` dict matching the subclass's ``action_features``.
    """

    config_class = IsaacTeleopBaseConfig

    def __init__(self, config: IsaacTeleopBaseConfig):
        super().__init__(config)
        self.config = config
        self._session: TeleopSession | None = None

    # ------------------------------------------------------------------
    # Pipeline construction (subclass override point)
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _build_pipeline(self) -> GraphExecutable:
        """Build the Isaac Teleop retargeting pipeline.

        Returns a ``GraphExecutable`` (e.g. ``OutputCombiner``,
        ``RetargeterSubgraph``) that will be passed to
        ``TeleopSessionConfig.pipeline``.  The base class calls this
        exactly once during ``connect()``.

        The returned pipeline's output keys must match what
        ``get_action()`` expects to unpack.
        """
        ...

    # ------------------------------------------------------------------
    # Teleoperator lifecycle (owned by base class)
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._session is not None

    @property
    def is_calibrated(self) -> bool:
        return True  # Tracking devices are self-calibrating

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        if self._session is not None:
            raise RuntimeError("Already connected. Call disconnect() first.")

        pipeline = self._build_pipeline()
        session_config = TeleopSessionConfig(
            app_name=self.config.app_name,
            pipeline=pipeline,
            plugins=self.config.plugins,
        )
        self._session = TeleopSession(session_config)
        self._session.__enter__()
        logger.info("Isaac Teleop session started: %s", self.config.app_name)

    def disconnect(self) -> None:
        if self._session is not None:
            self._session.__exit__(None, None, None)
            self._session = None
            logger.info("Isaac Teleop session ended")

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass  # Phase 2: haptic feedback
