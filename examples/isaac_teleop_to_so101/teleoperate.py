# !/usr/bin/env python

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

"""Teleoperate an SO-101 follower arm with an XR (VR) controller via Isaac Teleop.

This mirrors ``examples/phone_to_so100/teleoperate.py`` but swaps the phone for
an XR controller. The three Isaac Teleop SO-101 retargeters own the clutch /
roll / gripper logic (keyed off the session RUNNING/STOPPED lifecycle the XR
device drives from the squeeze), and emit an already clutch-rebased *absolute*
base-frame EE pose, so the LeRobot side is a thin absolute-pose, position-only
IK path::

    XRController.get_action()                       # clutch-rebased abs EE pose + roll + pitch + closedness + clutch
      -> MapXRControllerActionToRobotAction         # ee.x/y/z = abs pose, ee.w* = pitch rotvec, ee.gripper_pos, wrist_roll [rad]
      -> EEBoundsAndSafety                           # workspace clip + per-frame jump clamp
      -> InverseKinematicsEEToJoints(ow=0.0)         # position-only Placo IK (passes ee.gripper_pos -> gripper.pos)
      -> OverwriteWristRollFromAngle                 # wrist_roll [rad] -> wrist_roll.pos [deg]

Squeeze (and hold) the controller grip past ``clutch_threshold`` to engage; the
clutch retargeter latches its engage origin on the session RUNNING edge so the
arm does not jump. ``EEReferenceAndDelta`` / ``GripperVelocityToJoint`` are NOT
in this pipeline: the absolute pose goes straight to IK, and the analog gripper
closedness is emitted as an absolute ``ee.gripper_pos`` joint target.

Startup / safety contract: by default the script slews all joints to their URDF
origin (arm joints to 0°, gripper to 100 = fully open) over ``--reset-duration``
seconds before entering the loop.  Pass ``--no-reset-to-origin`` to skip this slew
and keep the arm exactly where it is.  After the slew (or if skipped) the clutch
seeds its reset-origin home from the arm's MEASURED pose (``home_base_T_ee`` = FK
of the joints read right after the slew), so the seeded home equals the post-reset
position and the first engage is jump-free.  The robot is commanded ONLY while the
clutch is engaged; while disengaged the loop re-sends the measured joints (an
explicit hold), and releasing the clutch freezes it in place.

NOTE: EEBoundsAndSafety raises on a per-frame jump > max_ee_step_m; the clutch's
no-teleport keeps frames small, but set a generous bound for bring-up.

Requires the ``isaac-teleop`` extra (``isaacteleop``) and an OpenXR runtime.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    RobotProcessorPipeline,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    InverseKinematicsEEToJoints,
)
from lerobot.teleoperators.isaac_teleop import (
    MapXRControllerActionToRobotAction,
    OverwriteWristRollFromAngle,
    XRController,
    XRControllerConfig,
)
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30

# Per-frame EE rate limit [m]. EEBoundsAndSafety (raise_on_jump=False below) clamps
# any per-frame position change above this instead of raising, so MAX_EE_STEP_M is a
# safety rate limit, not a crash threshold: at FPS=30, 0.1 m/frame caps EE speed at
# ~3 m/s, which deliberate teleop rarely exceeds while still absorbing controller
# tracking glitches as a single slow frame. (Only the per-frame change is bounded;
# the absolute target can still be far — that is what end_effector_bounds clips.)
MAX_EE_STEP_M = 0.1

# CloudXR device-profile env file passed to the launcher (see webxr.env next to
# this script). Resolved absolutely so it loads regardless of the working dir.
CLOUDXR_ENV_FILE = str(Path(__file__).parent / "webxr.env")

# Optional file written by record_reset_pose.py.  When present its values take
# priority over RESET_ORIGIN_DEG.
RESET_POSE_FILE = Path(__file__).parent / "reset_pose.json"

# Default duration [s] for the startup reset-to-origin slew.
RESET_DURATION_S = 5.0

# Reset target in each motor's native units.
# Arm joint values are the Lab SO-101 stack-task init pose (see Lab repo:
# source/isaaclab_tasks/isaaclab_tasks/contrib/stack/config/so101/
# stack_joint_pos_env_cfg.py, _SO101_STACK_INIT_JOINT_POS) converted from URDF
# radians to degrees via np.rad2deg.  This mid-range pose (elbow/wrist bent)
# avoids the boundary singularity of a fully-extended 5-DOF arm.
# Assumes standard calibration where 0° = URDF 0 rad (homing pose).
# Gripper uses MotorNormMode.RANGE_0_100; 100 = fully open (safe for teleop start).
RESET_ORIGIN_DEG: dict[str, float] = {
    "shoulder_pan":  float(np.rad2deg(0.0)),
    "shoulder_lift": float(np.rad2deg(-0.6)),
    "elbow_flex":    float(np.rad2deg(0.8)),
    "wrist_flex":    float(np.rad2deg(0.6)),
    "wrist_roll":    float(np.rad2deg(0.0)),
    "gripper":       100.0,
}


def _load_reset_target(motor_names: list[str]) -> dict[str, float]:
    """Return reset targets: reset_pose.json if present, else RESET_ORIGIN_DEG."""
    if RESET_POSE_FILE.exists():
        saved = json.loads(RESET_POSE_FILE.read_text())
        # Fill any missing motors from the fallback dict.
        return {name: float(saved.get(name, RESET_ORIGIN_DEG.get(name, 0.0))) for name in motor_names}
    return {name: RESET_ORIGIN_DEG.get(name, 0.0) for name in motor_names}


def move_to_origin(robot, motor_names: list[str], duration_s: float = RESET_DURATION_S) -> None:
    """Linearly slew all joints from their current positions to the reset target.

    Target source priority: reset_pose.json (recorded by record_reset_pose.py)
    > RESET_ORIGIN_DEG (Lab-derived hardcoded fallback).
    """
    obs = robot.get_observation()
    start = {name: float(obs[f"{name}.pos"]) for name in motor_names}
    target = _load_reset_target(motor_names)
    source = "reset_pose.json" if RESET_POSE_FILE.exists() else "hardcoded defaults"
    print(f"Reset target source: {source}")
    n_steps = max(1, int(duration_s * FPS))
    print(f"Resetting to origin over {duration_s:.1f} s ({n_steps} steps)…")
    for step in range(1, n_steps + 1):
        alpha = step / n_steps
        action = {f"{name}.pos": start[name] + alpha * (target[name] - start[name]) for name in motor_names}
        robot.send_action(action)
        precise_sleep(1.0 / FPS)
    print("Reset complete.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/ttyACM0",
        help="Serial port the SO-101 follower arm is connected to (default: /dev/ttyACM0).",
    )
    parser.add_argument(
        "--id",
        type=str,
        default="so101_follower_arm",
        help="Device id for the SO-101 follower arm (selects its calibration; default: so101_follower_arm).",
    )
    parser.add_argument(
        "--reset-to-origin",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Slew all joints to URDF origin before entering the teleop loop (default: True). "
             "Pass --no-reset-to-origin to keep the arm exactly where it is at startup.",
    )
    parser.add_argument(
        "--reset-duration",
        type=float,
        default=RESET_DURATION_S,
        metavar="SEC",
        help=f"Duration in seconds for the reset-to-origin slew (default: {RESET_DURATION_S}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    robot_config = SO100FollowerConfig(port=args.port, id=args.id, use_degrees=True)

    # SO100Follower is the shared SO-100/SO-101 follower class: so_follower
    # registers the same class under both "so100_follower" and "so101_follower".
    # Here it is configured for SO-101 (see the so101_new_calib.urdf below).
    robot = SO100Follower(robot_config)
    motor_names = list(robot.bus.motors.keys())

    # Loads ./SO101/so101_new_calib.urdf relative to this folder. Run
    # `python download_assets.py` from this directory first to fetch the URDF and
    # its meshes from the SO-ARM100 repo:
    # https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101
    kinematics_solver = RobotKinematics(
        urdf_path="./SO101/so101_new_calib.urdf",
        target_frame_name="gripper_frame_link",
        joint_names=motor_names,
    )

    # Connect the robot FIRST so the slew and clutch-home seed can use live
    # joint readings. With --reset-to-origin (default) the arm is smoothly moved
    # to URDF origin before the teleop loop starts; the clutch home is then seeded
    # from the post-slew measured pose so the first engage is jump-free.
    # Pass --no-reset-to-origin to skip the slew entirely.
    robot.connect()
    if args.reset_to_origin:
        move_to_origin(robot, motor_names, args.reset_duration)
    obs0 = robot.get_observation()
    q_measured_deg = np.array([float(obs0[f"{name}.pos"]) for name in motor_names], dtype=float)
    home_base_T_ee = kinematics_solver.forward_kinematics(q_measured_deg)  # noqa: N806

    teleop_config = XRControllerConfig(
        hand_side="right",
        clutch_threshold=0.5,
        cloudxr_env_file=CLOUDXR_ENV_FILE,
        home_base_T_ee=home_base_T_ee.tolist(),
    )
    teleop_device = XRController(teleop_config)

    # Build pipeline: XR action -> EE pose action -> joint action.
    xr_to_robot_joints_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            # Stateless: ee.x/y/z = clutch-rebased absolute base-frame pose,
            # ee.w* = absolute wrist-pitch orientation target (rotvec about base Y),
            # ee.gripper_pos = (1 - closedness) * 100, wrist_roll [rad] carried through.
            MapXRControllerActionToRobotAction(),
            # Clip to the workspace + RATE-LIMIT each frame. raise_on_jump=False:
            # an over-limit step (e.g. a transient XR controller tracking glitch)
            # is clamped to MAX_EE_STEP_M and warned, NOT raised -- a crash mid-loop
            # would leave the arm uncontrolled. A glitch is absorbed as one slow
            # frame; a target that is *persistently* out of reach will warn every
            # frame (investigate base_T_anchor / scale, not this clamp).
            # TODO(tune-on-hardware): tighten end_effector_bounds to the SO-101's
            # actual reachable workspace (the [-1,1]m box is loose for bring-up).
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
                max_ee_step_m=MAX_EE_STEP_M,
                raise_on_jump=False,
            ),
            # Pure position IK (orientation_weight=0.0): the SO-101 is 5-DOF and cannot
            # track a full 6-DOF pose; a non-zero orientation weight fights position
            # tracking on the redundant joints. Wrist roll is recovered post-IK by
            # OverwriteWristRollFromAngle; yaw and pitch are left free.
            # initial_guess_current_joints=False: use the previous IK solution as the
            # seed for smoother, branch-consistent joint trajectories frame-to-frame.
            InverseKinematicsEEToJoints(
                kinematics=kinematics_solver,
                motor_names=motor_names,
                initial_guess_current_joints=False,
                orientation_weight=0.0,
            ),
            # Post-IK: write the operator's wrist roll [rad] onto wrist_roll.pos [deg],
            # overriding the under-determined IK roll on the 5-DOF arm.
            OverwriteWristRollFromAngle(),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # Robot is already connected (above, for the measured-home seed). Connect teleop.
    teleop_device.connect()

    init_rerun(session_name="xr_so101_teleop")

    if not robot.is_connected or not teleop_device.is_connected:
        raise ValueError("Robot or teleop is not connected!")

    print("Starting teleop loop. Squeeze and move the controller to teleoperate the robot...")
    while True:
        t0 = time.perf_counter()

        robot_obs = robot.get_observation()
        xr_action = teleop_device.get_action()

        # SAFETY GATE: command the robot ONLY while the clutch is engaged. While
        # disengaged, re-send the MEASURED joints (an explicit hold) so the arm
        # stays exactly where it is — launching the script (clutch released) never
        # moves it, and releasing the clutch mid-session freezes it in place. This
        # is the joint-space hold; without it the IK would keep driving the arm
        # toward the home pose every frame regardless of engagement.
        if bool(xr_action["enabled"]):
            joint_action = xr_to_robot_joints_processor((xr_action, robot_obs))
        else:
            joint_action = {f"{name}.pos": float(robot_obs[f"{name}.pos"]) for name in motor_names}

        _ = robot.send_action(joint_action)

        log_rerun_data(observation=xr_action, action=joint_action)

        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))


if __name__ == "__main__":
    main()
