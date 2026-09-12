#!/usr/bin/env python

# Copyright 2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

"""``RobotProfile`` invariants that do not need ``isaacteleop``, a GPU or a headset.

The twin itself is hardware- and headset-dependent and is verified by running it. What is
worth pinning here is the data that decides which twin an arm gets, and the one fact that
spans two repositories: the preview holds the pose the hardware parks at.
"""

import pytest

from lerobot.teleoperators.isaac_teleop.robot_profiles import (
    ROBOT_PROFILES,
    RobotProfile,
    robot_profile_key,
)

# isaacteleop's viz.robot.PREVIEW_ARMS keys. Pinned literally because isaacteleop is an
# optional dependency, so importing it here would skip this test on the machines that most
# need it -- the ones that cannot run the twin.
KNOWN_PREVIEW_ARMS = {"so101", "rebot_devarm_rs"}


def test_every_named_preview_arm_exists_upstream():
    """A typo degrades to "no twin" with a warning at runtime, which is invisible. Catch it
    here instead."""
    named = {p.preview_arm for p in ROBOT_PROFILES.values() if p.preview_arm is not None}
    assert named <= KNOWN_PREVIEW_ARMS, f"unknown preview arm(s): {named - KNOWN_PREVIEW_ARMS}"


def test_the_arms_with_a_twin_are_the_ones_we_expect():
    with_twin = {k for k, p in ROBOT_PROFILES.items() if p.preview_arm is not None}
    assert with_twin == {
        "so101_follower",
        "so100_follower",
        robot_profile_key("rebot_b601_follower", motor_family="rs"),
    }


def test_the_damiao_rebot_has_no_profile_at_all():
    """The B601's two builds share a joint topology and differ in geometry, and Isaac
    Teleop's model is the RobStride one. A Damiao arm must get no twin rather than a near
    fit, which it does by having no profile."""
    assert robot_profile_key("rebot_b601_follower", motor_family="dm") not in ROBOT_PROFILES


@pytest.mark.parametrize("key", sorted(ROBOT_PROFILES))
def test_reset_pose_covers_every_motor(key):
    """``_step_reset`` interpolates to ``reset_pose`` over ``motor_names``; a missing entry
    silently parks that joint at 0.0 instead."""
    profile: RobotProfile = ROBOT_PROFILES[key]
    assert set(profile.reset_pose) == set(profile.motor_names)


# isaacteleop's Q_HOME_DEG / Q_HOME_REBOT_DEG, keyed by the motor each preview joint
# drives. The preview arm holds the pose its follower parks at, so these two move together
# -- a mismatch shows the operator a pose the robot will not adopt. Pinned here rather than
# imported for the reason given on KNOWN_PREVIEW_ARMS above.
#
# The reBot's preview names its joints jointN, mapped onto these motors positionally the way
# RobotProfile.urdf_joint_names already is. Its gripper is a pair of slide joints the twin
# does not address, so unlike the SO-101's it is absent here.
PREVIEW_Q_HOME_DEG = {
    "so101": {
        "shoulder_pan": 0.0,
        "shoulder_lift": -45.0,
        "elbow_flex": 45.0,
        "wrist_flex": 90.0,
        "wrist_roll": 0.0,
        "gripper": 0.0,
    },
    "rebot_devarm_rs": {
        "shoulder_pan": 0.0,
        "shoulder_lift": -5.0,
        "elbow_flex": -10.0,
        "wrist_flex": 0.0,
        "wrist_yaw": 0.0,
        "wrist_roll": 0.0,
    },
}


@pytest.mark.parametrize("key", sorted(ROBOT_PROFILES))
def test_reset_pose_matches_the_preview_arms_home_pose(key):
    profile: RobotProfile = ROBOT_PROFILES[key]
    if profile.preview_arm is None:
        pytest.skip("no twin, so nothing to agree with")
    expected = PREVIEW_Q_HOME_DEG[profile.preview_arm]
    assert set(expected) <= set(profile.motor_names), (
        f"{key}: the preview drives motors this arm does not have: {set(expected) - set(profile.motor_names)}"
    )
    mismatched = {
        name: (profile.reset_pose[name], want)
        for name, want in expected.items()
        if profile.reset_pose[name] != pytest.approx(want)
    }
    assert not mismatched, (
        f"{key}: reset_pose disagrees with the {profile.preview_arm} preview's q_home at "
        f"{mismatched} (reset_pose, q_home); the operator would be shown a pose the arm "
        "will not hold"
    )
