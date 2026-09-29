import numpy as np
import pytest

from src.models import JointDefinition, JointState, TargetPose, IKSolution
from src.module01_robot_configuration import create_robot_definition
from src.module03_transform import dh_to_transform
from src.module04_forward_kinematics import forward_kinematics
from src.module05_inverse_kinematics import solve_inverse_kinematics
from src.module06_validation import validate_ik_solutions

@pytest.fixture
def robot():
    return create_robot_definition(
        "Test 4DOF",
        [
            JointDefinition(0, "revolute", 0, 100, 0, 90, -180, 180),
            JointDefinition(1, "revolute", 0, 0, 150, 0, -90, 90),
            JointDefinition(2, "revolute", 0, 0, 120, 0, -120, 120),
            JointDefinition(3, "revolute", 0, 0, 80, 0, -180, 180),
        ],
    )

def test_dh_identity_case():
    assert np.allclose(dh_to_transform(0, 0, 0, 0), np.eye(4))

def test_robot_has_four_joints(robot):
    assert len(robot.joints) == 4

def test_fk_returns_valid_transform(robot):
    result = forward_kinematics(robot, JointState([0, 0, 0, 0]))
    assert result.final_transform.shape == (4, 4)
    assert result.pose.position.shape == (3,)

def test_fk_ik_consistency(robot):
    known_state = JointState([20, 25, -15, 10])
    fk = forward_kinematics(robot, known_state)
    target = TargetPose(position=fk.pose.position)

    guesses = [
        [0, 0, 0, 0],
        [30, 30, -20, 10],
        [-30, -20, 20, 0],
        [90, 20, -30, 30],
    ]

    candidates = solve_inverse_kinematics(robot, target, guesses, 1e-3)
    assert len(candidates) >= 1

    validated = validate_ik_solutions(robot, target, candidates, 1e-3)
    assert any(item.validation.valid for item in validated)

def test_invalid_joint_limit_is_rejected(robot):
    target = TargetPose(np.array([0.0, 0.0, 0.0]))
    bad = IKSolution(JointState([0, 200, 0, 0]))
    result = validate_ik_solutions(robot, target, [bad])

    assert result[0].validation.joint_limit_ok is False
    assert result[0].validation.valid is False
