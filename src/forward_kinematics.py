import numpy as np
from .models import RobotDefinition, JointState, Pose, FKResult
from .transform import dh_to_transform

def forward_kinematics(robot: RobotDefinition, joint_state: JointState) -> FKResult:
    if len(joint_state.positions) != len(robot.joints):
        raise ValueError("Joint-state length does not match robot DOF.")

    current = np.asarray(robot.base_transform, dtype=float).copy()
    chain = []

    for joint, q in zip(robot.joints, joint_state.positions):
        j_type = joint.joint_type.lower().strip()
        if j_type == "revolute":
            # Joint variable is theta (rotation about z_{i-1})
            transform = dh_to_transform(q, joint.d, joint.a, joint.alpha)
        elif j_type == "prismatic":
            # Joint variable is d (linear translation along z_{i-1})
            transform = dh_to_transform(joint.theta, q, joint.a, joint.alpha)
        else:
            raise ValueError(f"Unsupported joint type: {joint.joint_type}")
        current = current @ transform
        chain.append(current.copy())

    current = current @ robot.end_effector_transform

    return FKResult(
        pose=Pose(
            position=current[:3, 3].copy(),
            rotation=current[:3, :3].copy()
        ),
        final_transform=current,
        transformation_chain=chain,
    )

if __name__ == "__main__":
    from .models import JointDefinition
    from .robot_configuration import create_robot_definition

    robot = create_robot_definition(
        "4DOF Reference Robot",
        [
            JointDefinition(0, "revolute", 0, 100, 0, 90, -180, 180),
            JointDefinition(1, "revolute", 0, 0, 150, 0, -90, 90),
            JointDefinition(2, "revolute", 0, 0, 120, 0, -120, 120),
            JointDefinition(3, "revolute", 0, 0, 80, 0, -180, 180),
        ],
    )

    result = forward_kinematics(robot, JointState([30, 20, -15, 10]))
    print("End-effector position:")
    print(result.pose.position)
    print("\nFinal transformation:")
    print(result.final_transform)
