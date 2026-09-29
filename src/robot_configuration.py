import numpy as np
from .models import JointDefinition, RobotDefinition

SUPPORTED_JOINT_TYPES = {"revolute"}

def create_robot_definition(name, joints, base_transform=None, end_effector_transform=None):
    if not name.strip():
        raise ValueError("Robot name cannot be empty.")
    if not joints:
        raise ValueError("Robot must contain at least one joint.")

    seen = set()
    for joint in joints:
        if joint.index in seen:
            raise ValueError(f"Duplicate joint index: {joint.index}")
        seen.add(joint.index)

        if joint.joint_type not in SUPPORTED_JOINT_TYPES:
            raise ValueError(f"Unsupported joint type: {joint.joint_type}")

        if joint.min_limit > joint.max_limit:
            raise ValueError(f"Invalid limits for joint {joint.index}.")

        values = [joint.theta, joint.d, joint.a, joint.alpha,
                  joint.min_limit, joint.max_limit]
        if not all(np.isfinite(v) for v in values):
            raise ValueError(f"Non-finite value found in joint {joint.index}.")

    base = np.eye(4) if base_transform is None else np.asarray(base_transform, dtype=float)
    tool = np.eye(4) if end_effector_transform is None else np.asarray(end_effector_transform, dtype=float)

    if base.shape != (4, 4):
        raise ValueError("base_transform must be 4x4.")
    if tool.shape != (4, 4):
        raise ValueError("end_effector_transform must be 4x4.")

    return RobotDefinition(
        name=name,
        joints=sorted(joints, key=lambda j: j.index),
        base_transform=base,
        end_effector_transform=tool,
    )
