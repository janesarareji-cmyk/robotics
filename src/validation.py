import numpy as np
from .models import (
    RobotDefinition, TargetPose, IKSolution, JointState,
    ValidationReport, ValidatedIKSolution
)
from .forward_kinematics import forward_kinematics

def validate_solution(robot, target, solution, position_tolerance=1e-3,
                      orientation_tolerance=1e-3):
    messages = []
    q = np.asarray(solution.joint_state.positions, dtype=float)
    finite_ok = bool(np.all(np.isfinite(q)))

    joint_limit_ok = True
    if finite_ok:
        for value, joint in zip(q, robot.joints):
            if value < joint.min_limit or value > joint.max_limit:
                joint_limit_ok = False
                messages.append(f"Joint {joint.index + 1} is outside its allowed limits.")

    position_ok = False
    orientation_ok = True

    if finite_ok:
        fk = forward_kinematics(robot, JointState(q.tolist()))
        solution.pose = fk.pose

        solution.position_error = float(
            np.linalg.norm(fk.pose.position - target.position)
        )
        position_ok = solution.position_error <= position_tolerance

        if not position_ok:
            messages.append(
                f"Position error {solution.position_error:.6g} exceeds "
                f"tolerance {position_tolerance:.6g}."
            )

        if target.rotation is not None:
            solution.orientation_error = float(
                np.linalg.norm(fk.pose.rotation - target.rotation)
            )
            orientation_ok = solution.orientation_error <= orientation_tolerance
            if not orientation_ok:
                messages.append(
                    f"Orientation error {solution.orientation_error:.6g} exceeds "
                    f"tolerance {orientation_tolerance:.6g}."
                )

    valid = finite_ok and joint_limit_ok and position_ok and orientation_ok

    if valid:
        messages.append("Solution passed all Version 1 validation checks.")

    report = ValidationReport(
        valid=valid,
        joint_limit_ok=joint_limit_ok,
        position_ok=position_ok,
        orientation_ok=orientation_ok,
        finite_values_ok=finite_ok,
        messages=messages,
    )
    return ValidatedIKSolution(solution=solution, validation=report)

def validate_ik_solutions(robot, target, solutions, position_tolerance=1e-3,
                          orientation_tolerance=1e-3):
    return [
        validate_solution(robot, target, s, position_tolerance, orientation_tolerance)
        for s in solutions
    ]
