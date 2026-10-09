import numpy as np
from scipy.optimize import least_squares
from .models import RobotDefinition, JointState, TargetPose, IKSolution
from .forward_kinematics import forward_kinematics

def _position_error_vector(robot, joint_values, target):
    fk = forward_kinematics(robot, JointState(list(joint_values)))
    return fk.pose.position - target.position

def solve_inverse_kinematics(robot, target, initial_guesses, position_tolerance=1e-4):
    lower = np.array([j.min_limit for j in robot.joints], dtype=float)
    upper = np.array([j.max_limit for j in robot.joints], dtype=float)
    solutions = []

    for guess in initial_guesses:
        # silently skip guesses with wrong DOF instead of crashing
        if len(guess) != len(robot.joints):
            continue

        try:
            result = least_squares(
                lambda q: _position_error_vector(robot, q, target),
                x0=np.asarray(guess, dtype=float),
                bounds=(lower, upper),
                method="trf",
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
                max_nfev=2000,
            )
        except Exception:
            continue

        state = JointState(result.x.tolist())
        fk = forward_kinematics(robot, state)
        error = float(np.linalg.norm(fk.pose.position - target.position))

        # Use error directly — scipy's result.success is unreliable for least_squares
        if np.isfinite(error) and error <= position_tolerance:
            candidate = IKSolution(
                joint_state=state,
                pose=fk.pose,
                position_error=error,
                solver_method="scipy.least_squares",
                metadata={"cost": float(result.cost)},
            )

            duplicate = any(
                np.linalg.norm(
                    np.asarray(old.joint_state.positions) -
                    np.asarray(candidate.joint_state.positions)
                ) < 0.5   # 0.5 deg/mm — filter near-duplicate configs
                for old in solutions
            )
            if not duplicate:
                solutions.append(candidate)

    return solutions
