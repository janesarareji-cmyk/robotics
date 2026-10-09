"""
Jane's Robotic Manipulator — Version 1
End-to-End Pipeline Runner

Flow:
    Module 1 (Robot Configuration)
    → Module 2 (DH Parameter Management)
    → Module 3 (Transformation Engine)
    → Module 4 (Forward Kinematics)
    → Module 5 (Inverse Kinematics)
    → Module 6 (Solution Validation)
    → Handoff Report

Usage:
    python -m src.pipeline
"""

import sys
import numpy as np
from .models import JointDefinition, JointState, TargetPose
from .robot_configuration import create_robot_definition
from .dh import get_dh_table
from .transform import build_transform_chain
from .forward_kinematics import forward_kinematics
from .inverse_kinematics import solve_inverse_kinematics
from .validation import validate_ik_solutions

# Ensure Unicode output works on Windows terminals (cp1252 default).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ──────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ──────────────────────────────────────────────────────────────────────────────

_SEP  = "=" * 70
_SEP2 = "-" * 70


def _header(title: str) -> None:
    print(f"\n{_SEP}")
    print(f"  {title}")
    print(_SEP)


def _section(title: str) -> None:
    print(f"\n{_SEP2}")
    print(f"  {title}")
    print(_SEP2)


def _fmt_vec(v) -> str:
    return "  ".join(f"{x:+10.4f}" for x in v)


def _fmt_matrix(m) -> str:
    rows = []
    for row in m:
        rows.append("    " + "  ".join(f"{x:+8.4f}" for x in row))
    return "\n".join(rows)


def _tick(ok: bool) -> str:
    return "OK" if ok else "FAIL"


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    robot_name: str,
    joint_definitions: list,
    demo_joint_state: list,
    ik_guesses: list,
    base_transform=None,
    end_effector_transform=None,
    position_tolerance: float = 1e-3,
    orientation_tolerance: float = 1e-3,
) -> list:
    """
    Run the full Version-1 pipeline and print a formatted report.

    Parameters
    ----------
    robot_name            : Human-readable robot name.
    joint_definitions     : List of JointDefinition objects.
    demo_joint_state      : Joint angles (degrees) used for the FK demo.
    ik_guesses            : List of initial guess vectors for IK.
    base_transform        : Optional 4x4 base frame (identity by default).
    end_effector_transform: Optional 4x4 tool frame (identity by default).
    position_tolerance    : Acceptable position error (mm or project unit).
    orientation_tolerance : Acceptable orientation error (Frobenius norm).

    Returns
    -------
    List of ValidatedIKSolution objects (Module 6 handoff).
    """

    # ── STAGE 1 — Robot Configuration ────────────────────────────────────────
    _header("STAGE 1 — Robot Configuration  (Module 1)")

    robot = create_robot_definition(
        robot_name,
        joint_definitions,
        base_transform,
        end_effector_transform,
    )

    print(f"\n  Robot : {robot.name}")
    print(f"  DOF   : {len(robot.joints)}")
    print("\n  Joint table:")
    print(f"    {'Idx':>4}  {'Type':>10}  {'theta':>9}  {'d':>8}  "
          f"{'a':>8}  {'alpha':>9}  {'Min':>8}  {'Max':>8}  {'Variable'}")
    print("    " + "-" * 82)
    for j in robot.joints:
        j_type = j.joint_type.lower().strip()
        var_desc = "theta (deg)" if j_type == "revolute" else "d (mm)"
        print(f"    {j.index:>4}  {j.joint_type:>10}  {j.theta:>9.2f}  "
              f"{j.d:>8.2f}  {j.a:>8.2f}  {j.alpha:>9.2f}  "
              f"{j.min_limit:>8.2f}  {j.max_limit:>8.2f}  [{var_desc}]")

    # ── STAGE 2 — DH Parameter Management ────────────────────────────────────
    _header("STAGE 2 — DH Parameter Management  (Module 2)")

    dh_table = get_dh_table(robot)
    print(f"\n  Retrieved {len(dh_table)} DH entries from robot definition.")
    print("  (DH table is a direct view of robot.joints — no data duplication.)")

    # ── STAGE 3 — Transformation Engine ───────────────────────────────────────
    _header("STAGE 3 — Transformation Engine  (Module 3)")

    # Override joint variable (theta for revolute, d for prismatic) with demo state
    demo_joints_overridden = []
    for j, q in zip(robot.joints, demo_joint_state):
        j_type = j.joint_type.lower().strip()
        theta_val = q if j_type == "revolute" else j.theta
        d_val = q if j_type == "prismatic" else j.d
        demo_joints_overridden.append(
            JointDefinition(
                j.index, j.joint_type, theta_val,
                d_val, j.a, j.alpha,
                j.min_limit, j.max_limit,
            )
        )
    local_transforms, cumulative_transforms = build_transform_chain(demo_joints_overridden)

    print(f"\n  Built {len(local_transforms)} local 4x4 DH transform matrices.")
    print(f"  Joint values used: {[round(q, 2) for q in demo_joint_state]}\n")
    for i, (loc, cum) in enumerate(zip(local_transforms, cumulative_transforms)):
        print(f"  Joint {i} local T:")
        print(_fmt_matrix(loc))
        print(f"\n  Joint {i} cumulative T:")
        print(_fmt_matrix(cum))
        if i < len(local_transforms) - 1:
            print()

    # ── STAGE 4 — Forward Kinematics ──────────────────────────────────────────
    _header("STAGE 4 — Forward Kinematics  (Module 4)")

    fk_state = JointState(demo_joint_state)
    fk_result = forward_kinematics(robot, fk_state)

    print(f"\n  Input joint state (degrees): {[round(q, 2) for q in demo_joint_state]}")
    print(f"\n  End-effector position  [x, y, z] (mm):")
    print(f"    {_fmt_vec(fk_result.pose.position)}")
    print(f"\n  End-effector rotation matrix (3x3):")
    print(_fmt_matrix(fk_result.pose.rotation))
    print(f"\n  Full homogeneous transform (4x4):")
    print(_fmt_matrix(fk_result.final_transform))
    print(f"\n  Transformation chain length: {len(fk_result.transformation_chain)} matrices")

    # ── STAGE 5 — Inverse Kinematics ──────────────────────────────────────────
    _header("STAGE 5 — Inverse Kinematics  (Module 5)")

    target = TargetPose(position=fk_result.pose.position.copy())
    print(f"\n  Target position (taken from FK result above, mm):")
    print(f"    {_fmt_vec(target.position)}")
    print(f"\n  Running IK with {len(ik_guesses)} initial guess(es)...")
    print(f"  Position tolerance: {position_tolerance}")

    ik_solutions = solve_inverse_kinematics(
        robot, target, ik_guesses, position_tolerance
    )

    print(f"\n  Found {len(ik_solutions)} IK candidate(s) within tolerance.\n")
    for idx, sol in enumerate(ik_solutions):
        angles = [round(q, 4) for q in sol.joint_state.positions]
        print(f"  Candidate {idx + 1}:")
        print(f"    Joint angles (deg) : {angles}")
        print(f"    Position error     : {sol.position_error:.6g}")
        print(f"    Solver method      : {sol.solver_method}")
        print(f"    Metadata           : {sol.metadata}")

    if not ik_solutions:
        print("  [WARNING] No IK solutions found within tolerance.")
        print("  Consider adding more initial guesses or relaxing position_tolerance.")

    # ── STAGE 6 — Solution Validation ─────────────────────────────────────────
    _header("STAGE 6 — Solution Validation  (Module 6)")

    validated = validate_ik_solutions(
        robot, target, ik_solutions, position_tolerance, orientation_tolerance
    )

    print(f"\n  Validated {len(validated)} candidate(s).\n")
    for idx, v in enumerate(validated):
        sol = v.solution
        rep = v.validation
        status = "VALID" if rep.valid else "INVALID"
        print(f"  ── Candidate {idx + 1} — [{status}]")
        print(f"     Joint angles (deg)    : "
              f"{[round(q, 4) for q in sol.joint_state.positions]}")
        print(f"     Position error        : {sol.position_error:.6g}")
        if sol.orientation_error is not None:
            print(f"     Orientation error     : {sol.orientation_error:.6g}")
        print(f"     Joint limits OK       : {_tick(rep.joint_limit_ok)}")
        print(f"     Position OK           : {_tick(rep.position_ok)}")
        print(f"     Orientation OK        : {_tick(rep.orientation_ok)}")
        print(f"     Finite values OK      : {_tick(rep.finite_values_ok)}")
        print(f"     Messages:")
        for msg in rep.messages:
            print(f"       - {msg}")
        print()

    # ── HANDOFF SUMMARY ────────────────────────────────────────────────────────
    _header("HANDOFF SUMMARY — Module 6 Output for Module 7+")

    valid_count = sum(1 for v in validated if v.validation.valid)
    print(f"\n  Total candidates   : {len(validated)}")
    print(f"  Valid solutions    : {valid_count}")
    print(f"  Rejected           : {len(validated) - valid_count}")

    if valid_count:
        print("\n  Valid solution(s) ready for Module 7+ consumption:")
        for idx, v in enumerate(validated):
            if v.validation.valid:
                print(f"\n  ── Solution {idx + 1}")
                print(f"     joint_state.positions  : "
                      f"{[round(q, 4) for q in v.solution.joint_state.positions]}")
                print(f"     position_error         : {v.solution.position_error:.6g}")
                print(f"     validation.valid       : {v.validation.valid}")
                print(f"     validation.messages    : {v.validation.messages}")
    else:
        print("\n  [WARNING] No valid solutions to hand off.")
        print("  Review IK guesses, joint limits, or tolerances.")

    print(f"\n{_SEP}")
    print("  Pipeline complete.")
    print(_SEP)

    return validated


# ──────────────────────────────────────────────────────────────────────────────
# Demo entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Robot definition ──────────────────────────────────────────────────────
    # 4-DOF serial revolute manipulator.
    # DH convention: theta (deg), d (mm), a (mm), alpha (deg).
    joints = [
        #              idx  type         theta    d      a  alpha   min    max
        JointDefinition(0, "revolute",    0,   100,    0,   90, -180,  180),
        JointDefinition(1, "revolute",    0,     0,  150,    0,  -90,   90),
        JointDefinition(2, "revolute",    0,     0,  120,    0, -120,  120),
        JointDefinition(3, "revolute",    0,     0,   80,    0, -180,  180),
    ]

    # ── FK demo angles ────────────────────────────────────────────────────────
    # The resulting end-effector position becomes the IK target.
    demo_angles = [30.0, 20.0, -15.0, 10.0]  # degrees

    # ── IK initial guesses ────────────────────────────────────────────────────
    # A spread of starting points increases the chance of finding multiple
    # distinct IK solutions.
    guesses = [
        [  0,   0,   0,   0],
        [ 30,  30, -20,  10],
        [-30, -20,  20,   0],
        [ 90,  20, -30,  30],
        [-90,  40,  10, -20],
        [ 45, -30,  60,  15],
    ]

    run_pipeline(
        robot_name="Jane 4-DOF Reference Robot",
        joint_definitions=joints,
        demo_joint_state=demo_angles,
        ik_guesses=guesses,
        position_tolerance=1e-3,
        orientation_tolerance=1e-3,
    )
