# Module 5 — Inverse Kinematics

## Purpose
Find candidate joint configurations that reach a target pose.

## Inputs
- `RobotDefinition`
- `TargetPose`
- Multiple initial guesses

## Output
`IKSolution[]`

## Important requirement
Module 5 returns candidate solutions. It does NOT choose the preferred solution.

## Version 1 method
Use a numerical solver from SciPy. Run it from multiple initial guesses to discover different feasible configurations.

Conceptual flow:
Target → guess 1 → candidate 1
Target → guess 2 → candidate 2
Target → guess 3 → candidate 3
... → remove numerical duplicates → return candidates

## Numerical limitation
A numerical solver cannot guarantee that every mathematical IK solution was found. Version 1 therefore documents these as discovered candidate solutions.

## Candidate data
Each solution stores:
- joint values
- calculated pose
- position error
- orientation error where applicable
- solver method
- metadata

## Tests
- Reachable target
- Unreachable target
- Multiple initial guesses
- Duplicate removal
- FK/IK consistency
