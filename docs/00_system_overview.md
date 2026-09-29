# System Overview

## Purpose
A modular software implementation for modelling and analysing a configurable robotic manipulator.

## End-to-end flow
Robot Configuration → DH Parameters → Transformation Matrices → Forward Kinematics → Inverse Kinematics → Solution Validation → Handoff to Module 7+

## Module boundaries

| Module | Responsibility | Input | Output |
|---|---|---|---|
| 1 | Define/validate robot | Robot configuration | `RobotDefinition` |
| 2 | Manage DH parameters | `RobotDefinition` | validated DH table |
| 3 | Calculate transforms | DH parameters | 4x4 matrices |
| 4 | Calculate FK | robot + joint state | pose + transforms |
| 5 | Find IK candidates | robot + target pose | `IKSolution[]` |
| 6 | Validate candidates | solutions + constraints | `ValidatedIKSolution[]` |

## Architecture rule
A module must not depend on another module's internal variables. It uses documented inputs and outputs only.

## Future compatibility
Later modules can add solution ranking, ideal-path selection, ghost/alternative paths, trajectory planning, 3D simulation, and collision analysis without rewriting Modules 1–6.

## Version 1 priorities
1. Correct mathematics
2. Understandability
3. Explicit data flow
4. Testability
5. Stable interfaces
6. Easy future extension
