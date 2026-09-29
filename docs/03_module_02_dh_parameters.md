# Module 2 — DH Parameter Management

## Purpose
Provide a clean, validated representation of the Denavit–Hartenberg parameters.

## Inputs
`RobotDefinition`

## Output
`DHTable`, containing:
theta, d, a, alpha for each joint.

## Standard DH equation
T(i-1,i) =
[[cos(theta), -sin(theta)cos(alpha), sin(theta)sin(alpha), a cos(theta)],
 [sin(theta), cos(theta)cos(alpha), -cos(theta)sin(alpha), a sin(theta)],
 [0, sin(alpha), cos(alpha), d],
 [0, 0, 0, 1]]

## Main interface
`get_dh_table(robot)`

## Important distinction
The DH table is the parameter description. Module 3 converts each row into a transformation matrix.

## Tests
- Number of rows equals number of joints
- Values are finite
- Angles use the documented convention
- Reference robot table matches expected values
