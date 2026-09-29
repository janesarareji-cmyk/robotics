# Module 1 — Robot Configuration

## Purpose
Create a valid machine-readable description of the robot.

## Inputs
- Robot name
- Joint list
- Joint type
- Joint limits
- DH values
- Optional base transform
- Optional end-effector transform

## Output
A validated `RobotDefinition`.

## Main interface
`create_robot_definition(...)`

Input:
- name
- joint definitions
- optional transforms

Output:
- `RobotDefinition`

## Validation
- At least one joint
- Supported joint type
- Minimum limit <= maximum limit
- Finite numerical values
- Correct 4x4 transforms

## Why separate this?
Changing link dimensions or joint limits should not require rewriting FK or IK.

## Tests
1. Valid robot accepted
2. Empty robot rejected
3. Unsupported joint type rejected
4. Invalid limits rejected
5. Non-finite values rejected
