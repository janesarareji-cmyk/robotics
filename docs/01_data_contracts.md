# Data Contracts

## Why this matters
Multiple people will develop different modules. A stable contract means a developer only needs to know what a module receives and returns.

## JointDefinition
Fields:
- `index`: zero-based joint index
- `joint_type`: currently `"revolute"`
- `theta`: DH theta in degrees
- `d`: DH d parameter
- `a`: DH a parameter
- `alpha`: DH alpha in degrees
- `min_limit`: minimum joint angle in degrees
- `max_limit`: maximum joint angle in degrees

## RobotDefinition
Fields:
- `name`
- `joints`
- `base_transform`
- `end_effector_transform`

## JointState
Field:
- `positions`: list of joint values in degrees

## Pose
Fields:
- `position`: `[x, y, z]`
- `rotation`: 3x3 rotation matrix

## TargetPose
Fields:
- `position`
- optional `rotation`

## IKSolution
Fields:
- `joint_state`
- `pose`
- `position_error`
- `orientation_error`
- `solver_method`
- `metadata`

## ValidationReport
Fields:
- `valid`
- `joint_limit_ok`
- `position_ok`
- `orientation_ok`
- `finite_values_ok`
- `messages`

## ValidatedIKSolution
Fields:
- `solution`
- `validation`

## Unit convention
- Public angles: degrees
- Internal trig: radians
- Distances: consistent project unit (reference examples use millimetres)
- Transforms: homogeneous 4x4 matrices

## Module 6 handoff
Later modules should be able to use:
`solution.joint_state.positions`
`solution.validation.valid`
`solution.validation.messages`
`solution.position_error`
`solution.orientation_error`

They should not need to know how IK or validation was implemented.
