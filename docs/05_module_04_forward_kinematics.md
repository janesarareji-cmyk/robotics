# Module 4 — Forward Kinematics

## Purpose
Calculate the end-effector pose from known joint values.

## Inputs
- `RobotDefinition`
- `JointState`

## Output
`FKResult`
- end-effector pose
- final transformation
- transformation chain

## Mathematical flow
Given q = [q1, q2, ..., qn]:

T0,n = T0,1 × T1,2 × ... × T(n-1),n

Final matrix:
T0,n = [[R, p], [0, 1]]

R = orientation
p = [x, y, z]^T = position

## Main interface
`forward_kinematics(robot, joint_state)`

## Teaching requirement
Intermediate matrices must be retained so a later interface can show:
DH → individual transforms → multiplication → final pose → extracted position/orientation.

## Tests
Compare a known configuration against an independently verified result.
