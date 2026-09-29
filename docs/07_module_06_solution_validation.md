# Module 6 — Solution Validation

## Purpose
Determine whether IK candidates are physically and numerically acceptable.

## Inputs
- `RobotDefinition`
- `TargetPose`
- `IKSolution[]`
- Position/orientation tolerances

## Output
`ValidatedIKSolution[]`

## Checks

### Joint limits
q_min <= q_i <= q_max

### Position error
e_p = ||p_FK - p_target||

Pass when e_p <= epsilon_p.

### Orientation error
When orientation is supplied, compare calculated and target rotations using the documented tolerance.

### Finite values
Reject NaN and infinite joint values.

### Singularity information
A Jacobian-based indicator may be included as information for later ranking. Module 6 does not choose the ideal solution.

## Main interface
`validate_ik_solutions(robot, target, solutions, tolerances)`

## Example
Solution 1 → Valid
Solution 2 → Invalid: Joint 2 exceeds its limit

## Handoff
The validated solution list is the formal output from Jane's scope.
