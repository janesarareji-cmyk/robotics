# Module 6 Handoff to Mathew

## Boundary
Jane's implementation ends at Module 6.

## Final output
`ValidatedIKSolution[]`

Each item contains:
- joint positions
- calculated pose
- position error
- orientation error
- validation status
- validation messages
- solver method/metadata

## Guarantees for valid solutions
A valid solution:
- satisfies joint limits
- has position error within tolerance
- has orientation error within tolerance when orientation is required
- contains finite numerical values

## Later work
Mathew's modules can consume these outputs to:
- rank valid solutions
- choose an ideal solution
- evaluate joint movement
- use singularity safety
- generate trajectories
- display ideal and dimmed alternative paths
- allow solution switching
- simulate motion

## Integration rule
Later modules should consume the Module 6 output instead of duplicating the IK or validation logic.

## Example
TargetPose + RobotDefinition → IK → Validation → ValidatedIKSolution[] → HANDOFF
