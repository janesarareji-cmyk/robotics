# Testing Strategy

## Why test?
Robotics software can produce plausible-looking but wrong numbers. Numerical tests are required.

## Unit tests
Test individual operations:
- DH matrix
- Joint limits
- Pose error
- Duplicate IK solutions

## Integration tests
Test the module chain:
Robot → DH → Transform → FK
and
Robot + Target → IK → Validation

## FK/IK consistency
1. Choose known joint values.
2. Run FK to create a target.
3. Give the target to IK.
4. Confirm at least one candidate is discovered.
5. Run FK on the candidate.
6. Confirm the pose error is within tolerance.

## Floating-point comparisons
Do not use exact equality for numerical matrices. Use a tolerance such as `1e-6` where appropriate.

## Required Version 1 tests
- Valid robot
- Invalid robot
- DH identity case
- Transformation chain
- Known FK case
- Reachable IK
- Unreachable IK
- Multiple candidates
- Joint-limit rejection
- Position-error rejection
- FK/IK consistency
