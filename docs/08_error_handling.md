# Error Handling

## Goal
Errors must be understandable to a beginner and useful to later modules.

## Configuration errors
- Missing joint
- Unsupported joint type
- Invalid limit
- Non-finite value

## Mathematical errors
- Solver failure
- No candidate solution
- Invalid matrix/input

## Validation errors
- Joint outside limits
- Position error too large
- Orientation error too large

## Principle
Never silently return an incorrect result.

Errors should identify:
1. What failed
2. Why it failed
3. Which input caused it
4. What can be changed

Example:
“Target could not be reached by the current robot configuration. Try moving the target closer to the robot workspace.”
