# Jane's Robotic Manipulator Project — Version 1

## Scope
Modules 1–6:
1. Robot Configuration
2. DH Parameter Management
3. Transformation Engine
4. Forward Kinematics
5. Inverse Kinematics
6. Solution Validation

The output of Module 6 is the formal handoff to the later project stages.

## Development principle
Every module has clearly defined inputs, outputs, one main responsibility, no dependency on the GUI, testable functions, and documented assumptions.

## Version 1 goal
A working, understandable implementation built in VS Code before moving to AI/Antigravity-assisted expansion.

## Software
- Python 3.x
- NumPy
- SciPy
- pytest

## Project structure
See the `docs/` and `src/` folders.

## Run
Create a virtual environment:
`python -m venv .venv`

Activate on Windows:
`.venv\Scripts\activate`

Install:
`pip install -r requirements.txt`

Run tests:
`pytest`

Run FK demonstration:
`python -m src.module04_forward_kinematics`
