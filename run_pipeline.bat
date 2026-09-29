@echo off
echo Running Jane's Robotic Manipulator Pipeline...
call .venv\Scripts\activate.bat
python -m src.pipeline
pause
