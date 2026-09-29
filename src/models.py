from dataclasses import dataclass, field
from typing import Any
import numpy as np

@dataclass
class JointDefinition:
    index: int
    joint_type: str
    theta: float
    d: float
    a: float
    alpha: float
    min_limit: float
    max_limit: float

@dataclass
class RobotDefinition:
    name: str
    joints: list[JointDefinition]
    base_transform: np.ndarray = field(default_factory=lambda: np.eye(4))
    end_effector_transform: np.ndarray = field(default_factory=lambda: np.eye(4))

@dataclass
class JointState:
    positions: list[float]

@dataclass
class Pose:
    position: np.ndarray
    rotation: np.ndarray

@dataclass
class FKResult:
    pose: Pose
    final_transform: np.ndarray
    transformation_chain: list[np.ndarray]

@dataclass
class TargetPose:
    position: np.ndarray
    rotation: np.ndarray | None = None

@dataclass
class IKSolution:
    joint_state: JointState
    pose: Pose | None = None
    position_error: float | None = None
    orientation_error: float | None = None
    solver_method: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidationReport:
    valid: bool
    joint_limit_ok: bool
    position_ok: bool
    orientation_ok: bool
    finite_values_ok: bool
    messages: list[str] = field(default_factory=list)

@dataclass
class ValidatedIKSolution:
    solution: IKSolution
    validation: ValidationReport
