"""
Jane's Robotic Manipulator — Version 1  |  GUI

A PyQt5 desktop application that wraps all 6 pipeline modules with
interactive panels and a 3D arm visualisation.

Tabs
----
1. Robot Configuration  (Module 1 + 2)
2. Forward Kinematics   (Module 3 + 4)
3. Inverse Kinematics   (Module 5)
4. Validation           (Module 6)
5. Full Pipeline        (all stages end-to-end)

Usage
-----
    python -m src.gui

Design rule
-----------
This file is GUI-only.  All mathematics live in Modules 1-6.
No existing source file is modified.
"""

import sys
import io
import traceback
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QPushButton, QLineEdit, QTextEdit, QScrollArea,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QGroupBox, QSplitter, QSizePolicy, QSpacerItem,
    QDoubleSpinBox, QMessageBox, QFrame,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d import proj3d

from .models import JointDefinition, JointState, TargetPose, IKSolution
from .robot_configuration import create_robot_definition
from .dh import get_dh_table
from .transform import build_transform_chain
from .forward_kinematics import forward_kinematics
from .inverse_kinematics import solve_inverse_kinematics
from .validation import validate_ik_solutions
from .pipeline import run_pipeline

# ──────────────────────────────────────────────────────────────────────────────
# Colour palette (dark professional)
# ──────────────────────────────────────────────────────────────────────────────

COL = {
    "bg":        "#1e1e2e",
    "panel":     "#2a2a3e",
    "border":    "#3a3a5c",
    "accent":    "#7aa2f7",
    "accent2":   "#bb9af7",
    "green":     "#9ece6a",
    "red":       "#f7768e",
    "yellow":    "#e0af68",
    "text":      "#c0caf5",
    "text_dim":  "#565f89",
    "btn":       "#3d59a1",
    "btn_hover": "#4a6ebf",
}

STYLE_SHEET = f"""
    QMainWindow, QWidget {{
        background-color: {COL['bg']};
        color: {COL['text']};
        font-family: 'Segoe UI', 'Consolas', sans-serif;
        font-size: 13px;
    }}
    QTabWidget::pane {{
        border: 1px solid {COL['border']};
        background: {COL['bg']};
        border-radius: 6px;
    }}
    QTabBar::tab {{
        background: {COL['panel']};
        color: {COL['text_dim']};
        padding: 10px 22px;
        border: 1px solid {COL['border']};
        border-bottom: none;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        font-weight: 600;
        min-width: 140px;
    }}
    QTabBar::tab:selected {{
        background: {COL['bg']};
        color: {COL['accent']};
        border-bottom: 2px solid {COL['accent']};
    }}
    QTabBar::tab:hover:!selected {{
        color: {COL['text']};
        background: {COL['border']};
    }}
    QPushButton {{
        background-color: {COL['btn']};
        color: #ffffff;
        border: none;
        border-radius: 6px;
        padding: 8px 20px;
        font-weight: 600;
        font-size: 13px;
    }}
    QPushButton:hover {{
        background-color: {COL['btn_hover']};
    }}
    QPushButton:pressed {{
        background-color: {COL['accent']};
    }}
    QPushButton#danger {{
        background-color: #6b2737;
    }}
    QPushButton#danger:hover {{
        background-color: #8b3a4a;
    }}
    QPushButton#success {{
        background-color: #2d5a27;
    }}
    QPushButton#success:hover {{
        background-color: #3a7533;
    }}
    QLineEdit, QDoubleSpinBox {{
        background-color: {COL['panel']};
        border: 1px solid {COL['border']};
        border-radius: 4px;
        padding: 5px 8px;
        color: {COL['text']};
        selection-background-color: {COL['accent']};
    }}
    QLineEdit:focus, QDoubleSpinBox:focus {{
        border: 1px solid {COL['accent']};
    }}
    QTextEdit {{
        background-color: #0d0d1a;
        border: 1px solid {COL['border']};
        border-radius: 4px;
        color: {COL['text']};
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 12px;
        padding: 6px;
    }}
    QTableWidget {{
        background-color: {COL['panel']};
        border: 1px solid {COL['border']};
        border-radius: 4px;
        gridline-color: {COL['border']};
        color: {COL['text']};
        selection-background-color: {COL['btn']};
    }}
    QTableWidget::item {{
        padding: 4px 8px;
    }}
    QHeaderView::section {{
        background-color: {COL['border']};
        color: {COL['accent']};
        padding: 6px 8px;
        border: none;
        font-weight: 700;
        font-size: 12px;
    }}
    QGroupBox {{
        border: 1px solid {COL['border']};
        border-radius: 6px;
        margin-top: 14px;
        padding-top: 8px;
        font-weight: 700;
        color: {COL['accent']};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 6px;
    }}
    QLabel#heading {{
        font-size: 16px;
        font-weight: 700;
        color: {COL['accent']};
    }}
    QLabel#subheading {{
        font-size: 12px;
        color: {COL['text_dim']};
    }}
    QLabel#ok {{
        color: {COL['green']};
        font-weight: 700;
    }}
    QLabel#fail {{
        color: {COL['red']};
        font-weight: 700;
    }}
    QLabel#value {{
        color: {COL['accent2']};
        font-family: 'Consolas', monospace;
    }}
    QScrollBar:vertical {{
        background: {COL['bg']};
        width: 10px;
        border-radius: 5px;
    }}
    QScrollBar::handle:vertical {{
        background: {COL['border']};
        border-radius: 5px;
        min-height: 20px;
    }}
    QFrame#divider {{
        background: {COL['border']};
        max-height: 1px;
    }}
"""

# ──────────────────────────────────────────────────────────────────────────────
# Shared default robot (4-DOF reference)
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_JOINTS = [
    #  idx  type        theta    d      a   alpha   min    max
    [0, "revolute",  0, 100,   0,  90, -180,  180],
    [1, "revolute",  0,   0, 150,   0,  -90,   90],
    [2, "revolute",  0,   0, 120,   0, -120,  120],
    [3, "revolute",  0,   0,  80,   0, -180,  180],
]

DEFAULT_ANGLES = [30.0, 20.0, -15.0, 10.0]

DEFAULT_GUESSES = [
    [  0,   0,   0,   0],
    [ 30,  30, -20,  10],
    [-30, -20,  20,   0],
    [ 90,  20, -30,  30],
    [-90,  40,  10, -20],
    [ 45, -30,  60,  15],
]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_divider():
    f = QFrame()
    f.setObjectName("divider")
    f.setFrameShape(QFrame.HLine)
    return f


def heading(text):
    lbl = QLabel(text)
    lbl.setObjectName("heading")
    return lbl


def subheading(text):
    lbl = QLabel(text)
    lbl.setObjectName("subheading")
    return lbl


def value_label(text="—"):
    lbl = QLabel(text)
    lbl.setObjectName("value")
    lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return lbl


def ok_label(text):
    lbl = QLabel(text)
    lbl.setObjectName("ok")
    return lbl


def fail_label(text):
    lbl = QLabel(text)
    lbl.setObjectName("fail")
    return lbl


def tick_label(ok: bool):
    return ok_label("PASS") if ok else fail_label("FAIL")


# ──────────────────────────────────────────────────────────────────────────────
# 3D Arm Rendering Primitives
# ──────────────────────────────────────────────────────────────────────────────

def _get_perp_axes(v):
    """Return two unit vectors perpendicular to v."""
    v = v / np.linalg.norm(v)
    not_v = np.array([1.0, 0.0, 0.0])
    if np.abs(np.dot(v, not_v)) > 0.98:
        not_v = np.array([0.0, 1.0, 0.0])
    n1 = np.cross(v, not_v)
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(v, n1)
    return n1, n2


def _plot_cylinder(ax, p0, p1, R, color, alpha=0.92, segments=24):
    """Draw a solid 3D cylinder between two points p0→p1."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    v = p1 - p0
    mag = np.linalg.norm(v)
    if mag < 1e-6:
        return
    v /= mag
    n1, n2 = _get_perp_axes(v)
    theta = np.linspace(0, 2 * np.pi, segments)
    t_vals = np.array([0.0, mag])
    T, TH = np.meshgrid(t_vals, theta)
    X = p0[0] + v[0]*T + R*(np.sin(TH)*n1[0] + np.cos(TH)*n2[0])
    Y = p0[1] + v[1]*T + R*(np.sin(TH)*n1[1] + np.cos(TH)*n2[1])
    Z = p0[2] + v[2]*T + R*(np.sin(TH)*n1[2] + np.cos(TH)*n2[2])
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha,
                    linewidth=0, antialiased=True, shade=True)


def _plot_disc(ax, centre, normal, R, color, alpha=0.95, segments=24):
    """Draw a flat circular disc (for caps and joint plates)."""
    normal = np.asarray(normal, float)
    normal /= np.linalg.norm(normal)
    n1, n2 = _get_perp_axes(normal)
    theta = np.linspace(0, 2 * np.pi, segments)
    r_vals = np.array([0.0, R])
    T, TH = np.meshgrid(r_vals, theta)
    X = centre[0] + T*(np.cos(TH)*n1[0] + np.sin(TH)*n2[0])
    Y = centre[1] + T*(np.cos(TH)*n1[1] + np.sin(TH)*n2[1])
    Z = centre[2] + T*(np.cos(TH)*n1[2] + np.sin(TH)*n2[2])
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha,
                    linewidth=0, antialiased=True, shade=True)


def _plot_sphere(ax, p, R, color, alpha=0.95, segments=20):
    """Draw a 3D sphere at point p."""
    u = np.linspace(0, 2 * np.pi, segments)
    v = np.linspace(0, np.pi, segments // 2)
    X = p[0] + R * np.outer(np.cos(u), np.sin(v))
    Y = p[1] + R * np.outer(np.sin(u), np.sin(v))
    Z = p[2] + R * np.outer(np.ones(segments), np.cos(v))
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha,
                    linewidth=0, antialiased=True, shade=True)


def _plot_box(ax, centre, half_dims, color, alpha=0.9):
    """Draw an axis-aligned box (used for link bodies)."""
    cx, cy, cz = centre
    hx, hy, hz = half_dims
    # 6 faces
    faces = [
        # XY faces (top/bottom)
        ([cx-hx, cx+hx, cx+hx, cx-hx], [cy-hy, cy-hy, cy+hy, cy+hy], [cz-hz]*4),
        ([cx-hx, cx+hx, cx+hx, cx-hx], [cy-hy, cy-hy, cy+hy, cy+hy], [cz+hz]*4),
        # XZ faces (front/back)
        ([cx-hx, cx+hx, cx+hx, cx-hx], [cy-hy]*4, [cz-hz, cz-hz, cz+hz, cz+hz]),
        ([cx-hx, cx+hx, cx+hx, cx-hx], [cy+hy]*4, [cz-hz, cz-hz, cz+hz, cz+hz]),
        # YZ faces (left/right)
        ([cx-hx]*4, [cy-hy, cy+hy, cy+hy, cy-hy], [cz-hz, cz-hz, cz+hz, cz+hz]),
        ([cx+hx]*4, [cy-hy, cy+hy, cy+hy, cy-hy], [cz-hz, cz-hz, cz+hz, cz+hz]),
    ]
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    polys = [list(zip(xs, ys, zs)) for xs, ys, zs in faces]
    col = Poly3DCollection(polys, alpha=alpha, linewidth=0.3,
                           edgecolor=COL["border"])
    col.set_facecolor(color)
    ax.add_collection3d(col)


def _plot_revolute_joint(ax, p, axis, R_disc, H_disc, color_body, color_ring):
    """Draw a disc-type revolute joint: two rings + a central hub."""
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    hub_h = H_disc * 0.5
    ring_h = H_disc * 0.15
    hub_r = R_disc * 0.6
    ring_r = R_disc

    # Hub cylinder
    _plot_cylinder(ax, p - hub_h*axis, p + hub_h*axis,
                   hub_r, color_body, alpha=0.95)
    # Outer rings (flanges)
    for sign in [-1, 1]:
        ring_ctr = p + sign * hub_h * axis
        _plot_cylinder(ax, ring_ctr - ring_h*axis, ring_ctr + ring_h*axis,
                       ring_r, color_ring, alpha=0.95)
        _plot_disc(ax, ring_ctr + sign*ring_h*axis, axis, ring_r,
                   color_ring, alpha=0.95)

    # Central disc cap
    _plot_disc(ax, p + hub_h*axis, axis, hub_r, color_body, alpha=0.95)
    _plot_disc(ax, p - hub_h*axis, -axis, hub_r, color_body, alpha=0.95)


def _draw_frame_arrow(ax, origin, R_mat, scale=20.0):
    """Draw XYZ frame arrows at a joint."""
    colours = ["#ff4444", "#44ff44", "#4488ff"]
    for i, col in enumerate(colours):
        d = R_mat[:, i] * scale
        ax.quiver(origin[0], origin[1], origin[2],
                  d[0], d[1], d[2],
                  color=col, linewidth=1.2, arrow_length_ratio=0.25,
                  alpha=0.75)


# ──────────────────────────────────────────────────────────────────────────────
# ArmCanvas — full industrial robotic arm 3D view
# ──────────────────────────────────────────────────────────────────────────────

class ArmCanvas(FigureCanvas):
    """Matplotlib 3D realistic robotic arm visualisation."""

    # Visual constants (mm)
    LINK_R       = 9.0    # main link cylinder radius
    JOINT_R      = 14.0   # revolute joint disc outer radius
    JOINT_H      = 18.0   # revolute joint total height
    BASE_R       = 40.0   # base pedestal radius
    BASE_H       = 30.0   # base pedestal height
    WRIST_R      = 10.0   # wrist housing radius
    GRIPPER_L    = 28.0   # gripper finger length
    GRIPPER_SEP  = 14.0   # finger separation from palm centre
    FINGER_R     = 3.5    # finger cylinder radius

    COL_LINK     = "#4e88c7"   # steel blue for links
    COL_JOINT    = "#8888bb"   # muted violet for joints
    COL_BASE     = "#3a3a5c"   # dark for base
    COL_EE       = "#9ece6a"   # green for end-effector
    COL_TARGET   = "#f7768e"   # red for target
    COL_FRAME_BG = "#2a2a3e"

    def __init__(self, parent=None, width=5, height=5, dpi=90):
        self.fig = Figure(figsize=(width, height), dpi=dpi,
                          facecolor=COL["panel"])
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self._style_axes()
        self._draw_placeholder()

    def _style_axes(self):
        ax = self.ax
        ax.set_facecolor(self.COL_FRAME_BG)
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = True
            pane.set_facecolor(self.COL_FRAME_BG)
            pane.set_edgecolor(COL["border"])
            pane.set_alpha(0.6)
        ax.tick_params(colors=COL["text_dim"], labelsize=7)
        ax.xaxis.label.set_color(COL["text_dim"])
        ax.yaxis.label.set_color(COL["text_dim"])
        ax.zaxis.label.set_color(COL["text_dim"])
        ax.set_xlabel("X (mm)", fontsize=7, labelpad=2)
        ax.set_ylabel("Y (mm)", fontsize=7, labelpad=2)
        ax.set_zlabel("Z (mm)", fontsize=7, labelpad=2)
        ax.grid(True, color=COL["border"], linewidth=0.4, alpha=0.4)
        ax.title.set_color(COL["accent"])
        ax.title.set_fontsize(9)

    def _draw_placeholder(self):
        ax = self.ax
        ax.set_title("3D Arm — run FK to visualise", color=COL["text_dim"])
        # Draw a ghost arm silhouette
        pts = np.array([[0,0,0],[0,0,50],[50,0,50],[80,0,100],[100,0,130]])
        ax.plot(pts[:,0], pts[:,1], pts[:,2], color=COL["border"],
                linewidth=1.5, linestyle="--", alpha=0.4)
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], color=COL["border"],
                   s=20, alpha=0.4)
        ax.set_xlim(-50, 250); ax.set_ylim(-150, 150); ax.set_zlim(-20, 200)
        self.draw()

    # ── main draw entry points ────────────────────────────────────────────────

    # Ghost palette — distinct translucent colours for each IK solution
    GHOST_COLOURS = [
        "#e06c75", "#e5c07b", "#98c379", "#61afef",
        "#c678dd", "#56b6c2", "#d19a66", "#abb2bf",
    ]

    def plot_arm(self, robot, fk_result, target_pos=None,
                 ghost_fk_results=None, active_ghost_idx=None):
        """
        Render the primary arm + optional ghost overlays for all IK solutions.

        Parameters
        ----------
        ghost_fk_results : list[FKResult] | None
            All IK solution FK results to draw as ghost frames.
        active_ghost_idx : int | None
            Index of the currently selected ghost (drawn more opaque).
        """
        ax = self.ax
        ax.cla()
        self._style_axes()

        # ── collect primary joint frames ──────────────────────────────────────
        frames, ee_pos, ee_rot = self._extract_frames(fk_result)

        # ── ghost overlays FIRST (so primary arm renders on top) ──────────────
        self._ghost_ee_pts  = []   # store (x,y,z, idx) for pick detection
        if ghost_fk_results:
            for g_idx, g_fk in enumerate(ghost_fk_results):
                is_active = (g_idx == active_ghost_idx)
                g_col = self.GHOST_COLOURS[g_idx % len(self.GHOST_COLOURS)]
                g_frames, g_ee, g_rot = self._extract_frames(g_fk)
                # Draw ghost arm geometry (low alpha unless active)
                alpha_mult = 1.0 if is_active else 0.30
                self._draw_arm_geometry(
                    ax, g_frames, g_ee, g_rot,
                    link_col=g_col, joint_col=g_col,
                    link_alpha=0.65 * alpha_mult,
                    joint_alpha=0.70 * alpha_mult,
                    draw_base=False, draw_gripper=False,
                    draw_frames=False,
                )
                # Ghost EE marker — clickable
                marker_s = 180 if is_active else 90
                sc = ax.scatter(
                    [g_ee[0]], [g_ee[1]], [g_ee[2]],
                    color=g_col,
                    s=marker_s,
                    marker="D" if is_active else "o",
                    zorder=9,
                    depthshade=False,
                    label=f"Sol {g_idx+1} EE  ({g_ee[0]:.1f},{g_ee[1]:.1f},{g_ee[2]:.1f})"
                          + ("  ◀ selected" if is_active else ""),
                )
                self._ghost_ee_pts.append((g_ee.copy(), g_idx))

        # ── primary arm ───────────────────────────────────────────────────────
        base_origin = frames[0][0]
        self._draw_base(ax, base_origin)
        self._draw_arm_geometry(
            ax, frames, ee_pos, ee_rot,
            link_col=self.COL_LINK, joint_col=self.COL_JOINT,
            link_alpha=0.92, joint_alpha=0.95,
            draw_base=False, draw_gripper=True, draw_frames=True,
        )

        # ── target marker ─────────────────────────────────────────────────────
        if target_pos is not None:
            self._draw_target(ax, np.asarray(target_pos, float))

        # ── legend ────────────────────────────────────────────────────────────
        ax.scatter([], [], [], color=self.COL_EE, marker="o", s=40,
                   label=f"EE  ({ee_pos[0]:.1f}, {ee_pos[1]:.1f}, {ee_pos[2]:.1f}) mm")
        if target_pos is not None:
            ax.scatter([], [], [], color=self.COL_TARGET, marker="X", s=60,
                       label=f"Target  ({target_pos[0]:.1f}, {target_pos[1]:.1f}, {target_pos[2]:.1f}) mm")
        ax.legend(loc="upper left", fontsize=7,
                  facecolor=COL["panel"], edgecolor=COL["border"],
                  labelcolor=COL["text"], framealpha=0.8)

        # ── equal-aspect 3D ───────────────────────────────────────────────────
        all_pts = np.array([f[0] for f in frames] + [ee_pos])
        self._set_equal_aspect(ax, all_pts)

        ax.set_title("Robot Arm — 3D View", color=COL["accent"])
        self.draw()

    # ── internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_frames(fk_result):
        """Return (frames, ee_pos, ee_rot) from an FKResult."""
        frames = [(np.zeros(3), np.eye(3))]
        for T in fk_result.transformation_chain:
            frames.append((T[:3, 3].copy(), T[:3, :3].copy()))
        ee_pos = np.asarray(fk_result.pose.position, float)
        ee_rot = np.asarray(fk_result.pose.rotation, float)
        return frames, ee_pos, ee_rot

    def _draw_arm_geometry(
        self, ax, frames, ee_pos, ee_rot,
        link_col, joint_col,
        link_alpha, joint_alpha,
        draw_base, draw_gripper, draw_frames,
    ):
        """Core geometry: links, joints, optional base/gripper/frames."""
        if draw_base:
            self._draw_base(ax, frames[0][0])

        for i in range(len(frames) - 1):
            p0, R0 = frames[i]
            p1, _  = frames[i + 1]
            link_vec = p1 - p0
            link_len = np.linalg.norm(link_vec)
            joint_axis = R0[:, 2]

            if link_len > 1.0:
                inset = min(self.JOINT_H * 0.5, link_len * 0.12)
                lv = link_vec / link_len
                _plot_cylinder(ax, p0 + lv*inset, p1 - lv*inset,
                               self.LINK_R, link_col, alpha=link_alpha)
                _plot_disc(ax, p0 + lv*inset, -lv, self.LINK_R,
                           "#2d4a6e", alpha=joint_alpha)
                _plot_disc(ax, p1 - lv*inset,  lv, self.LINK_R,
                           "#2d4a6e", alpha=joint_alpha)

            _plot_revolute_joint(ax, p0, joint_axis,
                                 self.JOINT_R, self.JOINT_H,
                                 joint_col, joint_col,)

        if frames:
            p_last, R_last = frames[-1]
            _plot_revolute_joint(ax, p_last, R_last[:, 2],
                                 self.JOINT_R*0.85, self.JOINT_H*0.85,
                                 joint_col, joint_col)

        wrist_len = np.linalg.norm(ee_pos - frames[-1][0])
        if wrist_len > 1.0:
            _plot_cylinder(ax, frames[-1][0], ee_pos,
                           self.WRIST_R, "#3d59a1", alpha=link_alpha)

        if draw_gripper:
            self._draw_gripper(ax, ee_pos, ee_rot)

        if draw_frames:
            scale = max(20.0, np.linalg.norm(ee_pos) * 0.07)
            for (p, R) in frames[1:]:
                _draw_frame_arrow(ax, p, R, scale=scale)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _draw_base(self, ax, origin):
        """Draw a solid cylindrical base pedestal."""
        base_bot = origin.copy()
        base_top = origin + np.array([0.0, 0.0, self.BASE_H])
        # Outer body
        _plot_cylinder(ax, base_bot, base_top,
                       self.BASE_R, self.COL_BASE, alpha=0.97)
        # Top cap
        _plot_disc(ax, base_top, np.array([0.0, 0.0, 1.0]),
                   self.BASE_R, "#555577", alpha=0.97)
        # Ground ring
        _plot_disc(ax, base_bot, np.array([0.0, 0.0, -1.0]),
                   self.BASE_R * 1.4, "#222233", alpha=0.97)
        # Inner darker hub on top
        _plot_disc(ax, base_top + np.array([0,0,1]), np.array([0,0,1]),
                   self.BASE_R * 0.55, self.COL_JOINT, alpha=0.97)

    def _draw_gripper(self, ax, ee_pos, ee_rot):
        """Draw a realistic parallel-jaw gripper at the end-effector."""
        z_vec = ee_rot[:, 2]   # approach (tool Z)
        y_vec = ee_rot[:, 1]   # finger separation (tool Y)
        x_vec = ee_rot[:, 0]   # lateral

        # Palm block — short wide cylinder
        palm_end = ee_pos + z_vec * 10.0
        _plot_cylinder(ax, ee_pos, palm_end,
                       self.WRIST_R * 1.2, self.COL_EE, alpha=0.92)
        _plot_disc(ax, palm_end, z_vec, self.WRIST_R * 1.2,
                   "#6aae46", alpha=0.92)

        # Finger roots
        f1_root = palm_end + y_vec * self.GRIPPER_SEP
        f2_root = palm_end - y_vec * self.GRIPPER_SEP
        # Finger shafts
        f1_tip = f1_root + z_vec * self.GRIPPER_L
        f2_tip = f2_root + z_vec * self.GRIPPER_L

        for froot, ftip in [(f1_root, f1_tip), (f2_root, f2_tip)]:
            _plot_cylinder(ax, froot, ftip,
                           self.FINGER_R, "#5aae3a", alpha=0.95)
            # Fingertip rounded cap
            _plot_sphere(ax, ftip, self.FINGER_R * 1.3,
                         "#7ace5a", alpha=0.93)

        # Cross-bar connecting the two finger roots
        _plot_cylinder(ax, f1_root, f2_root,
                       self.FINGER_R * 0.8, "#4a9e2a", alpha=0.90)

    def _draw_target(self, ax, pos):
        """Draw a glowing target marker with crosshair rings."""
        R = 12.0
        # Three orthogonal rings
        for normal in [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]:
            theta = np.linspace(0, 2*np.pi, 60)
            n1, n2 = _get_perp_axes(normal)
            ring_x = pos[0] + R*(np.cos(theta)*n1[0] + np.sin(theta)*n2[0])
            ring_y = pos[1] + R*(np.cos(theta)*n1[1] + np.sin(theta)*n2[1])
            ring_z = pos[2] + R*(np.cos(theta)*n1[2] + np.sin(theta)*n2[2])
            ax.plot(ring_x, ring_y, ring_z,
                    color=self.COL_TARGET, linewidth=1.5, alpha=0.85)
        # Centre dot
        ax.scatter([pos[0]], [pos[1]], [pos[2]],
                   color=self.COL_TARGET, s=60, marker="X",
                   zorder=10, depthshade=False)

    @staticmethod
    def _set_equal_aspect(ax, pts):
        """Force equal 3D aspect ratio so cylinders don't look stretched."""
        if len(pts) == 0:
            return
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        ranges = maxs - mins
        max_r = max(ranges.max() / 2.0, 50.0)
        mids = (mins + maxs) / 2.0
        ax.set_xlim(mids[0] - max_r, mids[0] + max_r)
        ax.set_ylim(mids[1] - max_r, mids[1] + max_r)
        ax.set_zlim(mids[2] - max_r, mids[2] + max_r)
        try:
            ax.set_box_aspect((1, 1, 1))
        except AttributeError:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# IK Worker thread (keeps UI responsive)
# ──────────────────────────────────────────────────────────────────────────────

class IKWorker(QThread):
    finished = pyqtSignal(list, str)   # solutions, error_msg

    def __init__(self, robot, target, guesses, tol):
        super().__init__()
        self.robot   = robot
        self.target  = target
        self.guesses = guesses
        self.tol     = tol

    def run(self):
        try:
            sols = solve_inverse_kinematics(
                self.robot, self.target, self.guesses, self.tol
            )
            self.finished.emit(sols, "")
        except Exception as e:
            self.finished.emit([], traceback.format_exc())


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — Robot Configuration
# ──────────────────────────────────────────────────────────────────────────────

class RobotConfigTab(QWidget):
    robot_changed = pyqtSignal(object)   # emits RobotDefinition

    COLS = ["Index", "Type", "theta (deg)", "d", "a", "alpha (deg)",
            "Min (deg)", "Max (deg)"]

    def __init__(self):
        super().__init__()
        self.robot = None
        self._build_ui()
        self._populate_defaults()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        layout.addWidget(heading("Module 1 — Robot Configuration"))
        layout.addWidget(subheading(
            "Define the robot's DH parameters and joint limits. "
            "Click Validate to build the RobotDefinition."))
        layout.addWidget(make_divider())

        # Joint table
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setMinimumHeight(180)
        layout.addWidget(self.table)

        # Buttons row
        btn_row = QHBoxLayout()
        self.btn_add    = QPushButton("+ Add Joint")
        self.btn_remove = QPushButton("Remove Last")
        self.btn_remove.setObjectName("danger")
        self.btn_validate = QPushButton("Validate Robot")
        self.btn_validate.setObjectName("success")
        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_remove)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_validate)
        layout.addLayout(btn_row)

        # Result group
        grp = QGroupBox("Validation Result  (Module 1 + 2)")
        grp_lay = QVBoxLayout(grp)

        self.lbl_result = QLabel("Not validated yet.")
        self.lbl_result.setWordWrap(True)
        grp_lay.addWidget(self.lbl_result)

        self.txt_dh = QTextEdit()
        self.txt_dh.setReadOnly(True)
        self.txt_dh.setFixedHeight(160)
        self.txt_dh.setPlaceholderText("DH table will appear here after validation...")
        grp_lay.addWidget(QLabel("DH Table (Module 2):"))
        grp_lay.addWidget(self.txt_dh)
        layout.addWidget(grp)
        layout.addStretch()

        # Signals
        self.btn_add.clicked.connect(self._add_row)
        self.btn_remove.clicked.connect(self._remove_last)
        self.btn_validate.clicked.connect(self._validate)

    def _populate_defaults(self):
        for row_data in DEFAULT_JOINTS:
            self._add_row(row_data)

    def _add_row(self, data=None):
        r = self.table.rowCount()
        self.table.insertRow(r)
        defaults = [str(r), "revolute", "0", "0", "0", "0", "-180", "180"]
        if data:
            defaults = [str(v) for v in data]
        for col, val in enumerate(defaults):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(r, col, item)

    def _remove_last(self):
        r = self.table.rowCount()
        if r > 0:
            self.table.removeRow(r - 1)

    def _validate(self):
        try:
            joints = self._parse_joints()
            robot = create_robot_definition("Jane 4-DOF Reference Robot", joints)
            self.robot = robot

            # Success feedback
            self.lbl_result.setText(
                f"<b style='color:{COL['green']}'>✔ Valid</b> &nbsp; "
                f"Robot: <b>{robot.name}</b> &nbsp;|&nbsp; DOF: <b>{len(robot.joints)}</b>"
            )

            # DH table
            dh = get_dh_table(robot)
            lines = [
                f"{'Idx':>4}  {'Type':>10}  {'theta':>8}  {'d':>8}  "
                f"{'a':>8}  {'alpha':>8}  {'min':>8}  {'max':>8}",
                "-" * 72,
            ]
            for j in dh:
                lines.append(
                    f"{j.index:>4}  {j.joint_type:>10}  {j.theta:>8.2f}  "
                    f"{j.d:>8.2f}  {j.a:>8.2f}  {j.alpha:>8.2f}  "
                    f"{j.min_limit:>8.2f}  {j.max_limit:>8.2f}"
                )
            self.txt_dh.setPlainText("\n".join(lines))

            self.robot_changed.emit(robot)
        except Exception as e:
            self.lbl_result.setText(
                f"<b style='color:{COL['red']}'>✘ Error:</b> {e}"
            )
            self.robot = None

    def _parse_joints(self):
        joints = []
        for r in range(self.table.rowCount()):
            def cell(c):
                item = self.table.item(r, c)
                return item.text().strip() if item else ""
            joints.append(JointDefinition(
                index=int(cell(0)),
                joint_type=cell(1),
                theta=float(cell(2)),
                d=float(cell(3)),
                a=float(cell(4)),
                alpha=float(cell(5)),
                min_limit=float(cell(6)),
                max_limit=float(cell(7)),
            ))
        return joints

    def get_robot(self):
        return self.robot


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — Forward Kinematics
# ──────────────────────────────────────────────────────────────────────────────

class FKTab(QWidget):
    fk_done = pyqtSignal(object, object)   # robot, FKResult

    def __init__(self, get_robot_fn):
        super().__init__()
        self.get_robot = get_robot_fn
        self.last_fk = None
        self.last_robot = None
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # ── Left panel ────────────────────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(12)

        left.addWidget(heading("Module 4 — Forward Kinematics"))
        left.addWidget(subheading(
            "Enter joint angles (degrees) and click Run FK."
        ))
        left.addWidget(make_divider())

        # Joint angle inputs
        self.angle_inputs = []
        grp_angles = QGroupBox("Joint Angles (degrees)")
        grp_lay = QFormLayout(grp_angles)
        for i, default_angle in enumerate(DEFAULT_ANGLES):
            spin = QDoubleSpinBox()
            spin.setRange(-360, 360)
            spin.setDecimals(2)
            spin.setSingleStep(1.0)
            spin.setValue(default_angle)
            self.angle_inputs.append(spin)
            grp_lay.addRow(f"Joint {i} (q{i}):", spin)
        left.addWidget(grp_angles)

        self.btn_fk = QPushButton("Run FK  (Module 3 + 4)")
        self.btn_fk.setObjectName("success")
        left.addWidget(self.btn_fk)

        # Results
        grp_result = QGroupBox("End-Effector Result")
        res_lay = QFormLayout(grp_result)

        self.lbl_pos = value_label()
        self.lbl_err = QLabel()
        res_lay.addRow("Position [x, y, z] (mm):", self.lbl_pos)

        self.lbl_rot = QTextEdit()
        self.lbl_rot.setReadOnly(True)
        self.lbl_rot.setFixedHeight(100)
        self.lbl_rot.setPlaceholderText("Rotation matrix...")
        res_lay.addRow("Rotation matrix (3x3):", self.lbl_rot)

        self.lbl_t44 = QTextEdit()
        self.lbl_t44.setReadOnly(True)
        self.lbl_t44.setFixedHeight(120)
        self.lbl_t44.setPlaceholderText("Full 4x4 transform...")
        res_lay.addRow("Full transform (4x4):", self.lbl_t44)

        left.addWidget(grp_result)
        left.addStretch()

        # ── Right panel (3D plot) ──────────────────────────────────
        right = QVBoxLayout()
        right.addWidget(heading("3D Visualisation"))
        right.addWidget(subheading("Stick-figure arm from base to end-effector"))
        self.canvas = ArmCanvas()
        right.addWidget(self.canvas)

        # Assemble splitter
        left_w = QWidget()
        left_w.setLayout(left)
        right_w = QWidget()
        right_w.setLayout(right)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_w)
        splitter.addWidget(right_w)
        splitter.setSizes([420, 480])
        layout.addWidget(splitter)

        self.btn_fk.clicked.connect(self._run_fk)

    def _run_fk(self):
        robot = self.get_robot()
        if robot is None:
            QMessageBox.warning(self, "No Robot",
                "Please validate the robot in Tab 1 first.")
            return

        # Resize angle inputs to match DOF
        n = len(robot.joints)
        angles = []
        for i in range(n):
            if i < len(self.angle_inputs):
                angles.append(self.angle_inputs[i].value())
            else:
                angles.append(0.0)

        try:
            state = JointState(angles)
            result = forward_kinematics(robot, state)
            self.last_fk = result
            self.last_robot = robot

            pos = result.pose.position
            self.lbl_pos.setText(
                f"x={pos[0]:+.4f}   y={pos[1]:+.4f}   z={pos[2]:+.4f}"
            )

            rot = result.pose.rotation
            rot_lines = []
            for row in rot:
                rot_lines.append("  ".join(f"{v:+8.4f}" for v in row))
            self.lbl_rot.setPlainText("\n".join(rot_lines))

            t44 = result.final_transform
            t_lines = []
            for row in t44:
                t_lines.append("  ".join(f"{v:+8.4f}" for v in row))
            self.lbl_t44.setPlainText("\n".join(t_lines))

            self.canvas.plot_arm(robot, result)
            self.fk_done.emit(robot, result)
        except Exception as e:
            QMessageBox.critical(self, "FK Error", traceback.format_exc())

    def get_fk_result(self):
        return self.last_robot, self.last_fk


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — Inverse Kinematics
# ──────────────────────────────────────────────────────────────────────────────

class IKTab(QWidget):
    ik_done = pyqtSignal(object, object, list)  # robot, target, solutions

    def __init__(self, get_robot_fn, get_fk_fn):
        super().__init__()
        self.get_robot      = get_robot_fn
        self.get_fk         = get_fk_fn
        self.solutions      = []
        self.worker         = None
        self._ghost_fk      = []       # list[FKResult] for all solutions
        self._active_idx    = None     # currently highlighted ghost index
        self._ghosts_on     = True     # toggle state
        self._robot_ref     = None     # last robot used for IK
        self._target_ref    = None     # last target vector
        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        left = QVBoxLayout()
        left.setSpacing(12)

        left.addWidget(heading("Module 5 — Inverse Kinematics"))
        left.addWidget(subheading(
            "Set target position and initial guesses, then solve."
        ))
        left.addWidget(make_divider())

        # Target position
        grp_target = QGroupBox("Target Position (mm)")
        t_lay = QFormLayout(grp_target)
        self.spin_tx = QDoubleSpinBox()
        self.spin_ty = QDoubleSpinBox()
        self.spin_tz = QDoubleSpinBox()
        for s in [self.spin_tx, self.spin_ty, self.spin_tz]:
            s.setRange(-2000, 2000)
            s.setDecimals(4)
        self.spin_tx.setValue(200.0)
        self.spin_ty.setValue(100.0)
        self.spin_tz.setValue(150.0)
        t_lay.addRow("x:", self.spin_tx)
        t_lay.addRow("y:", self.spin_ty)
        t_lay.addRow("z:", self.spin_tz)
        self.btn_from_fk = QPushButton("Fill from FK Result")
        t_lay.addRow("", self.btn_from_fk)
        left.addWidget(grp_target)

        # Tolerance
        grp_tol = QGroupBox("Position Tolerance")
        tol_lay = QFormLayout(grp_tol)
        self.spin_tol = QDoubleSpinBox()
        self.spin_tol.setRange(1e-10, 100)
        self.spin_tol.setDecimals(6)
        self.spin_tol.setValue(1e-3)
        tol_lay.addRow("Tolerance:", self.spin_tol)
        left.addWidget(grp_tol)

        # Guesses table
        grp_guesses = QGroupBox("Initial Guesses")
        g_lay = QVBoxLayout(grp_guesses)
        self.guess_table = QTableWidget(0, 4)
        self.guess_table.setHorizontalHeaderLabels(["q0", "q1", "q2", "q3"])
        self.guess_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.guess_table.setFixedHeight(180)
        for g in DEFAULT_GUESSES:
            self._add_guess_row(g)
        g_row = QHBoxLayout()
        btn_ag = QPushButton("+ Add Guess")
        btn_rg = QPushButton("Remove Last")
        btn_rg.setObjectName("danger")
        btn_ag.clicked.connect(lambda: self._add_guess_row())
        btn_rg.clicked.connect(self._remove_guess)
        g_row.addWidget(btn_ag); g_row.addWidget(btn_rg); g_row.addStretch()
        g_lay.addWidget(self.guess_table)
        g_lay.addLayout(g_row)
        left.addWidget(grp_guesses)

        self.btn_ik = QPushButton("Solve IK  (Module 5)")
        self.btn_ik.setObjectName("success")
        left.addWidget(self.btn_ik)
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        left.addWidget(self.lbl_status)
        left.addStretch()

        # ── Right panel ───────────────────────────────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(8)

        # Heading + ghost toggle on same row
        top_row = QHBoxLayout()
        top_row.addWidget(heading("IK Candidates"))
        top_row.addStretch()
        ghost_lbl = QLabel("Ghost frames:")
        ghost_lbl.setStyleSheet(f"color:{COL['text_dim']}; font-size:12px;")
        top_row.addWidget(ghost_lbl)
        self.btn_ghost = QPushButton("ON")
        self.btn_ghost.setCheckable(True)
        self.btn_ghost.setChecked(True)
        self.btn_ghost.setFixedWidth(52)
        self.btn_ghost.setStyleSheet(
            f"""
            QPushButton {{
                background: {COL['green']}; color:#111; border-radius:4px;
                font-weight:700; font-size:12px; padding:4px 8px;
            }}
            QPushButton:!checked {{
                background: {COL['text_dim']}; color:#eee;
            }}
            """
        )
        self.btn_ghost.toggled.connect(self._on_ghost_toggle)
        top_row.addWidget(self.btn_ghost)
        right.addLayout(top_row)

        right.addWidget(subheading(
            "Click a row or a ghost EE marker in the 3D view to select a solution"
        ))

        self.result_table = QTableWidget(0, 6)
        self.result_table.setHorizontalHeaderLabels(
            ["#", "q0", "q1", "q2", "q3", "Pos Error"]
        )
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.result_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.result_table.setSelectionMode(QTableWidget.SingleSelection)
        self.result_table.setMaximumHeight(160)
        right.addWidget(self.result_table)

        self.canvas = ArmCanvas()
        right.addWidget(self.canvas)

        # Connect signals
        self.result_table.itemSelectionChanged.connect(self._on_table_selected)
        self.canvas.mpl_connect("pick_event",   self._on_canvas_pick)
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)

        left_w = QWidget(); left_w.setLayout(left)
        right_w = QWidget(); right_w.setLayout(right)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_w)
        splitter.addWidget(right_w)
        splitter.setSizes([440, 460])
        layout.addWidget(splitter)

        self.btn_from_fk.clicked.connect(self._fill_from_fk)
        self.btn_ik.clicked.connect(self._solve)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _add_guess_row(self, data=None):
        r = self.guess_table.rowCount()
        self.guess_table.insertRow(r)
        vals = data if data else [0, 0, 0, 0]
        for c, v in enumerate(vals[:self.guess_table.columnCount()]):
            item = QTableWidgetItem(str(float(v)))
            item.setTextAlignment(Qt.AlignCenter)
            self.guess_table.setItem(r, c, item)

    def _remove_guess(self):
        r = self.guess_table.rowCount()
        if r > 0:
            self.guess_table.removeRow(r - 1)

    def _fill_from_fk(self):
        robot, fk = self.get_fk()
        if fk is None:
            QMessageBox.warning(self, "No FK", "Run FK in Tab 2 first.")
            return
        pos = fk.pose.position
        self.spin_tx.setValue(float(pos[0]))
        self.spin_ty.setValue(float(pos[1]))
        self.spin_tz.setValue(float(pos[2]))

    def _parse_guesses(self):
        guesses = []
        for r in range(self.guess_table.rowCount()):
            row = []
            for c in range(self.guess_table.columnCount()):
                item = self.guess_table.item(r, c)
                row.append(float(item.text()) if item else 0.0)
            guesses.append(row)
        return guesses

    def _get_target_vec(self):
        return np.array([self.spin_tx.value(),
                         self.spin_ty.value(),
                         self.spin_tz.value()])

    # ── IK solve ──────────────────────────────────────────────────────────────

    def _solve(self):
        robot = self.get_robot()
        if robot is None:
            QMessageBox.warning(self, "No Robot",
                "Validate robot in Tab 1 first.")
            return
        try:
            target = TargetPose(position=self._get_target_vec())
            guesses = self._parse_guesses()
            tol = self.spin_tol.value()
        except ValueError:
            QMessageBox.warning(self, "Invalid Input",
                "Ensure all guesses are valid numbers.")
            return

        self.btn_ik.setEnabled(False)
        self.lbl_status.setText("Solving IK...")
        self.worker = IKWorker(robot, target, guesses, tol)
        self.worker.finished.connect(
            lambda sols, err: self._on_ik_done(robot, target, sols, err)
        )
        self.worker.start()

    def _on_ik_done(self, robot, target, solutions, err):
        self.btn_ik.setEnabled(True)
        if err:
            self.lbl_status.setText(
                f"<b style='color:{COL['red']}'>Error:</b> {err[:200]}"
            )
            return

        self.solutions   = solutions
        self._robot_ref  = robot
        self._target_ref = self._get_target_vec()
        self._active_idx = 0 if solutions else None

        n = len(solutions)
        color = COL["green"] if n > 0 else COL["yellow"]
        self.lbl_status.setText(
            f"<b style='color:{color}'>Found {n} candidate(s)</b> "
            f"within tolerance {self.spin_tol.value():.1e}")

        # Populate table
        self.result_table.blockSignals(True)
        self.result_table.setRowCount(0)
        n_joints = len(robot.joints)
        cols = ["#"] + [f"q{i}" for i in range(n_joints)] + ["Pos Error"]
        self.result_table.setColumnCount(len(cols))
        self.result_table.setHorizontalHeaderLabels(cols)
        for idx, sol in enumerate(solutions):
            r = self.result_table.rowCount()
            self.result_table.insertRow(r)
            g_col = ArmCanvas.GHOST_COLOURS[idx % len(ArmCanvas.GHOST_COLOURS)]
            values = [str(idx + 1)] + \
                     [f"{q:.4f}" for q in sol.joint_state.positions] + \
                     [f"{sol.position_error:.4e}"]
            for c, v in enumerate(values):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignCenter)
                # colour-code each row to match its ghost
                item.setForeground(QColor(g_col))
                self.result_table.setItem(r, c, item)
        self.result_table.blockSignals(False)

        # Pre-compute ghost FK results
        self._ghost_fk = []
        for sol in solutions:
            try:
                self._ghost_fk.append(forward_kinematics(robot, sol.joint_state))
            except Exception:
                self._ghost_fk.append(None)

        # Select first row and render
        if solutions:
            self.result_table.selectRow(0)
        else:
            self._redraw()

        self.ik_done.emit(robot, target, solutions)

    # ── selection / render ────────────────────────────────────────────────────

    def _select_solution(self, idx):
        """Set active ghost index, update table highlight and redraw."""
        if idx is None or idx >= len(self.solutions):
            return
        self._active_idx = idx
        # Update table without re-triggering signal
        self.result_table.blockSignals(True)
        self.result_table.selectRow(idx)
        self.result_table.blockSignals(False)
        self._redraw()

    def _redraw(self):
        """Render primary arm (best solution) + optional ghost overlays."""
        robot = self._robot_ref
        if robot is None or not self.solutions:
            return
        idx = self._active_idx if self._active_idx is not None else 0
        if idx >= len(self.solutions):
            return

        try:
            primary_fk = forward_kinematics(
                robot, self.solutions[idx].joint_state)
        except Exception:
            return

        ghosts = None
        if self._ghosts_on and self._ghost_fk:
            ghosts = [g for g in self._ghost_fk if g is not None]

        self.canvas.plot_arm(
            robot, primary_fk,
            target_pos=self._target_ref,
            ghost_fk_results=ghosts,
            active_ghost_idx=self._active_idx,
        )

    def _on_table_selected(self):
        items = self.result_table.selectedItems()
        if not items:
            return
        row = items[0].row()
        if row == self._active_idx:
            return
        self._active_idx = row
        self._redraw()

    def _on_ghost_toggle(self, checked):
        self.btn_ghost.setText("ON" if checked else "OFF")
        self._ghosts_on = checked
        self._redraw()

    def _on_canvas_click(self, event):
        """Select ghost frame by clicking near its EE marker in the 3D scene."""
        if not self._ghosts_on or not getattr(self.canvas, '_ghost_ee_pts', []):
            return
        if event.inaxes is None:
            return
        ax = self.canvas.ax
        best_idx, best_dist = None, 50.0   # 50px threshold
        for (pt, g_idx) in self.canvas._ghost_ee_pts:
            try:
                x2d, y2d, _ = proj3d.proj_transform(
                    pt[0], pt[1], pt[2], ax.get_proj()
                )
                # Convert from NDC [-1,1] to display pixels
                fig_w, fig_h = (self.canvas.fig.get_size_inches()
                                * self.canvas.fig.get_dpi())
                px = (x2d + 1) / 2 * fig_w
                py = (1 - (y2d + 1) / 2) * fig_h
                dist = ((event.x - px)**2 + (event.y - py)**2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_idx  = g_idx
            except Exception:
                pass
        if best_idx is not None:
            self._select_solution(best_idx)

    def _on_canvas_pick(self, event):
        """Fallback: matplotlib pick event."""
        pass

    def get_solutions(self):
        return self.solutions


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — Validation
# ──────────────────────────────────────────────────────────────────────────────

class ValidationTab(QWidget):

    def __init__(self, get_robot_fn):
        super().__init__()
        self.get_robot = get_robot_fn
        self._robot = None
        self._target = None
        self._solutions = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        layout.addWidget(heading("Module 6 — Solution Validation"))
        layout.addWidget(subheading(
            "Automatically populated after IK runs in Tab 3. "
            "Click Validate to re-run with custom tolerances."
        ))
        layout.addWidget(make_divider())

        # Tolerances
        tol_row = QHBoxLayout()
        tol_row.addWidget(QLabel("Position tolerance:"))
        self.spin_pos_tol = QDoubleSpinBox()
        self.spin_pos_tol.setRange(1e-10, 100)
        self.spin_pos_tol.setDecimals(6)
        self.spin_pos_tol.setValue(1e-3)
        tol_row.addWidget(self.spin_pos_tol)
        tol_row.addSpacing(20)
        tol_row.addWidget(QLabel("Orientation tolerance:"))
        self.spin_ori_tol = QDoubleSpinBox()
        self.spin_ori_tol.setRange(1e-10, 100)
        self.spin_ori_tol.setDecimals(6)
        self.spin_ori_tol.setValue(1e-3)
        tol_row.addWidget(self.spin_ori_tol)
        tol_row.addStretch()
        self.btn_validate = QPushButton("Validate  (Module 6)")
        self.btn_validate.setObjectName("success")
        tol_row.addWidget(self.btn_validate)
        layout.addLayout(tol_row)

        # Summary label
        self.lbl_summary = QLabel("Run IK in Tab 3 to populate solutions.")
        self.lbl_summary.setWordWrap(True)
        layout.addWidget(self.lbl_summary)

        # Scrollable results area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.results_widget = QWidget()
        self.results_layout = QVBoxLayout(self.results_widget)
        self.results_layout.setSpacing(12)
        self.results_layout.addStretch()
        scroll.setWidget(self.results_widget)
        layout.addWidget(scroll)

        self.btn_validate.clicked.connect(self._validate)

    def receive_ik(self, robot, target, solutions):
        """Called when IK finishes in Tab 3."""
        self._robot = robot
        self._target = target
        self._solutions = solutions
        self._validate()

    def _validate(self):
        robot = self._robot or self.get_robot()
        if robot is None:
            self.lbl_summary.setText(
                "No robot defined. Validate in Tab 1 first."
            )
            return
        if not self._solutions:
            self.lbl_summary.setText(
                "No IK solutions available. Run IK in Tab 3 first."
            )
            return
        if self._target is None:
            return

        pos_tol = self.spin_pos_tol.value()
        ori_tol = self.spin_ori_tol.value()

        validated = validate_ik_solutions(
            robot, self._target, self._solutions, pos_tol, ori_tol
        )

        valid_count = sum(1 for v in validated if v.validation.valid)
        color = COL["green"] if valid_count == len(validated) else COL["yellow"]
        self.lbl_summary.setText(
            f"<b style='color:{color}'>{valid_count} / {len(validated)} "
            f"solutions VALID</b>"
        )

        # Clear old results
        while self.results_layout.count() > 1:
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for idx, v in enumerate(validated):
            grp = QGroupBox(
                f"Candidate {idx + 1}  —  "
                f"{'VALID' if v.validation.valid else 'INVALID'}"
            )
            grp.setStyleSheet(
                f"QGroupBox {{ border-color: "
                f"{'#9ece6a' if v.validation.valid else '#f7768e'}; }}"
                f"QGroupBox::title {{ color: "
                f"{'#9ece6a' if v.validation.valid else '#f7768e'}; }}"
            )
            g_lay = QGridLayout(grp)

            sol = v.solution
            rep = v.validation

            angles_str = ", ".join(
                f"{q:.4f}" for q in sol.joint_state.positions
            )
            g_lay.addWidget(QLabel("Joint angles (deg):"), 0, 0)
            g_lay.addWidget(value_label(f"[{angles_str}]"), 0, 1)

            g_lay.addWidget(QLabel("Position error:"), 1, 0)
            g_lay.addWidget(
                value_label(f"{sol.position_error:.6g}"), 1, 1
            )

            checks = [
                ("Joint limits:",   rep.joint_limit_ok),
                ("Position:",       rep.position_ok),
                ("Orientation:",    rep.orientation_ok),
                ("Finite values:",  rep.finite_values_ok),
            ]
            for i, (lbl_txt, ok) in enumerate(checks):
                g_lay.addWidget(QLabel(lbl_txt), i, 2)
                g_lay.addWidget(tick_label(ok), i, 3)

            msgs = "  |  ".join(rep.messages)
            g_lay.addWidget(QLabel(f"Messages: {msgs}"), 4, 0, 1, 4)

            self.results_layout.insertWidget(
                self.results_layout.count() - 1, grp
            )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 5 — Full Pipeline
# ──────────────────────────────────────────────────────────────────────────────

class FullPipelineTab(QWidget):

    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # Left: config + controls
        left = QVBoxLayout()
        left.setSpacing(12)

        left.addWidget(heading("Full Pipeline  (Modules 1 – 6)"))
        left.addWidget(subheading(
            "Runs the complete end-to-end flow and displays the full report."
        ))
        left.addWidget(make_divider())

        # Joint angle inputs (replicates FK demo)
        grp_angles = QGroupBox("FK Demo Joint Angles (degrees)")
        a_lay = QFormLayout(grp_angles)
        self.angle_spins = []
        for i, a in enumerate(DEFAULT_ANGLES):
            s = QDoubleSpinBox()
            s.setRange(-360, 360)
            s.setDecimals(2)
            s.setSingleStep(1.0)
            s.setValue(a)
            self.angle_spins.append(s)
            a_lay.addRow(f"q{i}:", s)
        left.addWidget(grp_angles)

        # Tolerances
        grp_tol = QGroupBox("Tolerances")
        tol_lay = QFormLayout(grp_tol)
        self.spin_pos_tol = QDoubleSpinBox()
        self.spin_pos_tol.setRange(1e-10, 100)
        self.spin_pos_tol.setDecimals(6)
        self.spin_pos_tol.setValue(1e-3)
        self.spin_ori_tol = QDoubleSpinBox()
        self.spin_ori_tol.setRange(1e-10, 100)
        self.spin_ori_tol.setDecimals(6)
        self.spin_ori_tol.setValue(1e-3)
        tol_lay.addRow("Position:", self.spin_pos_tol)
        tol_lay.addRow("Orientation:", self.spin_ori_tol)
        left.addWidget(grp_tol)

        self.btn_run = QPushButton("Run Full Pipeline")
        self.btn_run.setObjectName("success")
        left.addWidget(self.btn_run)

        # Summary labels
        self.lbl_ee   = value_label()
        self.lbl_sols = value_label()
        grp_sum = QGroupBox("Quick Summary")
        s_lay = QFormLayout(grp_sum)
        s_lay.addRow("End-effector position:", self.lbl_ee)
        s_lay.addRow("Valid solutions:",       self.lbl_sols)
        left.addWidget(grp_sum)
        left.addStretch()

        # Right: text output + 3D plot
        right = QVBoxLayout()
        right.setSpacing(10)

        right.addWidget(heading("Pipeline Output"))
        self.txt_output = QTextEdit()
        self.txt_output.setReadOnly(True)
        self.txt_output.setMinimumWidth(480)
        right.addWidget(self.txt_output, stretch=2)

        right.addWidget(make_divider())
        right.addWidget(subheading("3D Arm at FK Demo State"))
        self.canvas = ArmCanvas(width=4, height=3, dpi=80)
        right.addWidget(self.canvas, stretch=1)

        left_w = QWidget(); left_w.setLayout(left)
        right_w = QWidget(); right_w.setLayout(right)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_w)
        splitter.addWidget(right_w)
        splitter.setSizes([300, 620])
        layout.addWidget(splitter)

        self.btn_run.clicked.connect(self._run)

    def _run(self):
        # Capture stdout from pipeline.run_pipeline
        joints = [
            JointDefinition(
                index=i,
                joint_type="revolute",
                theta=0,
                d=[100, 0, 0, 0][i],
                a=[0, 150, 120, 80][i],
                alpha=[90, 0, 0, 0][i],
                min_limit=[-180, -90, -120, -180][i],
                max_limit=[180, 90, 120, 180][i],
            )
            for i in range(4)
        ]

        demo_angles = [s.value() for s in self.angle_spins]
        guesses = DEFAULT_GUESSES
        pos_tol = self.spin_pos_tol.value()
        ori_tol = self.spin_ori_tol.value()

        # Redirect stdout
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        validated = []
        try:
            validated = run_pipeline(
                robot_name="Jane 4-DOF Reference Robot",
                joint_definitions=joints,
                demo_joint_state=demo_angles,
                ik_guesses=guesses,
                position_tolerance=pos_tol,
                orientation_tolerance=ori_tol,
            )
        except Exception:
            buf.write(traceback.format_exc())
        finally:
            sys.stdout = old_stdout

        self.txt_output.setPlainText(buf.getvalue())
        # Scroll to top
        self.txt_output.moveCursor(self.txt_output.textCursor().Start)

        # Quick summary + 3D plot
        valid_n = sum(1 for v in validated if v.validation.valid)
        color = COL["green"] if valid_n > 0 else COL["red"]
        self.lbl_sols.setText(
            f"<b style='color:{color}'>{valid_n} / {len(validated)}</b>"
        )

        # Rebuild robot for plot
        try:
            robot = create_robot_definition(
                "Jane 4-DOF Reference Robot", joints
            )
            state  = JointState(demo_angles)
            result = forward_kinematics(robot, state)
            pos = result.pose.position
            self.lbl_ee.setText(
                f"x={pos[0]:+.3f}   y={pos[1]:+.3f}   z={pos[2]:+.3f}"
            )
            # target is the FK position itself
            self.canvas.plot_arm(robot, result, target_pos=pos)
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Main Window
# ──────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "Jane's Robotic Manipulator  —  Version 1"
        )
        self.resize(1280, 820)
        self._build_ui()

    def _build_ui(self):
        tabs = QTabWidget()
        tabs.setDocumentMode(True)

        # Instantiate tabs
        self.tab_config  = RobotConfigTab()
        self.tab_fk      = FKTab(self.tab_config.get_robot)
        self.tab_ik      = IKTab(self.tab_config.get_robot,
                                  self.tab_fk.get_fk_result)
        self.tab_val     = ValidationTab(self.tab_config.get_robot)
        self.tab_pipe    = FullPipelineTab()

        # Wire cross-tab signals
        self.tab_config.robot_changed.connect(
            lambda _: None   # future: propagate to FK/IK spinbox counts
        )
        self.tab_ik.ik_done.connect(self.tab_val.receive_ik)

        tabs.addTab(self.tab_config, "1  Robot Config")
        tabs.addTab(self.tab_fk,     "2  Forward Kinematics")
        tabs.addTab(self.tab_ik,     "3  Inverse Kinematics")
        tabs.addTab(self.tab_val,    "4  Validation")
        tabs.addTab(self.tab_pipe,   "5  Full Pipeline")

        self.setCentralWidget(tabs)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE_SHEET)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
