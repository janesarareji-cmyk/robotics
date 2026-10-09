"""
Jane's Robotic Manipulator v1 | GUI v5
- Tabs: Setup | Forward Kinematics | Inverse Kinematics | Validate | Full Run
- Fixed base (axis limits locked to robot reach)
- Larger fonts throughout
- Continuous workflow: FK position auto-fills IK target, IK auto-triggers Validate
- IK guesses always match robot DOF
"""
import sys, io, traceback
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QPushButton, QTextEdit, QScrollArea,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QGroupBox, QSplitter, QSizePolicy,
    QDoubleSpinBox, QMessageBox, QFrame, QComboBox, QSlider,
    QStatusBar,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QColor, QFont
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d import proj3d
from .models import JointDefinition, JointState, TargetPose
from .robot_configuration import create_robot_definition
from .dh import get_dh_table
from .forward_kinematics import forward_kinematics
from .inverse_kinematics import solve_inverse_kinematics
from .validation import validate_ik_solutions
from .pipeline import run_pipeline
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Palette ───────────────────────────────────────────────────────────────────
C = {
    "bg":    "#0b0b14", "panel": "#13131f", "panel2": "#1a1a2e",
    "border":"#252540", "border2":"#3d3d70",
    "accent":"#00d4ff", "accent2":"#7c3aed", "accent3":"#06b6d4",
    "green": "#10b981", "red":   "#ef4444", "yellow":"#f59e0b",
    "text":  "#e2e8f0", "tdim":  "#94a3b8", "tmuted":"#2d3748",
    "btn":   "#1e40af", "btnh":  "#2563eb",
    "btng":  "#065f46", "btngh": "#047857",
    "btnr":  "#7f1d1d", "btnrh": "#991b1b",
}
GC = ["#ef4444","#f59e0b","#10b981","#3b82f6","#8b5cf6","#06b6d4","#f97316","#64748b"]

STYLE = f"""
QMainWindow,QWidget{{background:{C['bg']};color:{C['text']};
    font-family:'Segoe UI',Arial,sans-serif;font-size:14px}}
QTabWidget::pane{{border:1px solid {C['border']};background:{C['bg']};border-radius:6px}}
QTabBar::tab{{background:{C['panel']};color:{C['tdim']};padding:13px 22px;
    border:1px solid {C['border']};border-bottom:none;
    border-top-left-radius:7px;border-top-right-radius:7px;
    font-weight:700;font-size:14px;min-width:160px}}
QTabBar::tab:selected{{background:{C['panel2']};color:{C['accent']};
    border-bottom:3px solid {C['accent']}}}
QTabBar::tab:hover:!selected{{color:{C['text']};background:{C['border']}}}
QPushButton{{background:{C['btn']};color:#fff;border:none;border-radius:7px;
    padding:9px 20px;font-weight:700;font-size:14px}}
QPushButton:hover{{background:{C['btnh']}}}
QPushButton:pressed{{background:{C['accent']};color:#000}}
QPushButton:disabled{{background:{C['tmuted']};color:{C['tdim']}}}
QPushButton#danger{{background:{C['btnr']}}}
QPushButton#danger:hover{{background:{C['btnrh']}}}
QPushButton#success{{background:{C['btng']}}}
QPushButton#success:hover{{background:{C['btngh']}}}
QPushButton#accent{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
    stop:0 {C['accent2']},stop:1 {C['accent']});
    color:#fff;font-size:15px;padding:11px 28px;border-radius:9px;font-weight:800}}
QPushButton#accent:hover{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
    stop:0 #8b5cf6,stop:1 #06b6d4)}}
QPushButton#toggle{{background:{C['panel2']};border:2px solid {C['border']};
    color:{C['tdim']};border-radius:7px;padding:7px 15px;font-size:13px;font-weight:600}}
QPushButton#toggle:checked{{background:{C['accent']};border-color:{C['accent']};color:#000}}
QDoubleSpinBox,QComboBox{{background:{C['panel2']};border:1px solid {C['border']};
    border-radius:6px;padding:7px 11px;color:{C['text']};font-size:14px}}
QDoubleSpinBox:focus,QComboBox:focus{{border:2px solid {C['accent']}}}
QComboBox QAbstractItemView{{background:{C['panel2']};border:1px solid {C['border']};
    color:{C['text']};selection-background-color:{C['btn']}}}
QTextEdit{{background:#07070f;border:1px solid {C['border']};border-radius:6px;
    color:{C['text']};font-family:'Consolas',monospace;font-size:13px;padding:8px}}
QTableWidget{{background:{C['panel2']};border:1px solid {C['border']};border-radius:6px;
    gridline-color:{C['border']};color:{C['text']};
    selection-background-color:{C['btn']};alternate-background-color:{C['panel']}}}
QTableWidget::item{{padding:6px 10px;border-bottom:1px solid {C['border']};font-size:14px}}
QTableWidget::item:selected{{background:{C['btn']};color:#fff}}
QHeaderView::section{{background:{C['panel']};color:{C['accent']};padding:8px 10px;
    border:none;border-bottom:2px solid {C['accent']};font-weight:700;font-size:13px}}
QGroupBox{{border:1px solid {C['border']};border-radius:9px;margin-top:16px;
    padding-top:12px;font-weight:700;color:{C['accent']};font-size:14px}}
QGroupBox::title{{subcontrol-origin:margin;left:14px;padding:0 7px;
    background:{C['bg']};border-radius:4px}}
QSlider::groove:horizontal{{height:7px;background:{C['panel2']};border-radius:4px}}
QSlider::handle:horizontal{{background:{C['accent']};border:2px solid {C['accent3']};
    width:20px;height:20px;margin:-7px 0;border-radius:10px}}
QSlider::handle:horizontal:hover{{background:#fff;border-color:{C['accent']}}}
QSlider::sub-page:horizontal{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
    stop:0 {C['accent2']},stop:1 {C['accent']});border-radius:4px}}
QScrollBar:vertical{{background:{C['bg']};width:8px;border-radius:4px}}
QScrollBar::handle:vertical{{background:{C['border2']};border-radius:4px;min-height:24px}}
QScrollBar::handle:vertical:hover{{background:{C['accent']}}}
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{{height:0}}
QStatusBar{{background:{C['panel']};color:{C['tdim']};
    border-top:1px solid {C['border']};font-size:13px;padding:4px 14px}}
QLabel#section{{font-size:17px;font-weight:800;color:{C['accent']};padding:4px 0}}
QLabel#hint{{font-size:13px;color:{C['tdim']}}}
QLabel#value{{color:{C['accent']};font-family:Consolas,monospace;
    font-size:15px;font-weight:700}}
"""

DEFAULT_JOINTS = [
    [0,"revolute", 0,100,  0, 90,-180,180],
    [1,"revolute", 0,  0,150,  0, -90, 90],
    [2,"revolute", 0,  0,120,  0,-120,120],
    [3,"revolute", 0,  0, 80,  0,-180,180],
]
DEFAULT_ANGLES = [30., 20., -15., 10.]

PRESETS = {
    "4-DOF Articulated (RRRR)":
        [[0,"revolute",0,100,0,90,-180,180],[1,"revolute",0,0,150,0,-90,90],
         [2,"revolute",0,0,120,0,-120,120],[3,"revolute",0,0,80,0,-180,180]],
    "3-DOF SCARA (RRP)":
        [[0,"revolute",0,100,150,0,-180,180],[1,"revolute",0,0,120,180,-180,180],
         [2,"prismatic",0,50,0,0,0,200]],
    "3-DOF Cylindrical (RPP)":
        [[0,"revolute",0,100,0,90,-180,180],[1,"prismatic",0,50,0,90,0,200],
         [2,"prismatic",0,80,0,0,0,250]],
    "4-DOF Hybrid (RRPR)":
        [[0,"revolute",0,100,0,90,-180,180],[1,"revolute",0,0,140,0,-90,90],
         [2,"prismatic",0,60,0,90,0,150],[3,"revolute",0,0,80,0,-180,180]],
}

# ── Tiny UI helpers ───────────────────────────────────────────────────────────
def _div():
    f = QFrame(); f.setFrameShape(QFrame.HLine)
    f.setStyleSheet(f"background:{C['border']};max-height:1px;margin:6px 0;")
    return f

def _h(txt, size=13):
    l = QLabel(txt); l.setObjectName("hint")
    l.setStyleSheet(f"font-size:{size}px;color:{C['tdim']};")
    l.setWordWrap(True); return l

def _t(txt, sz=17):
    l = QLabel(txt); l.setObjectName("section")
    l.setStyleSheet(f"font-size:{sz}px;font-weight:800;color:{C['accent']};padding:3px 0;")
    return l

def _card(label, init="--", col=None):
    w = QWidget()
    w.setStyleSheet(f"background:{C['panel2']};border:1px solid {C['border']};border-radius:8px;")
    v = QVBoxLayout(w); v.setContentsMargins(12,8,12,8); v.setSpacing(3)
    tl = QLabel(label)
    tl.setStyleSheet(f"color:{C['tdim']};font-size:12px;font-weight:600;background:transparent;border:none;")
    vl = QLabel(init); c = col or C["accent"]
    vl.setStyleSheet(f"color:{c};font-size:16px;font-weight:800;font-family:Consolas;background:transparent;border:none;")
    v.addWidget(tl); v.addWidget(vl); return w, vl

def _ok_badge():
    l = QLabel("  PASS  ")
    l.setStyleSheet(f"background:#10b98122;color:#10b981;border:1px solid #10b98166;"
                    f"border-radius:5px;font-weight:700;font-size:13px;padding:3px 8px;")
    l.setAlignment(Qt.AlignCenter); return l

def _fail_badge():
    l = QLabel("  FAIL  ")
    l.setStyleSheet(f"background:#ef444422;color:#ef4444;border:1px solid #ef444466;"
                    f"border-radius:5px;font-weight:700;font-size:13px;padding:3px 8px;")
    l.setAlignment(Qt.AlignCenter); return l

def _make_guesses(robot, n_extra=8):
    """Generate diverse guesses always matching robot DOF."""
    joints = robot.joints
    lo = np.array([j.min_limit for j in joints])
    hi = np.array([j.max_limit for j in joints])
    mid = (lo + hi) / 2.
    nom = np.array([max(j.min_limit, min(j.max_limit, j.theta if j.joint_type=="revolute" else j.d)) for j in joints])
    guesses = [mid.tolist(), nom.tolist(),
               (lo + 0.25*(hi-lo)).tolist(), (lo + 0.75*(hi-lo)).tolist(),
               lo.tolist(), hi.tolist()]
    rng = np.random.default_rng(42)
    for _ in range(n_extra):
        guesses.append(rng.uniform(lo, hi).tolist())
    return guesses

# ── 3-D geometry ──────────────────────────────────────────────────────────────
def _perp(v):
    v = v/np.linalg.norm(v); n = np.array([1.,0.,0.])
    if abs(np.dot(v,n)) > 0.98: n = np.array([0.,1.,0.])
    n1 = np.cross(v,n); n1 /= np.linalg.norm(n1); return n1, np.cross(v,n1)

def _cyl(ax, p0, p1, r, col, alpha=0.92, ns=16):
    p0,p1 = np.asarray(p0,float), np.asarray(p1,float)
    d = p1-p0; L = np.linalg.norm(d)
    if L < 1e-6: return
    d /= L; n1,n2 = _perp(d); t = np.linspace(0,2*np.pi,ns)
    T,TH = np.meshgrid([0.,L], t)
    X = p0[0]+d[0]*T+r*(np.sin(TH)*n1[0]+np.cos(TH)*n2[0])
    Y = p0[1]+d[1]*T+r*(np.sin(TH)*n1[1]+np.cos(TH)*n2[1])
    Z = p0[2]+d[2]*T+r*(np.sin(TH)*n1[2]+np.cos(TH)*n2[2])
    ax.plot_surface(X,Y,Z,color=col,alpha=alpha,linewidth=0,antialiased=True,shade=True)

def _disc(ax, c, n, r, col, alpha=0.92, ns=16):
    n = np.asarray(n,float); n /= np.linalg.norm(n); n1,n2 = _perp(n)
    t = np.linspace(0,2*np.pi,ns); T,TH = np.meshgrid([0.,r], t)
    X = c[0]+T*(np.cos(TH)*n1[0]+np.sin(TH)*n2[0])
    Y = c[1]+T*(np.cos(TH)*n1[1]+np.sin(TH)*n2[1])
    Z = c[2]+T*(np.cos(TH)*n1[2]+np.sin(TH)*n2[2])
    ax.plot_surface(X,Y,Z,color=col,alpha=alpha,linewidth=0,antialiased=True,shade=True)

def _sph(ax, p, r, col, alpha=0.92, ns=12):
    u = np.linspace(0,2*np.pi,ns); v = np.linspace(0,np.pi,ns//2)
    X = p[0]+r*np.outer(np.cos(u),np.sin(v))
    Y = p[1]+r*np.outer(np.sin(u),np.sin(v))
    Z = p[2]+r*np.outer(np.ones(ns),np.cos(v))
    ax.plot_surface(X,Y,Z,color=col,alpha=alpha,linewidth=0,antialiased=True,shade=True)

def _revj(ax, p, ax2, Rd, Hd, c1, c2):
    ax2 = ax2/np.linalg.norm(ax2); hh,rh,hr = Hd*.5, Hd*.15, Rd*.6
    _cyl(ax, p-hh*ax2, p+hh*ax2, hr, c1, 0.93)
    for s in [-1,1]:
        rc = p+s*hh*ax2; _cyl(ax, rc-rh*ax2, rc+rh*ax2, Rd, c2, 0.93)
        _disc(ax, rc+s*rh*ax2, ax2, Rd, c2, 0.93)
    _disc(ax, p+hh*ax2, ax2, hr, c1, 0.93)
    _disc(ax, p-hh*ax2, -ax2, hr, c1, 0.93)

def _prisj(ax, p, ax2, w=16., h=24.):
    ax2 = ax2/max(np.linalg.norm(ax2),1e-8)
    _cyl(ax, p-h*.55*ax2, p+h*.55*ax2, w*.5, "#e0af68", 0.92)
    _disc(ax, p+h*.55*ax2, ax2, w*.5, "#c97f20", 0.92)
    _disc(ax, p-h*.55*ax2, -ax2, w*.5, "#c97f20", 0.92)
    _cyl(ax, p-h*.9*ax2, p+h*.9*ax2, w*.2, "#7aa2f7", 0.90)

def _floor(ax, size):
    step = max(size/6, 20.)
    for v in np.arange(-size, size+step, step):
        ax.plot([v,v],[-size,size],[0,0], color="#1a1a35", lw=0.5, alpha=0.7)
        ax.plot([-size,size],[v,v],[0,0], color="#1a1a35", lw=0.5, alpha=0.7)

def _ws_sphere(ax, reach):
    u = np.linspace(0,2*np.pi,36); v = np.linspace(0,np.pi,18)
    X = reach*np.outer(np.cos(u),np.sin(v)); Y = reach*np.outer(np.sin(u),np.sin(v))
    Z = reach*np.outer(np.ones(36),np.cos(v))
    ax.plot_surface(X,Y,Z,color="#00d4ff",alpha=0.03,linewidth=0,shade=False)
    th = np.linspace(0,2*np.pi,90)
    ax.plot(reach*np.cos(th),reach*np.sin(th),np.zeros(90),color="#00d4ff",lw=0.8,alpha=0.2)

def _triad(ax, sc=25.):
    for i,(col,lbl) in enumerate(zip(["#ff4444","#44ff44","#4499ff"],["X","Y","Z"])):
        d = np.zeros(3); d[i] = sc
        ax.quiver(0,0,0,d[0],d[1],d[2],color=col,lw=1.5,arrow_length_ratio=0.3,alpha=0.85)

def _target_rings(ax, pos, r=14., col="#ef4444"):
    for n in [np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])]:
        th = np.linspace(0,2*np.pi,60); n1,n2 = _perp(n)
        ax.plot(pos[0]+r*(np.cos(th)*n1[0]+np.sin(th)*n2[0]),
                pos[1]+r*(np.cos(th)*n1[1]+np.sin(th)*n2[1]),
                pos[2]+r*(np.cos(th)*n1[2]+np.sin(th)*n2[2]),
                color=col,lw=2,alpha=0.85)
    ax.scatter([pos[0]],[pos[1]],[pos[2]],color=col,s=120,marker="X",zorder=12,depthshade=False)


# ── ArmCanvas ─────────────────────────────────────────────────────────────────
class ArmCanvas(FigureCanvas):
    LR=8.; JR=13.; JH=17.; BR=38.; BH=28.; WR=9.; GL=26.; GS=13.; FR=3.
    CL="#3b82f6"; CJ="#8b5cf6"; CB="#1e293b"; CE="#10b981"; CT="#ef4444"

    def __init__(self, parent=None, w=6, h=6, dpi=90):
        self.fig = Figure(figsize=(w,h), dpi=dpi, facecolor=C["panel"])
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self._az = -55.; self._el = 22.
        self._rot = QTimer(); self._rot.timeout.connect(self._spin)
        self._reach = 400.
        self._show_ws = True
        self._gpts = []
        self._ghosts_on = True
        self._style(); self._idle()

    def set_reach(self, robot):
        self._reach = sum(abs(j.a)+abs(j.d) for j in robot.joints) + 60.

    def set_auto_rotate(self, on):
        if on: self._rot.start(35)
        else:  self._rot.stop()

    def _spin(self):
        self._az = (self._az+0.7)%360
        self.ax.view_init(elev=self._el, azim=self._az); self.draw_idle()

    def _style(self):
        ax = self.ax; ax.set_facecolor("#07070f")
        for pn in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pn.fill=True; pn.set_facecolor("#0a0a18"); pn.set_edgecolor(C["border"]); pn.set_alpha(0.6)
        ax.tick_params(colors=C["tdim"], labelsize=7)
        for a in [ax.xaxis, ax.yaxis, ax.zaxis]: a.label.set_color(C["tdim"])
        ax.set_xlabel("X mm",fontsize=7,labelpad=1)
        ax.set_ylabel("Y mm",fontsize=7,labelpad=1)
        ax.set_zlabel("Z mm",fontsize=7,labelpad=1)
        ax.grid(False)

    def _fix(self):
        r = self._reach * 1.1
        self.ax.set_xlim(-r,r); self.ax.set_ylim(-r,r); self.ax.set_zlim(-r*0.15,r)
        try: self.ax.set_box_aspect((1,1,1))
        except: pass

    def _idle(self):
        ax = self.ax; ax.cla(); self._style()
        ax.view_init(elev=self._el, azim=self._az)
        _floor(ax, self._reach); _triad(ax, self._reach*0.06)
        ax.set_title("3D Workspace", color=C["tdim"], fontsize=9, pad=4)
        self._fix(); self.draw()

    def draw_arm(self, robot, fk, target=None, ghost_fk=None, active_ghost=None):
        ax = self.ax; ax.cla(); self._style()
        ax.view_init(elev=self._el, azim=self._az)
        _floor(ax, self._reach)
        if self._show_ws: _ws_sphere(ax, self._reach*0.95)
        _triad(ax, self._reach*0.06)

        frames = [(np.zeros(3), np.eye(3))]
        for T in fk.transformation_chain:
            frames.append((T[:3,3].copy(), T[:3,:3].copy()))
        ee = np.asarray(fk.pose.position, float)
        er = np.asarray(fk.pose.rotation, float)
        jts = [j.joint_type.lower() for j in robot.joints] if robot else []

        self._gpts = []
        if ghost_fk and self._ghosts_on:
            for gi, gfk in enumerate(ghost_fk):
                if gfk is None: continue
                ia = (gi == active_ghost); gc = GC[gi%len(GC)]; am = 1.0 if ia else 0.22
                gfr = [(np.zeros(3),np.eye(3))]
                for T in gfk.transformation_chain: gfr.append((T[:3,3].copy(),T[:3,:3].copy()))
                ge = np.asarray(gfk.pose.position,float); gr = np.asarray(gfk.pose.rotation,float)
                self._geom(ax,gfr,ge,gr,gc,gc,0.55*am,0.6*am,False,jts)
                ax.scatter([ge[0]],[ge[1]],[ge[2]],color=gc,s=200 if ia else 70,
                    marker="D" if ia else "o",zorder=10,depthshade=False,
                    label="Sol%d (%.0f,%.0f,%.0f)"%(gi+1,ge[0],ge[1],ge[2]))
                self._gpts.append((ge.copy(),gi))

        self._base(ax)
        self._geom(ax,frames,ee,er,self.CL,self.CJ,0.92,0.95,True,jts)

        if target is not None:
            _target_rings(ax, np.asarray(target,float))
            ax.scatter([],[],[],color=self.CT,marker="X",s=60,
                label="Target (%.0f,%.0f,%.0f)"%(target[0],target[1],target[2]))

        ax.scatter([ee[0]],[ee[1]],[ee[2]],color=self.CE,s=80,marker="o",zorder=11,depthshade=False,
            label="EE (%.0f,%.0f,%.0f)"%(ee[0],ee[1],ee[2]))

        if self._gpts or target is not None:
            ax.legend(loc="upper left",fontsize=7.5,facecolor=C["panel"],
                      edgecolor=C["border"],labelcolor=C["text"],framealpha=0.85)
        self._fix()
        ax.set_title("3D Workspace — Base fixed at origin", color=C["accent"], fontsize=9, pad=4)
        self.draw()

    def _base(self, ax):
        top = np.array([0.,0.,self.BH])
        _cyl(ax, np.zeros(3), top, self.BR, self.CB, 0.97)
        _disc(ax, top, [0,0,1], self.BR, "#334155", 0.97)
        _disc(ax, np.zeros(3), [0,0,-1], self.BR*1.3, "#0f172a", 0.97)
        _disc(ax, top+[0,0,1], [0,0,1], self.BR*.5, self.CJ, 0.97)

    def _geom(self, ax, frames, ee, er, lc, jc, la, ja, gripper, jts):
        for i in range(len(frames)-1):
            p0,R0 = frames[i]; p1,_ = frames[i+1]
            lv = p1-p0; ll = np.linalg.norm(lv); jax = R0[:,2]
            if ll > 1.:
                ins = min(self.JH*.45, ll*.1); lu = lv/ll
                _cyl(ax, p0+lu*ins, p1-lu*ins, self.LR, lc, la)
                _disc(ax, p0+lu*ins, -lu, self.LR, "#1e3a5f", ja)
                _disc(ax, p1-lu*ins,  lu, self.LR, "#1e3a5f", ja)
            jt = jts[i] if i < len(jts) else "revolute"
            if jt=="prismatic": _prisj(ax, p0, jax, self.JR*1.2, self.JH*1.1)
            else: _revj(ax, p0, jax, self.JR, self.JH, jc, jc)
        if frames:
            pl,Rl = frames[-1]; lt = jts[-1] if jts else "revolute"
            if lt=="prismatic": _prisj(ax, pl, Rl[:,2], self.JR, self.JH)
            else: _revj(ax, pl, Rl[:,2], self.JR*.8, self.JH*.8, jc, jc)
        wl = np.linalg.norm(ee-frames[-1][0])
        if wl > 1.: _cyl(ax, frames[-1][0], ee, self.WR, "#1d4ed8", la)
        if gripper: self._grip(ax, ee, er)

    def _grip(self, ax, ee, er):
        zv,yv = er[:,2], er[:,1]; pe = ee+zv*10.
        _cyl(ax, ee, pe, self.WR*1.1, self.CE, 0.93)
        _disc(ax, pe, zv, self.WR*1.1, "#059669", 0.92)
        for s in [1,-1]:
            fr = pe+yv*self.GS*s; ft = fr+zv*self.GL
            _cyl(ax, fr, ft, self.FR, "#34d399", 0.94)
            _sph(ax, ft, self.FR*1.4, "#6ee7b7", 0.92)
        _cyl(ax, pe+yv*self.GS, pe-yv*self.GS, self.FR*.7, "#065f46", 0.88)


# ── Joint slider row ──────────────────────────────────────────────────────────
class JSlider(QWidget):
    changed = pyqtSignal()
    SC = 100
    def __init__(self, idx, jd, parent=None):
        super().__init__(parent); self.idx=idx; self.jd=jd; self._build()
    def _build(self):
        lay = QHBoxLayout(self); lay.setContentsMargins(4,4,4,4); lay.setSpacing(10)
        ir = self.jd.joint_type.lower()=="revolute"; col = C["accent"] if ir else C["yellow"]
        lbl = QLabel(("R" if ir else "P")+str(self.idx))
        lbl.setStyleSheet(f"color:{col};font-weight:800;font-size:14px;min-width:26px;background:transparent;")
        lay.addWidget(lbl)
        j = self.jd; lo=int(j.min_limit*self.SC); hi=int(j.max_limit*self.SC)
        dv = max(lo, min(hi, int((j.theta if ir else j.d)*self.SC)))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(lo,hi); self.slider.setValue(dv); self.slider.setMinimumWidth(120)
        lay.addWidget(self.slider, stretch=3)
        self.spin = QDoubleSpinBox()
        self.spin.setRange(j.min_limit,j.max_limit); self.spin.setDecimals(1)
        self.spin.setSingleStep(1. if ir else 5.); self.spin.setValue(dv/self.SC)
        self.spin.setFixedWidth(80); lay.addWidget(self.spin)
        ul = QLabel("d" if ir else "mm")
        ul.setStyleSheet(f"color:{C['tdim']};font-size:12px;min-width:20px;background:transparent;")
        lay.addWidget(ul)
        rl = QLabel("["+str(int(j.min_limit))+".."+str(int(j.max_limit))+"]")
        rl.setStyleSheet(f"color:{C['tmuted']};font-size:11px;min-width:88px;background:transparent;")
        lay.addWidget(rl)
        self._lock = False
        self.slider.valueChanged.connect(self._fs); self.spin.valueChanged.connect(self._fsp)
    def _fs(self,v):
        if self._lock: return
        self._lock=True; self.spin.setValue(v/self.SC); self._lock=False; self.changed.emit()
    def _fsp(self,v):
        if self._lock: return
        self._lock=True; self.slider.setValue(int(v*self.SC)); self._lock=False; self.changed.emit()
    def get(self): return self.spin.value()
    def set(self, v):
        c = max(self.jd.min_limit, min(self.jd.max_limit, v))
        self._lock=True; self.slider.setValue(int(c*self.SC)); self.spin.setValue(c); self._lock=False


# ── IK Worker ─────────────────────────────────────────────────────────────────
class IKWorker(QThread):
    done = pyqtSignal(list, str)
    def __init__(self, robot, target, guesses, tol):
        super().__init__(); self.robot=robot; self.target=target; self.guesses=guesses; self.tol=tol
    def run(self):
        try: self.done.emit(solve_inverse_kinematics(self.robot,self.target,self.guesses,self.tol),"")
        except Exception: self.done.emit([], traceback.format_exc())


# ── Tab 1: Setup ──────────────────────────────────────────────────────────────
class SetupTab(QWidget):
    robot_changed = pyqtSignal(object)
    COLS = ["#","Type","theta","d","a","alpha","Min","Max"]
    def __init__(self):
        super().__init__(); self.robot=None; self._build(); self._defaults()
    def _build(self):
        out = QHBoxLayout(self); out.setContentsMargins(20,20,20,20); out.setSpacing(20)
        L = QVBoxLayout(); L.setSpacing(12)
        L.addWidget(_t("Robot Setup"))
        L.addWidget(_h("Choose a preset or edit the joint table, then click Validate Robot."))
        L.addWidget(_div())
        pr = QHBoxLayout(); pr.addWidget(QLabel("Preset:"))
        self.combo = QComboBox()
        for n in PRESETS: self.combo.addItem(n)
        pr.addWidget(self.combo,stretch=1)
        bl = QPushButton("Load Preset"); bl.clicked.connect(self._load); pr.addWidget(bl)
        L.addLayout(pr)
        self.tbl = QTableWidget(0,len(self.COLS)); self.tbl.setHorizontalHeaderLabels(self.COLS)
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl.setAlternatingRowColors(True); self.tbl.setMinimumHeight(180)
        L.addWidget(self.tbl)
        br = QHBoxLayout()
        ba = QPushButton("+ Add Joint"); ba.clicked.connect(self._add)
        bx = QPushButton("− Remove Last"); bx.setObjectName("danger"); bx.clicked.connect(self._rem)
        bv = QPushButton("✔  Validate Robot"); bv.setObjectName("accent"); bv.clicked.connect(self._validate)
        br.addWidget(ba); br.addWidget(bx); br.addStretch(); br.addWidget(bv)
        L.addLayout(br)
        R = QVBoxLayout(); R.setSpacing(12)
        R.addWidget(_t("Status", sz=15))
        self.lres = QLabel("Not validated yet.")
        self.lres.setWordWrap(True)
        self.lres.setStyleSheet(f"background:{C['panel2']};border:1px solid {C['border']};"
                                f"border-radius:7px;padding:12px;font-size:14px;")
        R.addWidget(self.lres)
        cr = QHBoxLayout()
        c1,self.ldof = _card("Total Joints","--"); c2,self.ltype = _card("Configuration","--")
        cr.addWidget(c1); cr.addWidget(c2); R.addLayout(cr)
        gd = QGroupBox("DH Parameter Table"); dl = QVBoxLayout(gd)
        self.txt_dh = QTextEdit(); self.txt_dh.setReadOnly(True); self.txt_dh.setFixedHeight(170)
        self.txt_dh.setPlaceholderText("DH table appears after validation..."); dl.addWidget(self.txt_dh)
        R.addWidget(gd); R.addStretch()
        lw=QWidget(); lw.setLayout(L); rw=QWidget(); rw.setLayout(R)
        sp=QSplitter(Qt.Horizontal); sp.addWidget(lw); sp.addWidget(rw); sp.setSizes([510,390])
        out.addWidget(sp)
    def _defaults(self):
        for r in DEFAULT_JOINTS: self._add(r)
    def _load(self):
        n = self.combo.currentText()
        if n in PRESETS: self.tbl.setRowCount(0); [self._add(r) for r in PRESETS[n]]; self._validate()
    def _add(self, data=None):
        r = self.tbl.rowCount(); self.tbl.insertRow(r)
        d = [str(r),"revolute","0","0","0","0","-180","180"]
        if data: d = [str(v) for v in data]
        for c,val in enumerate(d):
            if c==1:
                cb=QComboBox(); cb.addItems(["revolute","prismatic"])
                cb.setCurrentText(str(val).lower().strip()); self.tbl.setCellWidget(r,c,cb)
            else:
                it=QTableWidgetItem(val); it.setTextAlignment(Qt.AlignCenter); self.tbl.setItem(r,c,it)
    def _rem(self):
        r=self.tbl.rowCount()
        if r>0: self.tbl.removeRow(r-1)
    def _validate(self):
        try:
            joints=self._parse(); name=self.combo.currentText().split("(")[0].strip()
            robot=create_robot_definition(name,joints); self.robot=robot
            n=len(robot.joints); rev=sum(1 for j in robot.joints if j.joint_type=="revolute"); pri=n-rev
            self.lres.setText(f"<b style='color:{C['green']}'>✔  Validated!</b>  {robot.name}")
            self.ldof.setText(str(n))
            parts=[]
            if rev: parts.append(f"{rev} revolute")
            if pri: parts.append(f"{pri} prismatic")
            self.ltype.setText(" + ".join(parts))
            dh=get_dh_table(robot)
            lines=["  #  type        theta       d        a    alpha    min    max","─"*67]
            for j in dh:
                lines.append("%3d  %-10s %7.1f %7.1f %7.1f %7.1f %6.0f %6.0f"%(
                    j.index,j.joint_type,j.theta,j.d,j.a,j.alpha,j.min_limit,j.max_limit))
            self.txt_dh.setPlainText("\n".join(lines))
            self.robot_changed.emit(robot)
        except Exception as e:
            self.lres.setText(f"<b style='color:{C['red']}'>Error: {e}</b>"); self.robot=None
    def _parse(self):
        joints=[]
        for r in range(self.tbl.rowCount()):
            def cell(c,r=r):
                if c==1:
                    w=self.tbl.cellWidget(r,c)
                    return w.currentText().strip().lower() if isinstance(w,QComboBox) else "revolute"
                it=self.tbl.item(r,c); return it.text().strip() if it else "0"
            joints.append(JointDefinition(int(cell(0)),cell(1),float(cell(2)),float(cell(3)),
                                          float(cell(4)),float(cell(5)),float(cell(6)),float(cell(7))))
        return joints
    def get_robot(self): return self.robot


# ── Tab 2: Forward Kinematics ─────────────────────────────────────────────────
class FKTab(QWidget):
    fk_done = pyqtSignal(object, object)      # (robot, FKResult)
    def __init__(self, get_robot, on_send_to_ik):
        super().__init__(); self.get_robot=get_robot; self.on_send_to_ik=on_send_to_ik
        self._sliders=[]; self._lfk=None; self._lrobot=None
        self._tmr=QTimer(); self._tmr.setSingleShot(True); self._tmr.setInterval(35)
        self._tmr.timeout.connect(self._run); self._build()
    def _build(self):
        out=QHBoxLayout(self); out.setContentsMargins(20,20,20,20); out.setSpacing(20)
        L=QVBoxLayout(); L.setSpacing(12)
        L.addWidget(_t("Forward Kinematics"))
        L.addWidget(_h("Drag any slider — the 3D arm updates live. Base stays fixed."))
        L.addWidget(_div())
        self.grp=QGroupBox("Joint Controls"); self.slay=QVBoxLayout(self.grp)
        self.slay.setSpacing(4); L.addWidget(self.grp)
        qa=QHBoxLayout()
        bz=QPushButton("Zero All"); bz.clicked.connect(self._zero)
        br=QPushButton("Randomize"); br.clicked.connect(self._rand)
        qa.addWidget(bz); qa.addWidget(br); qa.addStretch(); L.addLayout(qa)
        L.addWidget(_div())
        L.addWidget(_h("End-Effector Position:", size=14))
        pr=QHBoxLayout()
        c1,self.lx=_card("X (mm)","--"); c2,self.ly=_card("Y (mm)","--"); c3,self.lz=_card("Z (mm)","--")
        pr.addWidget(c1); pr.addWidget(c2); pr.addWidget(c3); L.addLayout(pr)
        g=QGroupBox("Rotation Matrix"); rl=QVBoxLayout(g)
        self.lrot=QTextEdit(); self.lrot.setReadOnly(True); self.lrot.setFixedHeight(85)
        rl.addWidget(self.lrot); L.addWidget(g)
        # Send to IK button — continuous workflow
        self.bsend=QPushButton("→  Send to Inverse Kinematics"); self.bsend.setObjectName("accent")
        self.bsend.clicked.connect(self._send_to_ik); self.bsend.setEnabled(False); L.addWidget(self.bsend)
        L.addStretch()
        R=QVBoxLayout(); R.setSpacing(8)
        topbar=QHBoxLayout(); topbar.addWidget(_t("3D View",sz=13)); topbar.addStretch()
        self.bar=QPushButton("Auto-Rotate: OFF"); self.bar.setObjectName("toggle"); self.bar.setCheckable(True)
        self.bar.toggled.connect(lambda c:(self.canvas.set_auto_rotate(c),
            self.bar.setText("Auto-Rotate: ON" if c else "Auto-Rotate: OFF")))
        topbar.addWidget(self.bar); R.addLayout(topbar)
        self.canvas=ArmCanvas(w=6,h=6,dpi=90); R.addWidget(self.canvas)
        ws_row=QHBoxLayout(); ws_row.addStretch()
        self.btn_ws=QPushButton("Workspace Sphere: ON"); self.btn_ws.setObjectName("toggle")
        self.btn_ws.setCheckable(True); self.btn_ws.setChecked(True)
        self.btn_ws.toggled.connect(self._tog_ws); ws_row.addWidget(self.btn_ws); R.addLayout(ws_row)
        lw=QWidget(); lw.setLayout(L); rw=QWidget(); rw.setLayout(R)
        sp=QSplitter(Qt.Horizontal); sp.addWidget(lw); sp.addWidget(rw); sp.setSizes([390,540])
        out.addWidget(sp)
    def on_robot(self, robot):
        if not robot: return
        while self.slay.count():
            it=self.slay.takeAt(0)
            if it.widget(): it.widget().deleteLater()
        self._sliders=[]
        for i,j in enumerate(robot.joints):
            sl=JSlider(i,j); sl.changed.connect(self._tmr.start)
            self.slay.addWidget(sl); self._sliders.append(sl)
        for i,j in enumerate(robot.joints):
            dv=j.theta if j.joint_type=="revolute" else j.d
            if i<len(DEFAULT_ANGLES) and j.joint_type=="revolute": dv=DEFAULT_ANGLES[i]
            self._sliders[i].set(max(j.min_limit,min(j.max_limit,dv)))
        self.canvas.set_reach(robot); self._run()
    def _zero(self): [s.set(0.) for s in self._sliders]; self._run()
    def _rand(self): [s.set(float(np.random.uniform(s.jd.min_limit,s.jd.max_limit))) for s in self._sliders]; self._run()
    def _run(self):
        robot=self.get_robot()
        if not robot or not self._sliders: return
        angles=[s.get() for s in self._sliders]
        if len(angles)!=len(robot.joints): return
        try:
            r=forward_kinematics(robot,JointState(angles))
            self._lfk=r; self._lrobot=robot
            pos=r.pose.position
            self.lx.setText(f"{pos[0]:+.1f}"); self.ly.setText(f"{pos[1]:+.1f}"); self.lz.setText(f"{pos[2]:+.1f}")
            rows=["  ".join(f"{v:+7.3f}" for v in row) for row in r.pose.rotation]
            self.lrot.setPlainText("\n".join(rows))
            self.canvas.draw_arm(robot,r)
            self.fk_done.emit(robot,r)
            self.bsend.setEnabled(True)
        except: pass
    def _tog_ws(self, on):
        self.canvas._show_ws=on; self.btn_ws.setText("Workspace Sphere: ON" if on else "Workspace Sphere: OFF")
        if self._lfk and self._lrobot: self.canvas.draw_arm(self._lrobot,self._lfk)
    def _send_to_ik(self):
        if self._lfk: self.on_send_to_ik(self._lfk.pose.position.copy())
    def get_fk(self): return self._lrobot, self._lfk


# ── Tab 3: Inverse Kinematics ─────────────────────────────────────────────────
class IKTab(QWidget):
    ik_done = pyqtSignal(object, object, list)   # robot, TargetPose, [IKSolution]
    def __init__(self, get_robot, get_fk, go_to_validate):
        super().__init__()
        self.get_robot=get_robot; self.get_fk=get_fk; self.go_to_validate=go_to_validate
        self._sols=[]; self._worker=None; self._gfk=[]; self._ai=None; self._rr=None; self._tr=None
        self._build()
    def _build(self):
        out=QHBoxLayout(self); out.setContentsMargins(20,20,20,20); out.setSpacing(20)
        L=QVBoxLayout(); L.setSpacing(12)
        L.addWidget(_t("Inverse Kinematics"))
        L.addWidget(_h("Enter a target position in mm and click Solve. All valid configurations are shown."))
        L.addWidget(_div())
        tg=QGroupBox("Target Position (mm)"); tgl=QGridLayout(tg)
        self.tx=QDoubleSpinBox(); self.ty=QDoubleSpinBox(); self.tz=QDoubleSpinBox()
        for s in [self.tx,self.ty,self.tz]: s.setRange(-2000,2000); s.setDecimals(1); s.setFixedHeight(38)
        self.tx.setValue(200.); self.ty.setValue(100.); self.tz.setValue(150.)
        tgl.addWidget(QLabel("X mm:"),0,0); tgl.addWidget(self.tx,0,1)
        tgl.addWidget(QLabel("Y mm:"),1,0); tgl.addWidget(self.ty,1,1)
        tgl.addWidget(QLabel("Z mm:"),2,0); tgl.addWidget(self.tz,2,1)
        bf=QPushButton("Use Current FK Position"); bf.clicked.connect(self._fill)
        tgl.addWidget(bf,3,0,1,2); L.addWidget(tg)
        tol_row=QHBoxLayout(); tol_row.addWidget(QLabel("Tolerance (mm):"))
        self.stol=QDoubleSpinBox(); self.stol.setRange(1e-8,100); self.stol.setDecimals(4)
        self.stol.setValue(1.0); self.stol.setFixedHeight(38)   # more permissive default
        tol_row.addWidget(self.stol); tol_row.addStretch(); L.addLayout(tol_row)
        self.bsolve=QPushButton("⚡  Solve IK"); self.bsolve.setObjectName("accent"); L.addWidget(self.bsolve)
        self.lstat=QLabel("Enter a target and click Solve.")
        self.lstat.setWordWrap(True)
        self.lstat.setStyleSheet(f"padding:10px;background:{C['panel2']};border-radius:6px;"
                                 f"border:1px solid {C['border']};color:{C['tdim']};font-size:14px;")
        L.addWidget(self.lstat)
        L.addWidget(_h("Solutions (click row to preview in 3D):"))
        self.rtbl=QTableWidget(0,3); self.rtbl.setHorizontalHeaderLabels(["#","Joint Angles","Pos Error (mm)"])
        self.rtbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.rtbl.setSelectionBehavior(QTableWidget.SelectRows)
        self.rtbl.setSelectionMode(QTableWidget.SingleSelection)
        self.rtbl.setAlternatingRowColors(True); self.rtbl.setMinimumHeight(160)
        L.addWidget(self.rtbl)
        g_row=QHBoxLayout()
        self.bg=QPushButton("Show All: ON"); self.bg.setObjectName("toggle")
        self.bg.setCheckable(True); self.bg.setChecked(True); self.bg.toggled.connect(self._tog)
        g_row.addWidget(self.bg); g_row.addStretch()
        # Continuous workflow: go straight to Validate
        self.bvalidate=QPushButton("→  Go to Validate"); self.bvalidate.setObjectName("success")
        self.bvalidate.clicked.connect(self.go_to_validate); self.bvalidate.setEnabled(False)
        g_row.addWidget(self.bvalidate); L.addLayout(g_row)
        L.addStretch()
        R=QVBoxLayout(); R.setSpacing(8)
        topbar=QHBoxLayout(); topbar.addWidget(_t("3D View",sz=13)); topbar.addStretch()
        self.bar=QPushButton("Auto-Rotate: OFF"); self.bar.setObjectName("toggle"); self.bar.setCheckable(True)
        self.bar.toggled.connect(lambda c:(self.canvas.set_auto_rotate(c),
            self.bar.setText("Auto-Rotate: ON" if c else "Auto-Rotate: OFF")))
        topbar.addWidget(self.bar); R.addLayout(topbar)
        self.canvas=ArmCanvas(w=6,h=6,dpi=90); R.addWidget(self.canvas)
        lw=QWidget(); lw.setLayout(L); rw=QWidget(); rw.setLayout(R)
        sp=QSplitter(Qt.Horizontal); sp.addWidget(lw); sp.addWidget(rw); sp.setSizes([410,510])
        out.addWidget(sp)
        self.bsolve.clicked.connect(self._solve); self.rtbl.itemSelectionChanged.connect(self._sel)
    def on_robot(self, robot):
        if not robot: return
        self._rr=robot; self.canvas.set_reach(robot)
    def set_target(self, pos):
        """Called from FK tab via 'Send to IK' button."""
        self.tx.setValue(float(pos[0])); self.ty.setValue(float(pos[1])); self.tz.setValue(float(pos[2]))
    def _fill(self):
        _,fk=self.get_fk()
        if not fk: QMessageBox.information(self,"Tip","Go to Forward Kinematics tab first."); return
        pos=fk.pose.position
        self.tx.setValue(float(pos[0])); self.ty.setValue(float(pos[1])); self.tz.setValue(float(pos[2]))
    def _solve(self):
        robot=self.get_robot()
        if not robot: QMessageBox.warning(self,"No Robot","Validate a robot in Setup tab first."); return
        tgt=TargetPose(position=np.array([self.tx.value(),self.ty.value(),self.tz.value()]))
        guesses=_make_guesses(robot, n_extra=10)   # always correct DOF
        tol=self.stol.value()
        self.bsolve.setEnabled(False); self.bsolve.setText("Solving...")
        self.lstat.setText("Searching for solutions (running in background)...")
        self._tr=np.array([self.tx.value(),self.ty.value(),self.tz.value()])
        self._worker=IKWorker(robot,tgt,guesses,tol)
        self._worker.done.connect(lambda s,e:self._done(robot,tgt,s,e)); self._worker.start()
    def _done(self, robot, target, solutions, err):
        self.bsolve.setEnabled(True); self.bsolve.setText("⚡  Solve IK")
        if err:
            self.lstat.setText(f"<b style='color:{C['red']}'>Error:</b><br>{err[:400]}"); return
        self._sols=solutions; self._rr=robot; self._ai=0 if solutions else None
        n=len(solutions)
        if n>0:
            self.lstat.setText(f"<b style='color:{C['green']}'>Found {n} solution(s)!</b>  "
                               f"Click a row to preview it in the 3D view.")
            self.bvalidate.setEnabled(True)
        else:
            self.lstat.setText(f"<b style='color:{C['yellow']}'>No solutions found.</b>  "
                               f"Try increasing the tolerance, or check that the target is reachable.")
        nj=len(robot.joints); self.rtbl.blockSignals(True); self.rtbl.setRowCount(0)
        self.rtbl.setColumnCount(nj+2)
        self.rtbl.setHorizontalHeaderLabels(["#"]+[f"q{i}" for i in range(nj)]+["Error"])
        for idx,sol in enumerate(solutions):
            r=self.rtbl.rowCount(); self.rtbl.insertRow(r); gc=GC[idx%len(GC)]
            vals=[str(idx+1)]+[f"{q:.1f}" for q in sol.joint_state.positions]+[f"{sol.position_error:.4f}"]
            for c,v in enumerate(vals):
                it=QTableWidgetItem(v); it.setTextAlignment(Qt.AlignCenter)
                it.setForeground(QColor(gc)); self.rtbl.setItem(r,c,it)
        self.rtbl.blockSignals(False)
        self._gfk=[]
        for sol in solutions:
            try: self._gfk.append(forward_kinematics(robot,sol.joint_state))
            except: self._gfk.append(None)
        if solutions: self.rtbl.selectRow(0)
        else: self._redraw()
        self.ik_done.emit(robot,target,solutions)
    def _sel(self):
        items=self.rtbl.selectedItems()
        if not items: return
        idx=items[0].row()
        if idx==self._ai: return
        self._ai=idx; self._redraw()
    def _redraw(self):
        if not self._rr or not self._sols: return
        idx=self._ai if self._ai is not None else 0
        if idx>=len(self._sols): return
        try: pfk=forward_kinematics(self._rr,self._sols[idx].joint_state)
        except: return
        ghosts=[g for g in self._gfk if g] if self.canvas._ghosts_on else None
        self.canvas.draw_arm(self._rr,pfk,target=self._tr,ghost_fk=ghosts,active_ghost=self._ai)
    def _tog(self, on):
        self.canvas._ghosts_on=on; self.bg.setText("Show All: ON" if on else "Show All: OFF")
        self._redraw()


# ── Tab 4: Validate ───────────────────────────────────────────────────────────
class ValidateTab(QWidget):
    def __init__(self, get_robot):
        super().__init__(); self.get_robot=get_robot
        self._robot=None; self._target=None; self._sols=[]; self._build()
    def _build(self):
        lay=QVBoxLayout(self); lay.setContentsMargins(22,20,22,20); lay.setSpacing(14)
        lay.addWidget(_t("Validate IK Solutions"))
        lay.addWidget(_h("Automatic validation runs after IK. Adjust tolerances below and re-run if needed."))
        lay.addWidget(_div())
        tr=QHBoxLayout()
        tr.addWidget(QLabel("Position tolerance (mm):"))
        self.spt=QDoubleSpinBox(); self.spt.setRange(1e-8,100); self.spt.setDecimals(4); self.spt.setValue(1.0)
        tr.addWidget(self.spt); tr.addSpacing(24); tr.addWidget(QLabel("Orientation tolerance:"))
        self.sot=QDoubleSpinBox(); self.sot.setRange(1e-8,100); self.sot.setDecimals(4); self.sot.setValue(0.1)
        tr.addWidget(self.sot); tr.addStretch()
        bv=QPushButton("Re-Validate"); bv.setObjectName("success"); bv.clicked.connect(self._validate)
        tr.addWidget(bv); lay.addLayout(tr)
        self.lsum=QLabel("Run Inverse Kinematics first to see results here.")
        self.lsum.setWordWrap(True)
        self.lsum.setStyleSheet(f"padding:12px;background:{C['panel2']};border-radius:7px;"
                                f"border:1px solid {C['border']};font-size:16px;font-weight:700;")
        lay.addWidget(self.lsum)
        sc=QScrollArea(); sc.setWidgetResizable(True)
        self.rw=QWidget(); self.rl=QVBoxLayout(self.rw); self.rl.setSpacing(12); self.rl.addStretch()
        sc.setWidget(self.rw); lay.addWidget(sc)
    def receive(self, robot, target, sols):
        self._robot=robot; self._target=target; self._sols=sols; self._validate()
    def _validate(self):
        robot=self._robot or self.get_robot()
        if not robot: self.lsum.setText("Validate a robot in Setup tab first."); return
        if not self._sols: self.lsum.setText("Solve Inverse Kinematics first."); return
        if self._target is None: return
        validated=validate_ik_solutions(robot,self._target,self._sols,self.spt.value(),self.sot.value())
        vn=sum(1 for v in validated if v.validation.valid)
        color=C["green"] if vn==len(validated) else (C["yellow"] if vn>0 else C["red"])
        self.lsum.setText(f"<b style='color:{color}'>{vn}/{len(validated)} solutions VALID</b>")
        while self.rl.count()>1:
            it=self.rl.takeAt(0)
            if it.widget(): it.widget().deleteLater()
        for idx,v in enumerate(validated):
            iv=v.validation.valid; gc=GC[idx%len(GC)]
            sc2=C["green"] if iv else C["red"]; st="VALID" if iv else "INVALID"
            grp=QGroupBox(f"  Solution {idx+1}  —  {st}")
            grp.setStyleSheet(f"QGroupBox{{border-color:{gc};border-width:2px;}}"
                              f"QGroupBox::title{{color:{sc2};font-size:15px;}}")
            gl=QGridLayout(grp); grp.setFont(QFont("Segoe UI",13))
            sol=v.solution; rep=v.validation
            lbl_q=QLabel("Joint angles:"); lbl_q.setStyleSheet("font-size:14px;")
            val_q=QLabel("[" + ", ".join(f"{q:.2f}" for q in sol.joint_state.positions) + "]")
            val_q.setStyleSheet(f"color:{C['accent']};font-family:Consolas;font-size:14px;")
            gl.addWidget(lbl_q,0,0); gl.addWidget(val_q,0,1,1,3)
            lbl_e=QLabel("Position error:"); lbl_e.setStyleSheet("font-size:14px;")
            val_e=QLabel(f"{sol.position_error:.6g} mm")
            val_e.setStyleSheet(f"color:{C['yellow']};font-family:Consolas;font-size:14px;")
            gl.addWidget(lbl_e,1,0); gl.addWidget(val_e,1,1)
            for i,(lt,ok) in enumerate([("Joint limits",rep.joint_limit_ok),("Position",rep.position_ok),
                                        ("Orientation",rep.orientation_ok),("Finite values",rep.finite_values_ok)]):
                lbl=QLabel(lt+":"); lbl.setStyleSheet("font-size:14px;")
                gl.addWidget(lbl,2+i//2,(i%2)*2); gl.addWidget(_ok_badge() if ok else _fail_badge(),2+i//2,(i%2)*2+1)
            if rep.messages:
                ml=QLabel("  •  ".join(rep.messages))
                ml.setStyleSheet(f"color:{C['tdim']};font-size:13px;"); ml.setWordWrap(True)
                gl.addWidget(ml,4,0,1,4)
            self.rl.insertWidget(self.rl.count()-1, grp)


# ── Tab 5: Full Run ───────────────────────────────────────────────────────────
class FullRunTab(QWidget):
    def __init__(self, get_robot=None):
        super().__init__(); self.get_robot=get_robot; self.cr=None; self._build()
    def _build(self):
        out=QHBoxLayout(self); out.setContentsMargins(20,20,20,20); out.setSpacing(20)
        L=QVBoxLayout(); L.setSpacing(12)
        L.addWidget(_t("Full Pipeline Run"))
        L.addWidget(_h("Runs all 6 pipeline modules end-to-end. Set demo angles below and click Run."))
        L.addWidget(_div())
        self.ga=QGroupBox("Demo Joint Angles"); self.al=QFormLayout(self.ga); self.asps=[]
        for i,a in enumerate(DEFAULT_ANGLES):
            s=QDoubleSpinBox(); s.setRange(-360,360); s.setDecimals(1); s.setSingleStep(1.); s.setValue(a)
            self.asps.append(s); self.al.addRow(f"q{i}:",s)
        L.addWidget(self.ga)
        gr=QGroupBox("Tolerances"); tl=QFormLayout(gr)
        self.spt=QDoubleSpinBox(); self.spt.setRange(1e-8,100); self.spt.setDecimals(4); self.spt.setValue(1.0)
        self.sot=QDoubleSpinBox(); self.sot.setRange(1e-8,100); self.sot.setDecimals(4); self.sot.setValue(0.1)
        tl.addRow("Position (mm):",self.spt); tl.addRow("Orientation:",self.sot); L.addWidget(gr)
        self.brun=QPushButton("▶  Run Full Pipeline"); self.brun.setObjectName("accent"); L.addWidget(self.brun)
        gs=QGroupBox("Quick Summary"); sl=QGridLayout(gs)
        c1,self.lee=_card("End-Effector","--"); c2,self.lsols=_card("Valid Solutions","--")
        sl.addWidget(c1,0,0); sl.addWidget(c2,0,1); L.addWidget(gs); L.addStretch()
        R=QVBoxLayout(); R.setSpacing(10)
        R.addWidget(_t("Pipeline Output",sz=13))
        self.tout=QTextEdit(); self.tout.setReadOnly(True); R.addWidget(self.tout,stretch=2)
        R.addWidget(_h("3D arm at demo state:"))
        self.canvas=ArmCanvas(w=4,h=3,dpi=85); R.addWidget(self.canvas,stretch=1)
        lw=QWidget(); lw.setLayout(L); rw=QWidget(); rw.setLayout(R)
        sp=QSplitter(Qt.Horizontal); sp.addWidget(lw); sp.addWidget(rw); sp.setSizes([300,640])
        out.addWidget(sp); self.brun.clicked.connect(self._run)
    def on_robot(self, robot):
        if not robot: return
        self.cr=robot
        while self.al.rowCount()>0: self.al.removeRow(0)
        self.asps=[]
        for i,j in enumerate(robot.joints):
            s=QDoubleSpinBox(); s.setRange(j.min_limit,j.max_limit); s.setDecimals(1)
            s.setSingleStep(1.); dv=j.theta if j.joint_type=="revolute" else j.d
            s.setValue(max(j.min_limit,min(j.max_limit,dv)))
            self.al.addRow(f"q{i}:",s); self.asps.append(s)
        self.canvas.set_reach(robot)
    def _run(self):
        robot=self.cr or (self.get_robot() if self.get_robot else None)
        if not robot:
            joints=[JointDefinition(0,"revolute",0,100,0,90,-180,180),
                    JointDefinition(1,"revolute",0,0,150,0,-90,90),
                    JointDefinition(2,"revolute",0,0,120,0,-120,120),
                    JointDefinition(3,"revolute",0,0,80,0,-180,180)]
            rname="Jane 4-DOF"
        else: joints=robot.joints; rname=robot.name
        demo=[s.value() for s in self.asps] if self.asps else [j.theta for j in joints]
        guesses=_make_guesses(robot if robot else create_robot_definition(rname,joints))
        self.brun.setText("Running..."); self.brun.setEnabled(False); QApplication.processEvents()
        buf=io.StringIO(); old=sys.stdout; sys.stdout=buf; validated=[]
        try: validated=run_pipeline(robot_name=rname,joint_definitions=joints,demo_joint_state=demo,
                                    ik_guesses=guesses,position_tolerance=self.spt.value(),
                                    orientation_tolerance=self.sot.value())
        except: buf.write(traceback.format_exc())
        finally: sys.stdout=old
        self.brun.setText("▶  Run Full Pipeline"); self.brun.setEnabled(True)
        self.tout.setPlainText(buf.getvalue()); self.tout.moveCursor(self.tout.textCursor().Start)
        vn=sum(1 for v in validated if v.validation.valid)
        color=C["green"] if vn>0 else C["red"]
        self.lsols.setText(f"{vn}/{len(validated)}")
        self.lsols.setStyleSheet(f"color:{color};font-size:16px;font-weight:800;font-family:Consolas;background:transparent;border:none;")
        try:
            cr=robot if robot else create_robot_definition(rname,joints)
            result=forward_kinematics(cr,JointState(demo)); pos=result.pose.position
            self.lee.setText(f"({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f})")
            self.canvas.draw_arm(cr,result)
        except: pass


# ── Main Window ───────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Jane Robot Manipulator  v1")
        self.resize(1400, 860)
        self._tabs = None
        self._build()

    def _build(self):
        self.sb=QStatusBar(); self.setStatusBar(self.sb)
        self.sb.showMessage("Welcome — start in the Setup tab, then move to Forward Kinematics.")
        tabs=QTabWidget(); tabs.setDocumentMode(True)
        self._tabs = tabs

        self.t1=SetupTab()
        self.t2=FKTab(self.t1.get_robot, self._send_to_ik)
        self.t3=IKTab(self.t1.get_robot, self.t2.get_fk, self._go_validate)
        self.t4=ValidateTab(self.t1.get_robot)
        self.t5=FullRunTab(self.t1.get_robot)

        # wire up signals
        self.t1.robot_changed.connect(self.t2.on_robot)
        self.t1.robot_changed.connect(self.t3.on_robot)
        self.t1.robot_changed.connect(self.t5.on_robot)
        self.t1.robot_changed.connect(lambda r: self.sb.showMessage(
            f"Robot validated: {r.name}  ({len(r.joints)} joints) — go to Forward Kinematics!"))
        self.t3.ik_done.connect(self.t4.receive)
        self.t3.ik_done.connect(lambda r,t,s: self.sb.showMessage(
            f"IK: {len(s)} solution(s) found — results auto-sent to Validate tab."))

        tabs.addTab(self.t1,"  Setup  ")
        tabs.addTab(self.t2,"  Forward Kinematics  ")
        tabs.addTab(self.t3,"  Inverse Kinematics  ")
        tabs.addTab(self.t4,"  Validate  ")
        tabs.addTab(self.t5,"  Full Run  ")

        tips=["Setup — configure robot joints and click Validate.",
              "Forward Kinematics — drag sliders to move joints live!",
              "Inverse Kinematics — enter a target XYZ and solve.",
              "Validate — check which IK solutions pass all constraints.",
              "Full Run — execute all 6 pipeline modules end-to-end."]
        tabs.currentChanged.connect(lambda i: self.sb.showMessage(tips[i]))
        self.setCentralWidget(tabs)
        self.t1._validate()   # auto-validate default robot on start

    def _send_to_ik(self, pos):
        """FK → IK: pre-fill target, then switch to IK tab."""
        self.t3.set_target(pos)
        self._tabs.setCurrentIndex(2)
        self.sb.showMessage("Target pre-filled from current FK position. Click Solve IK!")

    def _go_validate(self):
        self._tabs.setCurrentIndex(3)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    win = MainWindow(); win.show(); sys.exit(app.exec_())
