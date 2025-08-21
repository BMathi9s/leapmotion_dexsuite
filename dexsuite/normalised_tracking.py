# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import sys
import time
import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

# If your class is in another module, import it here:
# from ultraleap_hand_kinematics import UltraleapHandKinematics
from raw_hand_tracking import UltraleapHandKinematics   # <-- use this if your class lives here

# --- Safe YAML helpers (no hard dependency on PyYAML) ---
def _save_yaml_safe(path: str, data: dict) -> None:
    """
    Writes a YAML file. Uses PyYAML if available; otherwise writes JSON (valid YAML subset).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        import yaml  # type: ignore
        with open(path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=True, default_flow_style=False)
    except Exception:
        # Fallback: JSON is a valid subset of YAML
        with open(path, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)

def _load_yaml_safe(path: str) -> dict:
    """
    Reads a YAML file. Tries PyYAML then JSON.
    """
    with open(path, "r") as f:
        text = f.read()
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text)
    except Exception:
        return json.loads(text)

# --- Small math helpers ---
def _linmap_clamped(x: float, a: float, b: float, A: float, B: float) -> float:
    """Map x from [a,b] to [A,B] and clamp to [A,B]. Handles swapped a/b."""
    if a == b:
        raise RuntimeError("Calibration min == max; cannot normalize.")
    if a > b:
        a, b = b, a
    t = (x - a) / (b - a)
    if t < 0.0: t = 0.0
    if t > 1.0: t = 1.0
    return A + (B - A) * t

# --- Dataclasses for saved files ---
@dataclass
class AxisCal:
    min: float
    max: float

@dataclass
class WorldFrameCal:
    type: str
    units: str
    axes: Dict[str, AxisCal]          # keys: 'x','y','z'
    created_utc: float
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        # dataclasses nested -> convert AxisCal entries
        d['axes'] = {k: asdict(v) for k, v in self.axes.items()}
        return d

@dataclass
class RangeCal:
    min: Optional[float] = None
    max: Optional[float] = None

    def update(self, v: Optional[float]) -> None:
        if v is None:
            return
        if self.min is None or v < self.min: self.min = v
        if self.max is None or v > self.max: self.max = v

    def as_tuple(self) -> Tuple[float, float]:
        if self.min is None or self.max is None:
            raise RuntimeError("Range not captured.")
        return (self.min, self.max)

@dataclass
class FingerJointCal:
    mcp: RangeCal
    pip: RangeCal
    dip: RangeCal
    ip:  RangeCal           # mainly for thumb; harmless to keep for others
    abduction: RangeCal

@dataclass
class HandCal:
    type: str
    fingers: Dict[str, FingerJointCal]   # 'thumb','index','middle','ring','pinky'
    created_utc: float

    def to_dict(self) -> dict:
        def rc_to_d(rc: RangeCal) -> dict:
            return {"min": rc.min, "max": rc.max}
        out = {
            "type": self.type,
            "created_utc": self.created_utc,
            "fingers": {}
        }
        for name, fj in self.fingers.items():
            out["fingers"][name] = {
                "mcp": rc_to_d(fj.mcp),
                "pip": rc_to_d(fj.pip),
                "dip": rc_to_d(fj.dip),
                "ip":  rc_to_d(fj.ip),
                "abduction": rc_to_d(fj.abduction),
            }
        return out

class CalibrationAndNormalization:
    """
    A thin utility around UltraleapHandKinematics to:
      - Calibrate world-frame ranges (X/Y/Z)
      - Calibrate per-finger ranges (MCP/PIP/DIP, thumb IP, abduction)
      - Normalize values: X/Y in [-1,1], Z in [0,1], joints & abductions in [-1,1]
    """

    FINGERS = ("thumb","index","middle","ring","pinky")
    JOINTS  = ("mcp","pip","dip")
    BASE_DIR = "/home/dexsuite/leapmtion/leapc-python-bindings"
    WORLD_DIR = os.path.join(BASE_DIR, "calibration", "worldframe")
    HAND_DIR  = os.path.join(BASE_DIR, "calibration", "hand_calibration")

    def __init__(self, kin: UltraleapHandKinematics):
        self.k = kin
        self.world_cal: Optional[WorldFrameCal] = None
        self.hand_cal:  Optional[HandCal] = None

    # ----------------- Normalized getters (with checks) -----------------
    def get_normalized_xyz(self, use_raw_world: bool = True) -> Tuple[float,float,float]:
        """
        Return (nx, ny, nz) with X,Y in [-1,1], Z in [0,1].
        - If use_raw_world=True, use raw world XYZ from Leap.
        - If False, uses origin-corrected kinematics XYZ.
        """
        self._require_world_cal()
        xyz = self.k.get_xyz(raw=use_raw_world)
        if xyz is None:
            raise RuntimeError("No hand visible.")
        x, y, z = xyz
        ax = self.world_cal.axes["x"]; ay = self.world_cal.axes["y"]; az = self.world_cal.axes["z"]
        nx = _linmap_clamped(x, ax.min, ax.max, -10.0,  10.0)
        ny = _linmap_clamped(y, ay.min, ay.max, 0.0,  10.0)
        nz = _linmap_clamped(z, az.min, az.max,  -10.0,  10.0)
        return (nx, ny, nz)

    def get_normalized_joint(self, finger: str, joint: str) -> float:
        """
        Normalize joint angle for (finger, joint) into [-1,1].
        joint ∈ {'mcp','pip','dip'}; for thumb you may prefer get_normalized_thumb_ip().
        """
        self._require_hand_cal()
        finger = finger.lower()
        if finger not in self.FINGERS:
            raise ValueError("Unknown finger: %s" % finger)
        if joint not in self.JOINTS:
            raise ValueError("Unknown joint: %s" % joint)

        ang = self.k.get_joint_angle(finger, joint)
        if ang is None:
            raise RuntimeError("Angle unavailable for %s.%s" % (finger, joint))

        rc = self._get_fj(finger).__dict__[joint]
        mn, mx = rc.as_tuple()
        return _linmap_clamped(ang, mn, mx, -1.0, 1.0)

    def get_normalized_thumb_ip(self) -> float:
        self._require_hand_cal()
        ang = (self.k.get_thumb_angles() or {}).get("ip")
        if ang is None:
            raise RuntimeError("Thumb IP angle unavailable.")
        rc = self._get_fj("thumb").ip
        mn, mx = rc.as_tuple()
        return _linmap_clamped(ang, mn, mx, -1.0, 1.0)

    def get_normalized_abduction(self, finger: str) -> float:
        """
        Normalize abduction for finger into [-1,1].
        """
        self._require_hand_cal()
        finger = finger.lower()
        ang = self.k.get_abduction(finger)
        if ang is None:
            raise RuntimeError("Abduction unavailable for %s" % finger)
        rc = self._get_fj(finger).abduction
        mn, mx = rc.as_tuple()
        return _linmap_clamped(ang, mn, mx, -1.0, 1.0)

    def get_all_normalized_joints(self) -> Dict[str, Dict[str, Optional[float]]]:
        out: Dict[str, Dict[str, Optional[float]]] = {}
        for f in self.FINGERS:
            out[f] = {}
            for j in self.JOINTS:
                try:
                    out[f][j] = self.get_normalized_joint(f, j)
                except Exception:
                    out[f][j] = None
            if f == "thumb":
                try:
                    out[f]['ip'] = self.get_normalized_thumb_ip()
                except Exception:
                    out[f]['ip'] = None
        return out

    def get_all_normalized_abductions(self) -> Dict[str, Optional[float]]:
        out = {}
        for f in self.FINGERS:
            try:
                out[f] = self.get_normalized_abduction(f)
            except Exception:
                out[f] = None
        return out

    # ----------------- Wizards -----------------
    def run_worldframe_wizard(self) -> None:
        print("\n=== World-frame Calibration Wizard ===")
        choice = self._ask_choice("(L)oad existing or (N)ew calibration? [l/n]: ", ("l","n"))
        if choice == "l":
            self._load_worldframe_interactive()
        else:
            self._new_worldframe_interactive()
            
    def format_range(self,val):
        return f"{val:.1f}" if val is not None else "nan"

    def run_finger_calibration_wizard(self) -> None:
        print("\n=== Finger Calibration Wizard ===")
        choice = self._ask_choice("(L)oad existing or (N)ew calibration? [l/n]: ", ("l","n"))
        if choice == "l":
            self._load_hand_interactive()
        else:
            self._new_hand_interactive()

    # ----------------- Internals: world-frame (XYZ) -----------------
    def _new_worldframe_interactive(self) -> None:
        """
        Capture min/max for X/Y/Z using keys 1..6 on the console.
        1=minX 2=maxX 3=minY 4=maxY 5=minZ 6=maxZ  s=save  p=print  q=quit
        """
        print("\nNew world-frame calibration.")
        print("Instructions:")
        print("  - Move your chosen hand into the position you want to capture,")
        print("  - Then press a key to assign the current palm position:")
        print("      1=minX  2=maxX  3=minY  4=maxY  5=minZ  6=maxZ")
        print("      p=print current values,  s=save,  q=abort\n")

        vals = {"x": AxisCal(float('nan'), float('nan')),
                "y": AxisCal(float('nan'), float('nan')),
                "z": AxisCal(float('nan'), float('nan'))}

        while True:
            key = input("[1..6/p/s/q] > ").strip().lower()
            if key == "q":
                print("Aborted. Nothing saved.")
                return
            if key == "p":
                self._print_world_progress(vals)
                continue
            if key in {"1","2","3","4","5","6"}:
                xyz = self.k.get_xyz(raw=True)
                if xyz is None:
                    print("  ✖ No hand visible. Try again.")
                    continue
                x, y, z = xyz
                if key == "1": vals["x"].min = x
                elif key == "2": vals["x"].max = x
                elif key == "3": vals["y"].min = y
                elif key == "4": vals["y"].max = y
                elif key == "5": vals["z"].min = z
                elif key == "6": vals["z"].max = z
                self._print_world_progress(vals)
                continue
            if key == "s":
                # validate
                for axis in ("x","y","z"):
                    mn, mx = vals[axis].min, vals[axis].max
                    if math.isnan(mn) or math.isnan(mx):
                        print(f"  ✖ Axis {axis.upper()} missing values. Capture all six before saving.")
                        break
                else:
                    # save
                    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
                    default_name = f"world_{ts}.yaml"
                    name = input(f"Save as [{default_name}]: ").strip()
                    if not name:
                        name = default_name
                    if not name.endswith(".yaml"):
                        name += ".yaml"
                    wf = WorldFrameCal(
                        type="worldframe",
                        units="mm",
                        axes=vals,
                        created_utc=time.time(),
                        notes="generated by CalibrationAndNormalization"
                    )
                    path = os.path.join(self.WORLD_DIR, name)
                    _save_yaml_safe(path, wf.to_dict())
                    self.world_cal = wf
                    print(f"  ✔ Saved world-frame calibration -> {path}")
                    return

    def _print_world_progress(self, vals: Dict[str, AxisCal]) -> None:
        def _fmt(v): return "NA" if math.isnan(v) else f"{v:7.1f}"
        print(f"    X: min={_fmt(vals['x'].min)}  max={_fmt(vals['x'].max)}")
        print(f"    Y: min={_fmt(vals['y'].min)}  max={_fmt(vals['y'].max)}")
        print(f"    Z: min={_fmt(vals['z'].min)}  max={_fmt(vals['z'].max)}")

    def _load_worldframe_interactive(self) -> None:
        files = self._list_yaml(self.WORLD_DIR)
        if not files:
            print("  (No world-frame calibrations found; switching to New.)")
            self._new_worldframe_interactive()
            return
        idx = self._pick_file(files)
        path = os.path.join(self.WORLD_DIR, files[idx])
        data = _load_yaml_safe(path)
        axes = {
            "x": AxisCal(**data["axes"]["x"]),
            "y": AxisCal(**data["axes"]["y"]),
            "z": AxisCal(**data["axes"]["z"])
        }
        self.world_cal = WorldFrameCal(
            type="worldframe",
            units=data.get("units","mm"),
            axes=axes,
            created_utc=data.get("created_utc", time.time()),
            notes=data.get("notes","")
        )
        print(f"  ✔ Loaded world-frame calibration from {path}")

    # ----------------- Internals: finger calibration -----------------
    def _new_hand_interactive(self) -> None:
        print("\nNew finger calibration.")
        print("Instructions:")
        print("  - For each finger, enter a sampling duration (seconds).")
        print("  - During sampling, move that finger through FULL range (open <-> fully curled),")
        print("    and splay left/right for abduction. Keep your other fingers as neutral as possible.\n")
        
        
        print(f"DEBUG: self.FINGERS = {self.FINGERS}")
        fingers: Dict[str, FingerJointCal] = {}
        
        for name in self.FINGERS:
            print(f"DEBUG: Processing finger name = {name}")  # Add this line
            if name is None:  # Add this check
                print("WARNING: Found None in FINGERS list, skipping...")
                continue    
            dur_s = self._ask_float(f"  {name.capitalize()} sampling seconds [default 5]: ", default=5.0, minv=0.5, maxv=20.0)
            fj = FingerJointCal(mcp=RangeCal(), pip=RangeCal(), dip=RangeCal(), ip=RangeCal(), abduction=RangeCal())
            t0 = time.time()
            n_valid = 0
            while time.time() - t0 < dur_s:
                # joints
                for j in self.JOINTS:
                    try:
                        ang = self.k.get_joint_angle(name, j)
                        if ang is not None:
                            getattr(fj, j).update(ang)
                            n_valid += 1
                    except Exception:
                        pass
                # thumb IP convenience
                if name == "thumb":
                    th = self.k.get_thumb_angles() or {}
                    if th.get("ip") is not None:
                        fj.ip.update(th["ip"])
                        n_valid += 1
                # abduction
                try:
                    abd = self.k.get_abduction(name)
                    if abd is not None:
                        fj.abduction.update(abd)
                        n_valid += 1
                except Exception:
                    pass
                time.sleep(0.02)

            if n_valid == 0:
                print(f"  ✖ No valid samples for {name}. Repeating.")
                return self._new_hand_interactive()

            # Sanity check, swap if min>max
            for rc in (fj.mcp, fj.pip, fj.dip, fj.ip, fj.abduction):
                if rc.min is not None and rc.max is not None and rc.min > rc.max:
                    rc.min, rc.max = rc.max, rc.min

            fingers[name] = fj
            print("\nFinger calibration complete. Summary:")
            print(f"  ✔ {name.capitalize()} captured: "
                f"mcp[{self.format_range(fj.mcp.min)},{self.format_range(fj.mcp.max)}] "
                f"pip[{self.format_range(fj.pip.min)},{self.format_range(fj.pip.max)}] "
                f"dip[{self.format_range(fj.dip.min)},{self.format_range(fj.dip.max)}] "
                f"ip[{self.format_range(fj.ip.min)},{self.format_range(fj.ip.max)}] "
                f"abd[{self.format_range(fj.abduction.min)},{self.format_range(fj.abduction.max)}]")

        # Save
        ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        default_name = f"hand_{ts}.yaml"
        filename = input(f"\nSave hand calibration as [{default_name}]: ").strip()
        if not filename:
            filename = default_name
        if not filename.endswith(".yaml"):
            filename += ".yaml"
        hc = HandCal(type="hand_calibration", fingers=fingers, created_utc=time.time())
        path = os.path.join(self.HAND_DIR, filename)
        _save_yaml_safe(path, hc.to_dict())
        self.hand_cal = hc
        print(f"  ✔ Saved hand calibration -> {path}")

    def _load_hand_interactive(self) -> None:
        files = self._list_yaml(self.HAND_DIR)
        if not files:
            print("  (No hand calibrations found; switching to New.)")
            self._new_hand_interactive()
            return
        idx = self._pick_file(files)
        path = os.path.join(self.HAND_DIR, files[idx])
        data = _load_yaml_safe(path)

        fingers: Dict[str, FingerJointCal] = {}
        for name in self.FINGERS:
            fd = data["fingers"][name]
            fj = FingerJointCal(
                mcp=RangeCal(fd["mcp"]["min"], fd["mcp"]["max"]),
                pip=RangeCal(fd["pip"]["min"], fd["pip"]["max"]),
                dip=RangeCal(fd["dip"]["min"], fd["dip"]["max"]),
                ip= RangeCal(fd["ip"]["min"],  fd["ip"]["max"]),
                abduction=RangeCal(fd["abduction"]["min"], fd["abduction"]["max"]),
            )
            fingers[name] = fj
        self.hand_cal = HandCal(type="hand_calibration", fingers=fingers, created_utc=data.get("created_utc", time.time()))
        print(f"  ✔ Loaded hand calibration from {path}")

    # ----------------- Utility helpers -----------------
    def _get_fj(self, finger: str) -> FingerJointCal:
        if not self.hand_cal:
            raise RuntimeError("Hand calibration not loaded.")
        return self.hand_cal.fingers[finger]

    def _require_world_cal(self) -> None:
        if self.world_cal is None:
            raise RuntimeError("World-frame calibration not loaded. Run run_worldframe_wizard() first.")

    def _require_hand_cal(self) -> None:
        if self.hand_cal is None:
            raise RuntimeError("Hand calibration not loaded. Run run_finger_calibration_wizard() first.")
        


    @staticmethod
    def _ask_choice(prompt: str, valid: Tuple[str, ...]) -> str:
        valid = tuple(v.lower() for v in valid)
        while True:
            s = input(prompt).strip().lower()
            if s in valid:
                return s
            print("  (invalid choice)")

    @staticmethod
    def _ask_float(prompt: str, default: float, minv: float, maxv: float) -> float:
        while True:
            s = input(prompt).strip()
            if not s:
                return default
            try:
                v = float(s)
                if not (minv <= v <= maxv):
                    print(f"  (enter between {minv} and {maxv})")
                    continue
                return v
            except Exception:
                print("  (enter a number)")

    @staticmethod
    def _list_yaml(dir_path: str):
        if not os.path.isdir(dir_path):
            return []
        return sorted([f for f in os.listdir(dir_path) if f.lower().endswith(".yaml")])

    @staticmethod
    def _pick_file(files):
        print("\nAvailable calibrations:")
        for i, f in enumerate(files):
            print(f"  {i}) {f}")
        while True:
            s = input("Select index: ").strip()
            try:
                idx = int(s)
                if 0 <= idx < len(files):
                    return idx
                print("  (index out of range)")
            except Exception:
                print("  (enter a number)")
