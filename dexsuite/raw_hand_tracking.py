from __future__ import annotations
import math
import threading
import time
from typing import Dict, Optional, Tuple

import leap  # Ultraleap leapc-python-bindings

RAD2DEG = 180.0 / math.pi
EPS = 1e-6

def _v_sub(a, b):
    return (float(a.x) - float(b.x), float(a.y) - float(b.y), float(a.z) - float(b.z))

def _v_dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def _v_len(v):
    return math.sqrt(_v_dot(v, v))

def _v_unit(v):
    L = _v_len(v)
    if L < EPS:
        return (0.0, 0.0, 0.0)
    return (v[0]/L, v[1]/L, v[2]/L)

def _v_cross(a, b):
    return (a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0])

def _angle_between(u, v) -> Optional[float]:
    """Inner angle (deg) between unit vectors u and v. Returns None if degenerate."""
    Lu, Lv = _v_len(u), _v_len(v)
    if Lu < EPS or Lv < EPS:
        return None
    d = max(-1.0, min(1.0, (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) / (Lu * Lv)))
    return math.degrees(math.acos(d))

def _bone_dir(bone) -> Tuple[float, float, float]:
    """Unit direction from prev_joint -> next_joint."""
    pj, nj = bone.prev_joint, bone.next_joint
    return _v_unit(_v_sub(nj, pj))

def _hand_is_right(hand) -> bool:
    # HandType enum prints like "HandType.Right"
    return "Right" in str(hand.type)

def _hand_is_left(hand) -> bool:
    return "Left" in str(hand.type)

class _LatestTracking(leap.Listener):
    """Keeps only the latest tracking event, thread-safe."""
    def __init__(self):
        self._event = None
        self._lock = threading.Lock()

    @property
    def event(self):
        with self._lock:
            return self._event

    def on_tracking_event(self, event):
        with self._lock:
            self._event = event

class UltraleapHandKinematics:
    """
    High-level kinematics for a single hand using Ultraleap LeapC Python bindings.

    Features:
      - init(hand='right'|'left'|'any', tracking_mode=Desktop/HMD/ScreenTop)
      - set_origin_xyz(...) / set_origin_rpy(...) calibration offsets
      - get_xyz(raw=False) and get_rpy(raw=False)
      - per-joint curl angles: MCP/PIP/DIP (inner angles, degrees)
      - abduction (signed, degrees): + toward hand's right, - toward left
      - thumb getters incl. an extra 'ip' (proximal<->distal) convenience angle
    """

    _DIGIT_NAMES = ('thumb', 'index', 'middle', 'ring', 'pinky')
    _JOINTS = ('mcp', 'pip', 'dip')

    def __init__(self,
                 hand: str = 'right',
                 tracking_mode = leap.TrackingMode.Desktop):
        assert hand in ('right', 'left', 'any')
        self.hand_pref = hand
        self.tracking_mode = tracking_mode

        # Calibration origins
        self._origin_xyz = (0.0, 0.0, 0.0)     # mm
        self._origin_rpy = (0.0, 0.0, 0.0)     # deg (roll, pitch, yaw)

        # Connection + listener
        self._listener = _LatestTracking()
        self._conn = leap.Connection()
        self._conn.add_listener(self._listener)
        self._cm = None  # context manager handle

    # -------------- lifecycle --------------
    def __enter__(self):
        # Open connection and set tracking mode
        self._cm = self._conn.open()
        self._cm.__enter__()
        self._conn.set_tracking_mode(self.tracking_mode)
        # Wait briefly for first frame
        t0 = time.time()
        while self._listener.event is None and (time.time() - t0) < 3.0:
            time.sleep(0.01)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._cm:
            self._cm.__exit__(exc_type, exc, tb)
            self._cm = None

    # -------------- calibration --------------
    def set_origin_xyz(self, x: Optional[float]=None, y: Optional[float]=None,
                       z: Optional[float]=None, use_current: bool=False) -> bool:
        """Set XYZ origin. If use_current=True, capture current palm position."""
        if use_current:
            p = self.get_xyz(raw=True)
            if p is None:
                return False
            self._origin_xyz = p
            return True
        if x is None or y is None or z is None:
            return False
        self._origin_xyz = (float(x), float(y), float(z))
        return True

    def set_origin_rpy(self, roll: Optional[float]=None, pitch: Optional[float]=None,
                       yaw: Optional[float]=None, use_current: bool=False) -> bool:
        """Set RPY origin (deg). If use_current=True, capture current RPY."""
        if use_current:
            rpy = self.get_rpy(raw=True)
            if rpy is None:
                return False
            self._origin_rpy = rpy
            return True
        if roll is None or pitch is None or yaw is None:
            return False
        self._origin_rpy = (float(roll), float(pitch), float(yaw))
        return True

    # -------------- hand selection --------------
    def _get_hand(self):
        ev = self._listener.event
        if ev is None or len(ev.hands) == 0:
            return None
        # Choose by preference
        if self.hand_pref == 'any':
            return ev.hands[0]
        for h in ev.hands:
            if self.hand_pref == 'right' and _hand_is_right(h):
                return h
            if self.hand_pref == 'left' and _hand_is_left(h):
                return h
        # fallback
        return ev.hands[0]

    # -------------- pose getters --------------
    def get_xyz(self, raw: bool=False) -> Optional[Tuple[float,float,float]]:
        """Palm XYZ (mm). If raw=False, subtract origin."""
        hand = self._get_hand()
        if hand is None:
            return None
        p = hand.palm.position
        xyz = (float(p.x), float(p.y), float(p.z))
        if raw:
            return xyz
        ox, oy, oz = self._origin_xyz
        return (xyz[0]-ox, xyz[1]-oy, xyz[2]-oz)

    def get_rpy(self, raw: bool=False) -> Optional[Tuple[float,float,float]]:
        """
        Roll, Pitch, Yaw (deg) from palm normal/direction.
        roll from palm normal; pitch,yaw from palm direction.
        """
        hand = self._get_hand()
        if hand is None:
            return None
        d = hand.palm.direction
        n = hand.palm.normal

        dx, dy, dz = float(d.x), float(d.y), float(d.z)
        nx, ny, _  = float(n.x), float(n.y), float(n.z)

        # Same convention you used in your sample
        pitch = math.atan2(dy, math.sqrt(dx*dx + dz*dz)) * RAD2DEG
        yaw   = math.atan2(dx, -dz) * RAD2DEG
        roll  = math.atan2(nx, -ny) * RAD2DEG

        if raw:
            return (roll, pitch, yaw)
        or_, op, oyaw = self._origin_rpy
        return (roll - or_, pitch - op, yaw - oyaw)

    # -------------- joints & curl --------------
    def _digit_by_name(self, hand, name: str):
        idx = self._DIGIT_NAMES.index(name)
        return hand.digits[idx] if idx < len(hand.digits) else None

    def _bones_list(self, digit):
        """
        Return bones as [metacarpal, proximal, intermediate, distal].
        Handles either digit.bones[...] or named attributes.
        """
        if hasattr(digit, "bones"):
            return list(digit.bones)
        parts = []
        for nm in ("metacarpal", "proximal", "intermediate", "distal"):
            parts.append(getattr(digit, nm))
        return parts

    def get_joint_angle(self, finger_name: str, joint: str) -> Optional[float]:
        """
        joint in {'mcp','pip','dip'}
        Returns inner angle (degrees), or None if bone(s) are degenerate.
        Thumb: MCP can be None (zero-length metacarpal).
        """
        assert joint in self._JOINTS
        hand = self._get_hand()
        if hand is None:
            return None
        digit = self._digit_by_name(hand, finger_name)
        if digit is None:
            return None
        bones = self._bones_list(digit)  # 0:META,1:PROX,2:INTER,3:DIST

        if joint == 'mcp':
            a, b = bones[0], bones[1]
        elif joint == 'pip':
            a, b = bones[1], bones[2]
        else:  # 'dip'
            a, b = bones[2], bones[3]

        u = _bone_dir(a)
        v = _bone_dir(b)
        return _angle_between(u, v)

    def get_all_joint_angles(self) -> Dict[str, Dict[str, Optional[float]]]:
        """
        { finger: { 'mcp':deg or None, 'pip':deg or None, 'dip':deg or None, (thumb adds 'ip') } }
        """
        out: Dict[str, Dict[str, Optional[float]]] = {}
        for name in self._DIGIT_NAMES:
            d: Dict[str, Optional[float]] = {}
            for j in self._JOINTS:
                d[j] = self.get_joint_angle(name, j)
            if name == 'thumb':
                th = self.get_thumb_angles()
                if th is not None:
                    d.update(th)  # includes 'ip'
            out[name] = d
        return out

    # Per-joint convenience getters (explicit methods)
    def get_index_mcp(self):  return self.get_joint_angle('index','mcp')
    def get_index_pip(self):  return self.get_joint_angle('index','pip')
    def get_index_dip(self):  return self.get_joint_angle('index','dip')
    def get_middle_mcp(self): return self.get_joint_angle('middle','mcp')
    def get_middle_pip(self): return self.get_joint_angle('middle','pip')
    def get_middle_dip(self): return self.get_joint_angle('middle','dip')
    def get_ring_mcp(self):   return self.get_joint_angle('ring','mcp')
    def get_ring_pip(self):   return self.get_joint_angle('ring','pip')
    def get_ring_dip(self):   return self.get_joint_angle('ring','dip')
    def get_pinky_mcp(self):  return self.get_joint_angle('pinky','mcp')
    def get_pinky_pip(self):  return self.get_joint_angle('pinky','pip')
    def get_pinky_dip(self):  return self.get_joint_angle('pinky','dip')

    # -------------- thumb helpers --------------
    def get_thumb_angles(self) -> Optional[Dict[str, Optional[float]]]:
        """
        Returns dict for the thumb:
          - 'mcp' (may be None if metacarpal is zero-length)
          - 'pip', 'dip' (always computed like other fingers)
          - 'ip'  (convenience: proximal<->distal angle)
        """
        hand = self._get_hand()
        if hand is None:
            return None
        finger = self._digit_by_name(hand, 'thumb')
        if finger is None:
            return None
        bones = self._bones_list(finger)
        meta, prox, inter, dist = bones

        out: Dict[str, Optional[float]] = {}
        out['mcp'] = _angle_between(_bone_dir(meta), _bone_dir(prox))  # often None (zero-length meta) :contentReference[oaicite:4]{index=4}
        out['pip'] = _angle_between(_bone_dir(prox), _bone_dir(inter))
        out['dip'] = _angle_between(_bone_dir(inter), _bone_dir(dist))
        # Collapsed IP angle (proximal <-> distal) — handy for thumb controls
        out['ip']  = _angle_between(_bone_dir(prox), _bone_dir(dist))
        return out

    def get_thumb_mcp(self): return (self.get_thumb_angles() or {}).get('mcp')
    def get_thumb_pip(self): return (self.get_thumb_angles() or {}).get('pip')
    def get_thumb_dip(self): return (self.get_thumb_angles() or {}).get('dip')
    def get_thumb_ip(self):  return (self.get_thumb_angles() or {}).get('ip')

    # -------------- abduction (splay) --------------
    def get_abduction(self, finger_name: str) -> Optional[float]:
        """
        Signed abduction angle (deg) in the palm plane.
        0 = forward (palm direction); + toward hand's right; - toward hand's left.
        """
        hand = self._get_hand()
        if hand is None:
            return None
        digit = self._digit_by_name(hand, finger_name)
        if digit is None:
            return None

        # Palm basis
        n = hand.palm.normal
        z = hand.palm.direction
        n_hat = _v_unit((float(n.x), float(n.y), float(n.z)))
        z_hat = _v_unit((float(z.x), float(z.y), float(z.z)))
        x_hat = _v_unit(_v_cross(n_hat, z_hat))  # palm right axis (normal × direction) :contentReference[oaicite:5]{index=5}

        # Use proximal bone to represent the finger axis near the hand
        bones = self._bones_list(digit)
        v = _bone_dir(bones[1])  # proximal
        if _v_len(v) < EPS:
            v = _bone_dir(digit.proximal) if hasattr(digit, "proximal") else v

        # Project onto palm plane (remove component along normal)
        v_proj = (v[0] - n_hat[0]*_v_dot(v, n_hat),
                  v[1] - n_hat[1]*_v_dot(v, n_hat),
                  v[2] - n_hat[2]*_v_dot(v, n_hat))
        if _v_len(v_proj) < EPS:
            return 0.0
        v_proj = _v_unit(v_proj)

        ang = math.atan2(_v_dot(v_proj, x_hat), _v_dot(v_proj, z_hat)) * RAD2DEG
        return ang

    def get_all_abductions(self) -> Dict[str, Optional[float]]:
        return {name: self.get_abduction(name) for name in self._DIGIT_NAMES}
