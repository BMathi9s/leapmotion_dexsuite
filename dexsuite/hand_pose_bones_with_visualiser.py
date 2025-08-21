# leap_hand_tracker.py
import math
import time
from typing import Optional, Tuple, Dict

import leap  # Ultraleap LeapC python bindings

RAD2DEG = 180.0 / math.pi
DEG2RAD = math.pi / 180.0

def v3(x, y=None, z=None):
    if y is None:
        # assume (x) is a leap vec or 3-tuple
        p = x
        if hasattr(p, "x"):
            return (float(p.x), float(p.y), float(p.z))
        return (float(p[0]), float(p[1]), float(p[2]))
    return (float(x), float(y), float(z))

def sub(a, b):  # a - b
    ax, ay, az = v3(a); bx, by, bz = v3(b)
    return (ax - bx, ay - by, az - bz)

def add(a, b):  # a + b
    ax, ay, az = v3(a); bx, by, bz = v3(b)
    return (ax + bx, ay + by, az + bz)

def dot(a, b):
    ax, ay, az = v3(a); bx, by, bz = v3(b)
    return ax*bx + ay*by + az*bz

def cross(a, b):
    ax, ay, az = v3(a); bx, by, bz = v3(b)
    return (ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bx)

def norm(a):
    ax, ay, az = v3(a)
    return math.sqrt(ax*ax + ay*ay + az*az)

def normalize(a):
    n = norm(a)
    if n <= 1e-8:
        return (0.0, 0.0, 0.0)
    ax, ay, az = v3(a)
    return (ax/n, ay/n, az/n)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def angle_between(a, b):
    """Unsigned angle (rad) between unit vectors a,b."""
    ua, ub = normalize(a), normalize(b)
    c = clamp(dot(ua, ub), -1.0, 1.0)
    return math.acos(c)

def angle_between_deg(a, b):  # unsigned
    return angle_between(a, b) * RAD2DEG

def signed_angle_in_plane(u, v, plane_normal):
    """
    Signed angle u->v (rad) in the plane with given normal.
    Positive if rotation from u to v is in the direction of the normal (right-hand rule).
    """
    n = normalize(plane_normal)
    u_proj = sub(u, tuple(dot(u, n)*c for c in n))
    v_proj = sub(v, tuple(dot(v, n)*c for c in n))
    u_proj = normalize(u_proj)
    v_proj = normalize(v_proj)
    # atan2( ||u×v||, u·v ) gives unsigned, sign via (u×v)·n
    x = dot(cross(u_proj, v_proj), n)
    y = clamp(dot(u_proj, v_proj), -1.0, 1.0)
    return math.atan2(x, y)

def rpy_from_palm(direction, normal):
    """
    Matches your sample's idea (roll, pitch, yaw) using palm direction & normal.
    Returns radians.
    """
    dx, dy, dz = v3(direction)
    nx, ny, _  = v3(normal)
    pitch = math.atan2(dy, math.sqrt(dx*dx + dz*dz))
    yaw   = math.atan2(dx, -dz)
    roll  = math.atan2(nx, -ny)
    return roll, pitch, yaw

def bone_dir(bone):
    """Unit vector from prev_joint -> next_joint."""
    return normalize(sub(bone.next_joint, bone.prev_joint))

def joint_inner_angle_deg(bone_prox, bone_dist):
    """
    Angle at the joint where bone_prox meets bone_dist.
    We take angle between (-dir_prox) and (dir_dist) to get the inner bend angle.
    """
    u1 = bone_dir(bone_prox)
    u2 = bone_dir(bone_dist)
    c = clamp(dot(tuple(-c for c in u1), u2), -1.0, 1.0)
    return math.degrees(math.acos(c))

def digit_name(d):
    """Return lower-case name like 'thumb','index','middle','ring','pinky' from digit.type."""
    t = str(getattr(d, "type", ""))
    t = t.lower()
    for key in ["thumb", "index", "middle", "ring", "pinky", "little"]:
        if key in t:
            return "pinky" if key == "little" else key
    return ""  # fallback if unknown

class _SingleHandListener(leap.Listener):
    def __init__(self, want_hand: str = "right"):
        super().__init__()
        self.want_hand = want_hand.lower()  # 'right' or 'left'
        self.latest_hand = None
        self.frame_id = None

    def set_hand(self, want_hand: str):
        self.want_hand = want_hand.lower()

    def on_connection_event(self, event):
        print("Connected to Ultraleap service")

    def on_device_event(self, event):
        try:
            with event.device.open():
                info = event.device.get_info()
        except leap.LeapCannotOpenDeviceError:
            info = event.device.get_info()
        print(f"Found device {info.serial}")

    def on_tracking_event(self, event):
        # Pick only the desired hand
        chosen = None
        for hand in event.hands:
            htype = "left" if "left" in str(hand.type).lower() else "right"
            if htype == self.want_hand:
                chosen = hand
                break
        self.latest_hand = chosen
        self.frame_id = event.tracking_frame_id

class LeapHandTracker:
    """
    - Start/stop connection
    - Choose hand ('right' default)
    - Get calibrated XYZ / RPY
    - Get per-finger MCP/PIP/DIP curl and abduction
    - Thumb: MCP curl, IP curl, and CMC(out-of-plane) 'abduction' (thumb lift)
    """
    def __init__(self, hand: str = "right", tracking_mode=leap.TrackingMode.Desktop):
        self.listener = _SingleHandListener(hand)
        self.conn = leap.Connection()
        self.conn.add_listener(self.listener)
        self.tracking_mode = tracking_mode

        # Origins (offsets)
        self._pos_origin = (0.0, 0.0, 0.0)  # mm
        self._rpy_origin = (0.0, 0.0, 0.0)  # rad

    # ---- lifecycle ----
    def start(self):
        self._conn_ctx = self.conn.open()  # context manager
        self.conn.set_tracking_mode(self.tracking_mode)

    def stop(self):
        # Exit context manager to close
        if hasattr(self, "_conn_ctx"):
            self._conn_ctx.__exit__(None, None, None)

    def set_hand(self, hand: str):
        """Switch to 'right' or 'left' at runtime."""
        self.listener.set_hand(hand)

    # ---- presence ----
    def hand_present(self) -> bool:
        return self.listener.latest_hand is not None

    # ---- calibration ----
    def set_xyz_origin(self):
        """Set the current palm position as origin (XYZ -> 0,0,0 after this)."""
        hand = self.listener.latest_hand
        if not hand: return
        pos = v3(hand.palm.position)
        self._pos_origin = pos

    def set_rpy_origin(self):
        """Set current palm RPY as origin (RPY -> 0,0,0 after this)."""
        hand = self.listener.latest_hand
        if not hand: return
        r, p, y = rpy_from_palm(hand.palm.direction, hand.palm.normal)
        self._rpy_origin = (r, p, y)

    # ---- getters: pose ----
    def get_xyz(self) -> Optional[Tuple[float, float, float]]:
        hand = self.listener.latest_hand
        if not hand: return None
        x, y, z = v3(hand.palm.position)
        ox, oy, oz = self._pos_origin
        return (x - ox, y - oy, z - oz)  # mm

    def get_rpy(self) -> Optional[Tuple[float, float, float]]:
        hand = self.listener.latest_hand
        if not hand: return None
        r, p, y = rpy_from_palm(hand.palm.direction, hand.palm.normal)
        ro, po, yo = self._rpy_origin
        return ( (r - ro)*RAD2DEG, (p - po)*RAD2DEG, (y - yo)*RAD2DEG )

    # ---- helpers for fingers ----
    def _find_digit(self, name_lc: str):
        """Find a digit by name: 'thumb','index','middle','ring','pinky'."""
        hand = self.listener.latest_hand
        if not hand: return None
        for d in hand.digits:
            if digit_name(d) == name_lc:
                return d
        return None

    def _palm_axes(self, hand):
        """Return forward, normal, right as unit vectors."""
        fwd = normalize(v3(hand.palm.direction))
        nrm = normalize(v3(hand.palm.normal))
        # Form a right-handed basis on the palm
        right = normalize(cross(fwd, nrm))
        # Re-orthogonalize forward in the plane if needed
        fwd = normalize(sub(fwd, tuple(dot(fwd, nrm)*c for c in nrm)))
        return fwd, nrm, right

    def _finger_curl_angles_deg(self, name_lc: str) -> Optional[Dict[str, float]]:
        """
        Returns dict with 'MCP','PIP','DIP' for non-thumb.
        For thumb: 'MCP','IP' (DIP absent).
        """
        digit = self._find_digit(name_lc)
        if digit is None:
            return None

        # Available bones can vary. Guard with hasattr.
        mcp = pip = dip = None

        has_meta = hasattr(digit, "metacarpal")
        has_prox = hasattr(digit, "proximal")
        has_inter = hasattr(digit, "intermediate")
        has_dist = hasattr(digit, "distal")

        if name_lc != "thumb":
            if has_meta and has_prox:
                mcp = joint_inner_angle_deg(digit.metacarpal, digit.proximal)
            if has_prox and has_inter:
                pip = joint_inner_angle_deg(digit.proximal, digit.intermediate)
            if has_inter and has_dist:
                dip = joint_inner_angle_deg(digit.intermediate, digit.distal)
            return {"MCP": mcp, "PIP": pip, "DIP": dip}
        else:
            # Thumb: treat like MCP (metacarpal↔proximal), IP (proximal↔distal or proximal↔intermediate→distal)
            if has_meta and has_prox:
                mcp = joint_inner_angle_deg(digit.metacarpal, digit.proximal)
            # Some models have 'intermediate' for thumb; IP is at proximal↔distal (or ↔intermediate if present)
            if has_prox and has_dist:
                ip = joint_inner_angle_deg(digit.proximal, digit.distal)
            elif has_prox and has_inter:
                ip = joint_inner_angle_deg(digit.proximal, digit.intermediate)
            else:
                ip = None
            return {"MCP": mcp, "IP": ip}

    def _finger_abduction_signed_deg(self, name_lc: str) -> Optional[float]:
        """
        Signed abduction:
          - Project finger proximal bone dir into palm plane.
          - Angle from palm forward (in-plane). Positive if toward palm-right, negative if toward palm-left.
          - For thumb, this returns in-plane spread (use thumb_cmc_out_of_plane_deg for lift/opposition).
        """
        hand = self.listener.latest_hand
        if not hand: return None
        digit = self._find_digit(name_lc)
        if digit is None: return None

        fwd, nrm, right = self._palm_axes(hand)

        # use proximal bone (or metacarpal if proximal missing)
        if hasattr(digit, "proximal"):
            fdir = bone_dir(digit.proximal)
        elif hasattr(digit, "metacarpal"):
            fdir = bone_dir(digit.metacarpal)
        else:
            return None

        # project into palm plane
        fdir_plane = sub(fdir, tuple(dot(fdir, nrm)*c for c in nrm))
        if norm(fdir_plane) < 1e-6:
            return 0.0
        # unsigned angle in plane
        ang = abs(signed_angle_in_plane(fwd, fdir_plane, nrm)) * RAD2DEG
        # sign by which side of palm-right the finger projects
        sgn = 1.0 if dot(fdir_plane, right) >= 0.0 else -1.0
        return sgn * ang

    def thumb_cmc_out_of_plane_deg(self) -> Optional[float]:
        """
        Thumb 'abduction/opposition' relative to palm plane (out-of-plane).
        Positive when thumb lifts out of palm (dot(dir, palm_normal) > 0).
        """
        hand = self.listener.latest_hand
        if not hand: return None
        digit = self._find_digit("thumb")
        if digit is None: return None
        nrm = normalize(v3(hand.palm.normal))
        # use thumb metacarpal direction for CMC behavior
        if hasattr(digit, "metacarpal"):
            tdir = bone_dir(digit.metacarpal)
        elif hasattr(digit, "proximal"):
            tdir = bone_dir(digit.proximal)
        else:
            return None
        s = clamp(dot(tdir, nrm), -1.0, 1.0)  # sin of angle out of plane (approx)
        # Use asin for small angles; map [-1,1] to [-90,90]
        return math.degrees(math.asin(s))

    # ---- public getters for each finger ----
    def get_finger_curls(self, finger: str) -> Optional[Dict[str, float]]:
        """finger in {'thumb','index','middle','ring','pinky'}"""
        return self._finger_curl_angles_deg(finger.lower())

    def get_finger_abduction(self, finger: str) -> Optional[float]:
        """Signed abduction in degrees (right=+, left=-)."""
        return self._finger_abduction_signed_deg(finger.lower())

    def get_all_joint_angles(self) -> Optional[Dict[str, Dict[str, float]]]:
        """Convenience: returns curls/abductions for all digits."""
        if not self.hand_present():
            return None
        out = {}
        for name in ["thumb", "index", "middle", "ring", "pinky"]:
            curls = self.get_finger_curls(name)
            abd = self.get_finger_abduction(name)
            d = {}
            if curls: d.update(curls)
            if abd is not None: d["Abduction"] = abd
            if name == "thumb":
                opp = self.thumb_cmc_out_of_plane_deg()
                if opp is not None:
                    d["CMC_OutOfPlane"] = opp
            out[name] = d
        return out

# ---- Example usage ----
if __name__ == "__main__":
    tracker = LeapHandTracker(hand="right")  # change to "left" if you want
    with tracker.conn.open():
        tracker.conn.set_tracking_mode(leap.TrackingMode.Desktop)
        print("Running... (Ctrl+C to quit)")
        try:
            # Optional: set origins after you place your hand in a 'neutral' pose
            # time.sleep(1.0)
            # tracker.set_xyz_origin()
            # tracker.set_rpy_origin()

            while True:
                if tracker.hand_present():
                    xyz = tracker.get_xyz()
                    rpy = tracker.get_rpy()
                    if xyz:
                        print(f"pos(mm): ({xyz[0]:.1f},{xyz[1]:.1f},{xyz[2]:.1f})", end="  ")
                    if rpy:
                        print(f"RPY(deg): ({rpy[0]:.1f},{rpy[1]:.1f},{rpy[2]:.1f})", end="  ")

                    curls = tracker.get_all_joint_angles()
                    if curls:
                        # Example: print compact per finger
                        parts = []
                        for name in ["thumb","index","middle","ring","pinky"]:
                            if name in curls:
                                parts.append(f"{name[:3]}:{ {k: round(v,1) for k,v in curls[name].items() if v is not None} }")
                        print(" | " + "  ".join(parts))
                    else:
                        print()
                else:
                    print("No selected hand visible.")
                time.sleep(0.02)
        except KeyboardInterrupt:
            pass
