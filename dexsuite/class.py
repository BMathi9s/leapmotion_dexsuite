# Requires: pip install leapc-python-bindings (per Ultraleap instructions)
# Python 3.10, Ubuntu 22.04

import math
import time
from typing import Optional, Tuple, Literal

import leap
from leap import datatypes as ldt

RAD2DEG = 180.0 / math.pi
HandSide = Literal["left", "right"]


def _vec3(v) -> Tuple[float, float, float]:
    # Works with leap.Vector and tuples
    if hasattr(v, "x"):
        return float(v.x), float(v.y), float(v.z)
    return float(v[0]), float(v[1]), float(v[2])


def _norm(v):
    x, y, z = _vec3(v)
    n = math.sqrt(x * x + y * y + z * z) or 1e-9
    return (x / n, y / n, z / n)


def _dot(a, b) -> float:
    ax, ay, az = _vec3(a)
    bx, by, bz = _vec3(b)
    return ax * bx + ay * by + az * bz


def _cross(a, b):
    ax, ay, az = _vec3(a)
    bx, by, bz = _vec3(b)
    return (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)


def _project_onto_plane(v, plane_n):
    v = _vec3(v)
    n = _norm(plane_n)
    # v_proj = v - (v·n) n
    d = _dot(v, n)
    return (v[0] - d * n[0], v[1] - d * n[1], v[2] - d * n[2])


def _signed_angle(u, v, axis) -> float:
    """Signed angle from u to v about 'axis' (right-hand rule). Returns radians."""
    u_n = _norm(u)
    v_n = _norm(v)
    cross_uv = _cross(u_n, v_n)
    s = _dot(axis, cross_uv)  # signed magnitude around axis
    c = _dot(u_n, v_n)
    return math.atan2(s, max(-1.0, min(1.0, c)))


def _rpy_from_palm(direction, normal) -> Tuple[float, float, float]:
    """Roll, Pitch, Yaw in radians following the sample you shared (converted to a helper)."""
    dx, dy, dz = _vec3(direction)
    nx, ny, _ = _vec3(normal)
    pitch = math.atan2(dy, math.sqrt(dx * dx + dz * dz))
    yaw = math.atan2(dx, -dz)
    roll = math.atan2(nx, -ny)
    return roll, pitch, yaw


class LeapHandTracker(leap.Listener):
    """
    High-level Ultraleap hand tracker focused on single-hand control.

    - Choose 'left' or 'right' hand in __init__ (default 'right').
    - Get calibrated XYZ / RPY.
    - Get per-finger MCP/PIP/DIP (flexion/curl) and abduction angles.
    - Get thumb CMC/MCP/IP flexion and thumb abduction.

    Conventions:
    - Angles are in degrees.
    - Flexion (curl) positive when bending toward the palm.
    - Abduction on palm plane: 0° when finger points forward (palm direction),
      positive when rotated toward the hand's right side, negative toward left.
    """

    def __init__(
        self,
        hand: HandSide = "right",
        tracking_mode: leap.TrackingMode = leap.TrackingMode.Desktop,
    ):
        super().__init__()
        self._desired_hand = hand.lower()
        assert self._desired_hand in ("left", "right")

        self._conn = leap.Connection()
        self._conn.add_listener(self)

        self._open_ctx = None
        self._latest_event: Optional[ldt.TrackingEvent] = None

        # Calibration offsets
        self._xyz_origin = (0.0, 0.0, 0.0)
        self._rpy_origin = (0.0, 0.0, 0.0)  # degrees

        self._tracking_mode = tracking_mode

    # ---------- Lifecycle ----------
    def start(self):
        self._open_ctx = self._conn.open()
        self._open_ctx.__enter__()
        self._conn.set_tracking_mode(self._tracking_mode)
        # Wait briefly for tracking
        t0 = time.time()
        while self._latest_event is None and time.time() - t0 < 3.0:
            time.sleep(0.01)

    def stop(self):
        if self._open_ctx:
            try:
                self._open_ctx.__exit__(None, None, None)
            finally:
                self._open_ctx = None

    # ---------- Leap callbacks ----------
    def on_connection_event(self, event):
        # Connected
        pass

    def on_tracking_mode_event(self, event):
        # Track mode changed
        pass

    def on_tracking_event(self, event: ldt.TrackingEvent):
        self._latest_event = event  # keep latest

    # ---------- Helpers ----------
    def _pick_hand(self) -> Optional[ldt.Hand]:
        evt = self._latest_event
        if not evt or len(evt.hands) == 0:
            return None
        # Pick the requested side; if not present, None
        for h in evt.hands:
            side = "left" if str(h.type) == "HandType.Left" else "right"
            if side == self._desired_hand:
                return h
        return None

    @staticmethod
    def _finger_by_name(hand: ldt.Hand, name: str) -> ldt.Digit:
        """
        name ∈ {'thumb','index','middle','ring','pinky'}
        Leap order is 0..4 = thumb,index,middle,ring,pinky
        """
        idx = {"thumb": 0, "index": 1, "middle": 2, "ring": 3, "pinky": 4}[name]
        return hand.digits[idx]

    # ---------- Calibration ----------
    def set_origin_xyz(self):
        """Zero the XYZ at the current palm position."""
        hand = self._pick_hand()
        if not hand:
            return
        self._xyz_origin = _vec3(hand.palm.position)

    def set_origin_rpy(self):
        """Zero the RPY (deg) at the current palm orientation."""
        hand = self._pick_hand()
        if not hand or not hasattr(hand.palm, "direction"):
            return
        r, p, y = _rpy_from_palm(hand.palm.direction, hand.palm.normal)
        self._rpy_origin = (r * RAD2DEG, p * RAD2DEG, y * RAD2DEG)

    # ---------- XYZ / RPY getters ----------
    def get_xyz(self) -> Optional[Tuple[float, float, float]]:
        """Palm position in mm, origin-corrected."""
        hand = self._pick_hand()
        if not hand:
            return None
        x, y, z = _vec3(hand.palm.position)
        ox, oy, oz = self._xyz_origin
        return (x - ox, y - oy, z - oz)

    def get_rpy(self) -> Optional[Tuple[float, float, float]]:
        """Palm roll/pitch/yaw in degrees, origin-corrected."""
        hand = self._pick_hand()
        if not hand or not hasattr(hand.palm, "direction"):
            return None
        r, p, y = _rpy_from_palm(hand.palm.direction, hand.palm.normal)  # radians
        rpy = (r * RAD2DEG, p * RAD2DEG, y * RAD2DEG)
        return (rpy[0] - self._rpy_origin[0], rpy[1] - self._rpy_origin[1], rpy[2] - self._rpy_origin[2])

    # ---------- Joint angle math ----------
    @staticmethod
    def _bone_vec(bone: ldt.Bone):
        return (
            bone.next_joint.x - bone.prev_joint.x,
            bone.next_joint.y - bone.prev_joint.y,
            bone.next_joint.z - bone.prev_joint.z,
        )

    @staticmethod
    def _flexion_signed(prev_bone: ldt.Bone, next_bone: ldt.Bone, palm_normal) -> float:
        """
        Inner flexion angle with sign toward the palm.
        - Unsigned inner angle = acos( u·v )
        - Sign = sign( palm_normal · (u × v) ) so that bending toward palm is +.
        Returns degrees.
        """
        u = _norm(LeapHandTracker._bone_vec(prev_bone))
        v = _norm(LeapHandTracker._bone_vec(next_bone))
        unsigned = math.acos(max(-1.0, min(1.0, _dot(u, v))))
        s = math.copysign(1.0, _dot(palm_normal, _cross(u, v)))
        return float(unsigned * s * RAD2DEG)

    @staticmethod
    def _abduction_signed(finger_vec, palm_direction, palm_normal) -> float:
        """
        Signed abduction on the palm plane.
        - Project finger_vec onto palm plane.
        - Measure signed angle from palm_direction → projected finger,
          around +palm_normal (right-hand rule).
        - Positive means rotated toward the hand's +right = palm_direction × palm_normal.
        Returns degrees.
        """
        proj = _project_onto_plane(finger_vec, palm_normal)
        if _dot(proj, proj) < 1e-8:
            return 0.0
        ang = _signed_angle(palm_direction, proj, palm_normal)
        return float(ang * RAD2DEG)

    # ---------- Generic getters (MCP/PIP/DIP & abduction) ----------
    def get_mcp_angle(self, finger: Literal["index", "middle", "ring", "pinky"]) -> Optional[float]:
        hand = self._pick_hand()
        if not hand:
            return None
        d = self._finger_by_name(hand, finger)
        # Metacarpal->Proximal gives the MCP joint flexion with the next bone
        return self._flexion_signed(d.metacarpal, d.proximal, hand.palm.normal)

    def get_pip_angle(self, finger: Literal["index", "middle", "ring", "pinky"]) -> Optional[float]:
        hand = self._pick_hand()
        if not hand:
            return None
        d = self._finger_by_name(hand, finger)
        return self._flexion_signed(d.proximal, d.intermediate, hand.palm.normal)

    def get_dip_angle(self, finger: Literal["index", "middle", "ring", "pinky"]) -> Optional[float]:
        hand = self._pick_hand()
        if not hand:
            return None
        d = self._finger_by_name(hand, finger)
        return self._flexion_signed(d.intermediate, d.distal, hand.palm.normal)

    def get_abduction(self, finger: Literal["index", "middle", "ring", "pinky"]) -> Optional[float]:
        """
        Abduction relative to palm forward direction (0° forward).
        + right, - left (matches your requested sign convention).
        """
        hand = self._pick_hand()
        if not hand:
            return None
        d = self._finger_by_name(hand, finger)
        # Use proximal bone direction as the finger ray at MCP
        fwd = self._bone_vec(d.proximal)
        return self._abduction_signed(fwd, hand.palm.direction, hand.palm.normal)

    # ---------- Thumb getters (CMC/MCP/IP flexion + abduction) ----------
    def get_thumb_cmc(self) -> Optional[float]:
        """CMC flexion (metacarpal vs proximal)."""
        hand = self._pick_hand()
        if not hand:
            return None
        t = self._finger_by_name(hand, "thumb")
        return self._flexion_signed(t.metacarpal, t.proximal, hand.palm.normal)

    def get_thumb_mcp(self) -> Optional[float]:
        """MCP flexion (proximal vs intermediate)."""
        hand = self._pick_hand()
        if not hand:
            return None
        t = self._finger_by_name(hand, "thumb")
        return self._flexion_signed(t.proximal, t.intermediate, hand.palm.normal)

    def get_thumb_ip(self) -> Optional[float]:
        """IP flexion (intermediate vs distal)."""
        hand = self._pick_hand()
        if not hand:
            return None
        t = self._finger_by_name(hand, "thumb")
        return self._flexion_signed(t.intermediate, t.distal, hand.palm.normal)

    def get_thumb_abduction(self) -> Optional[float]:
        """
        Thumb abduction: angle of the thumb metacarpal on the palm plane.
        Positive toward the hand's right.
        """
        hand = self._pick_hand()
        if not hand:
            return None
        t = self._finger_by_name(hand, "thumb")
        mc_vec = self._bone_vec(t.metacarpal)
        return self._abduction_signed(mc_vec, hand.palm.direction, hand.palm.normal)


# ---------- Minimal usage example ----------
if __name__ == "__main__":
    tracker = LeapHandTracker(hand="right", tracking_mode=leap.TrackingMode.Desktop)
    tracker.start()
    print("Press Ctrl+C to quit. Calibrating origins in 2 seconds...")
    time.sleep(2.0)
    tracker.set_origin_xyz()
    tracker.set_origin_rpy()

    try:
        while True:
            xyz = tracker.get_xyz()
            rpy = tracker.get_rpy()
            if xyz and rpy:
                x, y, z = xyz
                r, p, yv = rpy
                # Basic readout
                print(
                    f"XYZ(mm)=({x:6.1f},{y:6.1f},{z:6.1f})  "
                    f"RPY(deg)=({r:6.1f},{p:6.1f},{yv:6.1f})  "
                    f"Idx MCP/PIP/DIP=({tracker.get_mcp_angle('index') or 0:5.1f},"
                    f"{tracker.get_pip_angle('index') or 0:5.1f},"
                    f"{tracker.get_dip_angle('index') or 0:5.1f})  "
                    f"Idx Abd={tracker.get_abduction('index') or 0:5.1f}  "
                    f"Thumb CMC/MCP/IP=({tracker.get_thumb_cmc() or 0:5.1f},"
                    f"{tracker.get_thumb_mcp() or 0:5.1f},"
                    f"{tracker.get_thumb_ip() or 0:5.1f})  "
                    f"Thumb Abd={tracker.get_thumb_abduction() or 0:5.1f}"
                )
            time.sleep(0.02)  # ~50 Hz loop; tune as needed
    except KeyboardInterrupt:
        pass
    finally:
        tracker.stop()
