# Python 3.x — Ultraleap Gemini (LeapC) bindings
# Prints per-hand: position (x,y,z), roll/pitch/yaw, and key bone segments.
# Also computes RPY either from quaternion (preferred) or from palm basis fallback.

import math
import time

# The 'leap' package comes from ultraleap/leapc-python-bindings
#   pip install -e leapc-python-api
from leap import connection, tracking  # high-level helpers + typed structs if available

RAD2DEG = 180.0 / math.pi

def quat_to_euler_xyz(qw, qx, qy, qz):
    """
    Convert quaternion (w, x, y, z) -> roll(x), pitch(y), yaw(z) in radians.
    Assumes right-handed coordinates and unit quaternion.
    """
    # roll (x-axis)
    t0 = +2.0 * (qw * qx + qy * qz)
    t1 = +1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(t0, t1)

    # pitch (y-axis)
    t2 = +2.0 * (qw * qy - qz * qx)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)

    # yaw (z-axis)
    t3 = +2.0 * (qw * qz + qx * qy)
    t4 = +1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(t3, t4)
    return roll, pitch, yaw

def vec_len(v): return math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]) if v else 0.0

def angle_between(u, v):
    if not u or not v: return 0.0
    lu, lv = vec_len(u), vec_len(v)
    if lu < 1e-8 or lv < 1e-8: return 0.0
    dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) / (lu*lv)
    dot = max(-1.0, min(1.0, dot))
    return math.acos(dot)

def project_onto_plane(v, n):
    # v - (v·n) n
    dot = v[0]*n[0] + v[1]*n[1] + v[2]*n[2]
    out = (v[0]-dot*n[0], v[1]-dot*n[1], v[2]-dot*n[2])
    l = vec_len(out)
    return (out[0]/l, out[1]/l, out[2]/l) if l > 1e-8 else (0.0,0.0,0.0)

def signed_angle_in_plane(u, v, plane_n):
    # angle from u->v around plane normal
    ang = angle_between(u, v)
    # sign via cross product
    cx = u[1]*v[2] - u[2]*v[1]
    cy = u[2]*v[0] - u[0]*v[2]
    cz = u[0]*v[1] - u[1]*v[0]
    s = plane_n[0]*cx + plane_n[1]*cy + plane_n[2]*cz
    return ang if s >= 0 else -ang

def print_bone(name, b):
    # b has prev_joint, next_joint, direction, rotation (quat) in LeapC structs
    pj = getattr(b, "prev_joint", None)
    nj = getattr(b, "next_joint", None)
    d  = getattr(b, "direction", None)
    print(f"    {name:<12} start:{tuple(round(x,1) for x in pj)} end:{tuple(round(x,1) for x in nj)} dir:{tuple(round(x,3) for x in d)}")

def main():
    # Open a LeapC connection (helpers mirror LeapPollConnection loop internally)
    with connection.Connection() as conn:
        print("Connected to Ultraleap service. Press Ctrl+C to quit.")
        while True:
            msg = conn.wait_for_frame(timeout=200)  # ms; returns a TrackingEvent wrapper
            if not msg or not isinstance(msg, tracking.TrackingEvent):
                continue

            te = msg  # alias
            # Each TrackingEvent contains a list of 'hands'
            for h in te.hands:
                hid   = getattr(h, "id", 0)
                left  = getattr(h, "type", 0) == tracking.HandType.LEFT
                which = "Left " if left else "Right"
                # Palm pose
                palm_pos = tuple(h.palm.position)   # mm
                # Try quaternion first (preferred in LeapC)
                quat = getattr(h.palm, "orientation", None)  # (w,x,y,z) in LeapC
                if quat:
                    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
                    roll, pitch, yaw = quat_to_euler_xyz(qw, qx, qy, qz)
                else:
                    # Fallback: derive from basis vectors like legacy API
                    palm_normal = tuple(h.palm.normal)      # like legacy: roll source
                    palm_dir    = tuple(h.palm.direction)   # like legacy: pitch/yaw source
                    # Roll ≈ atan2(n.x, n.y) style depends on axis conventions.
                    # A commonly used legacy mapping:
                    pitch = math.atan2(palm_dir[1], palm_dir[2])   # direction.pitch
                    roll  = math.atan2(palm_normal[0], palm_normal[1])  # normal.roll
                    yaw   = math.atan2(palm_dir[0], palm_dir[2])   # direction.yaw

                print(f"\n{which}hand id:{hid}  pos(mm):({palm_pos[0]:.1f},{palm_pos[1]:.1f},{palm_pos[2]:.1f})"
                      f"  RPY(deg):({roll*RAD2DEG:.1f},{pitch*RAD2DEG:.1f},{yaw*RAD2DEG:.1f})")

                # Arm / wrist / elbow if present
                arm = getattr(h, "arm", None)
                if arm:
                    wp = tuple(getattr(arm, "wrist_position", (0,0,0)))
                    ep = tuple(getattr(arm, "elbow_position", (0,0,0)))
                    print(f"  Arm  wrist:{tuple(round(x,1) for x in wp)}  elbow:{tuple(round(x,1) for x in ep)}")

                # Fingers + bones (metacarpal, proximal, intermediate, distal)
                for f in h.fingers:
                    fname = f.type.name.title()  # Thumb/Index/Middle/Ring/Pinky
                    print(f"  {fname}:")
                    if getattr(f, "bones", None):
                        # Iterate in anatomical order if indices are provided
                        bones = {b.type: b for b in f.bones}
                        # Names here mirror legacy sample.py
                        order = [
                            (tracking.BoneType.METACARPAL,   "Metacarpal"),
                            (tracking.BoneType.PROXIMAL,     "Proximal"),
                            (tracking.BoneType.INTERMEDIATE, "Intermediate"),
                            (tracking.BoneType.DISTAL,       "Distal"),
                        ]
                        for t, nm in order:
                            if t in bones:
                                print_bone(nm, bones[t])

            # simple throttle so we don’t spam the console too hard
            time.sleep(0.02)

if __name__ == "__main__":
    main()
