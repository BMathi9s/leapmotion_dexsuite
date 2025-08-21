# Python 3.x — Ultraleap Gemini (LeapC) bindings
# Prints: per-hand position (x,y,z), roll/pitch/yaw, plus per-finger bones.
# Uses the same listener/connection style as the official examples.

import math
import time
import leap

RAD2DEG = 180.0 / math.pi

def quat_to_euler_xyz(qw, qx, qy, qz):
    """Quaternion (w,x,y,z) -> roll(X), pitch(Y), yaw(Z) in radians."""
    # roll (x-axis)
    t0 = +2.0 * (qw * qx + qy * qz)
    t1 = +1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(t0, t1)
    # pitch (y-axis)
    t2 = +2.0 * (qw * qy - qz * qx)
    t2 = +1.0 if t2 > +1.0 else (-1.0 if t2 < -1.0 else t2)
    pitch = math.asin(t2)
    # yaw (z-axis)
    t3 = +2.0 * (qw * qz + qx * qy)
    t4 = +1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(t3, t4)
    return roll, pitch, yaw

def safe_attr(obj, name, default=None):
    return getattr(obj, name, default)

def vec_tuple(v):
    # supports LeapC vectors which behave like sequences or have x,y,z attrs
    if v is None:
        return (0.0, 0.0, 0.0)
    if hasattr(v, "x"):
        return (float(v.x), float(v.y), float(v.z))
    return (float(v[0]), float(v[1]), float(v[2]))

def print_bone(label, bone):
    pj = vec_tuple(safe_attr(bone, "prev_joint"))
    nj = vec_tuple(safe_attr(bone, "next_joint"))
    print(f"    {label:<12} start:{tuple(round(x,1) for x in pj)}  end:{tuple(round(x,1) for x in nj)}")

class PoseListener(leap.Listener):
    def on_connection_event(self, event):
        print("Connected to Ultraleap service.")

    def on_device_event(self, event):
        try:
            with event.device.open():
                info = event.device.get_info()
        except leap.LeapCannotOpenDeviceError:
            info = event.device.get_info()
        print(f"Found device {info.serial}")

    def on_tracking_event(self, event):
        # Match the cadence of the official examples
        print(f"\nFrame {event.tracking_frame_id} with {len(event.hands)} hands.")

        for hand in event.hands:
            # Hand label
            hand_type = "Left" if str(hand.type) == "HandType.Left" else "Right"
            hid = safe_attr(hand, "id", 0)

            # Position (mm)
            palm = hand.palm
            pos = vec_tuple(safe_attr(palm, "position"))

            # Orientation: prefer quaternion; else fall back to legacy basis
            rpy_deg = (0.0, 0.0, 0.0)
            quat = safe_attr(palm, "orientation") or safe_attr(palm, "quaternion")
            if quat:
                # orientation/quaternion is typically (w, x, y, z)
                w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
                r, p, y_ = quat_to_euler_xyz(w, x, y, z)
                rpy_deg = (r * RAD2DEG, p * RAD2DEG, y_ * RAD2DEG)
            else:
                # Fallback using palm basis if available
                direction = vec_tuple(safe_attr(palm, "direction"))
                normal    = vec_tuple(safe_attr(palm, "normal"))
                # These formulas match legacy-style outputs
                pitch = math.atan2(direction[1], direction[2])
                roll  = math.atan2(normal[0],    normal[1])
                yaw   = math.atan2(direction[0], direction[2])
                rpy_deg = (roll * RAD2DEG, pitch * RAD2DEG, yaw * RAD2DEG)

            print(f"{hand_type} hand id:{hid}  pos(mm):({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f})  "
                  f"RPY(deg):({rpy_deg[0]:.1f},{rpy_deg[1]:.1f},{rpy_deg[2]:.1f})")

            # Optional: arm info if present
            arm = safe_attr(hand, "arm")
            if arm:
                wrist = vec_tuple(safe_attr(arm, "wrist_position"))
                elbow = vec_tuple(safe_attr(arm, "elbow_position"))
                print(f"  Arm  wrist:{tuple(round(x,1) for x in wrist)}  elbow:{tuple(round(x,1) for x in elbow)}")

            # Fingers & bones — mirror the style of the examples
            for digit in hand.digits:
                # digit has bones: metacarpal, proximal, intermediate, distal
                # (names align with the examples)
                # Some devices report no metacarpal for the thumb; guard with getattr.
                print(f"  {str(digit.type)[9:]}:")  # 'DigitType.Index' -> 'Index'
                if hasattr(digit, "metacarpal"):
                    print_bone("Metacarpal", digit.metacarpal)
                if hasattr(digit, "proximal"):
                    print_bone("Proximal", digit.proximal)
                if hasattr(digit, "intermediate"):
                    print_bone("Intermediate", digit.intermediate)
                if hasattr(digit, "distal"):
                    print_bone("Distal", digit.distal)

def main():
    listener = PoseListener()
    conn = leap.Connection()
    conn.add_listener(listener)

    # Open and stream (same pattern as the official examples)
    running = True
    with conn.open():
        # Desktop mode like the sample
        conn.set_tracking_mode(leap.TrackingMode.Desktop)
        try:
            while running:
                time.sleep(0.02)  # gentle throttle
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    main()
