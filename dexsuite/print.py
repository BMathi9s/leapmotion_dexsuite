import math
import time
import leap

RAD2DEG = 180.0 / math.pi

def vec3(v):
    # Works with leap vectors (attributes) or sequences
    if hasattr(v, "x"):
        return (float(v.x), float(v.y), float(v.z))
    return (float(v[0]), float(v[1]), float(v[2]))

def roll_pitch_yaw_from_palm(direction, normal):
    # direction: unit vector from palm → fingers
    # normal: palm normal (points “out” of the palm)
    dx, dy, dz = vec3(direction)
    nx, ny, _  = vec3(normal)
    # Canonical Leap-style angles (match legacy behavior):
    #  - pitch from direction vs horizontal plane
    #  - yaw from direction projected toward camera axes
    #  - roll from palm normal vs horizontal
    pitch = math.atan2(dy, math.sqrt(dx*dx + dz*dz))
    yaw   = math.atan2(dx, -dz)
    roll  = math.atan2(nx, -ny)
    return roll, pitch, yaw

def fmt_xyz(v): 
    x, y, z = vec3(v)
    return f"({x:.1f},{y:.1f},{z:.1f})"

def print_bone(label, bone):
    try:
        pj = bone.prev_joint
        nj = bone.next_joint
        print(f"    {label:<12} start:{fmt_xyz(pj)}  end:{fmt_xyz(nj)}")
    except Exception:
        pass

class PoseListener(leap.Listener):
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
        print(f"\nFrame {event.tracking_frame_id} with {len(event.hands)} hands.")
        for hand in event.hands:
            hand_type = "Left" if str(hand.type) == "HandType.Left" else "Right"
            palm      = hand.palm

            # Position (mm)
            pos = palm.position
            # Orientation from basis vectors (no quaternion needed)
            direction = getattr(palm, "direction", None)
            normal    = getattr(palm, "normal", None)

            if direction is not None and normal is not None:
                r, p, y = roll_pitch_yaw_from_palm(direction, normal)
                rpy_deg = (r*RAD2DEG, p*RAD2DEG, y*RAD2DEG)
                rpy_txt = f"RPY(deg):({rpy_deg[0]:.1f},{rpy_deg[1]:.1f},{rpy_deg[2]:.1f})"
            else:
                rpy_txt = "RPY: N/A (no direction/normal on palm)"

            print(f"{hand_type} hand id:{hand.id}  pos(mm):{fmt_xyz(pos)}  {rpy_txt}")

            # Optional arm info (if present)
            arm = getattr(hand, "arm", None)
            if arm:
                print(f"  Arm  wrist:{fmt_xyz(arm.wrist_position)}  elbow:{fmt_xyz(arm.elbow_position)}")

            # Fingers & bones (Metacarpal → Distal)
            for digit in hand.digits:
                # Using the same pattern as the examples (don’t rely on .type names)
                print("  Digit:")
                if hasattr(digit, "metacarpal"):   print_bone("Metacarpal",   digit.metacarpal)
                if hasattr(digit, "proximal"):     print_bone("Proximal",     digit.proximal)
                if hasattr(digit, "intermediate"): print_bone("Intermediate", digit.intermediate)
                if hasattr(digit, "distal"):       print_bone("Distal",       digit.distal)

def main():
    listener = PoseListener()
    conn = leap.Connection()
    conn.add_listener(listener)

    with conn.open():
        # Desktop mode (same as the official examples)
        conn.set_tracking_mode(leap.TrackingMode.Desktop)
        try:
            while True:
                time.sleep(0.02)
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    main()
