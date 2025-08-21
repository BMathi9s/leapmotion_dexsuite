#!/usr/bin/env python3
"""
Combined Ultraleap Gemini visualiser + verbose console printout.

- Shows the skeleton using the user's visualiser.Canvas
- Prints per-hand pose: x,y,z (mm) and roll/pitch/yaw (deg)
- Prints per-finger MCP/PIP/DIP flexion (deg), MCP ab/adduction (deg), and normalized curl (-1..1)
- Robust to arm being presented as an Arm struct or Bone (prev_joint/elbow, next_joint/wrist)

Keys in the visualiser window:
  x: Exit
  h: HMD tracking mode
  s: ScreenTop tracking mode
  d: Desktop tracking mode
  f: Toggle skeleton/dots
"""

import math
import time
import cv2
import leap

# import the user's visualiser module (must be in the same folder or on PYTHONPATH)
import visualiser

RAD2DEG = 180.0 / math.pi

# ---------------------- vector helpers ----------------------
def v_tuple(v):
    if hasattr(v, "x"):
        return (float(v.x), float(v.y), float(v.z))
    # assume sequence
    return (float(v[0]), float(v[1]), float(v[2]))

def v_sub(a, b):
    ax, ay, az = v_tuple(a); bx, by, bz = v_tuple(b)
    return (ax-bx, ay-by, az-bz)

def v_len(v):
    x, y, z = v_tuple(v)
    return math.sqrt(x*x + y*y + z*z)

def v_unit(v):
    x, y, z = v_tuple(v)
    n = math.sqrt(x*x + y*y + z*z)
    if n < 1e-8:
        return (0.0, 0.0, 0.0)
    return (x/n, y/n, z/n)

def v_dot(a, b):
    ax, ay, az = v_tuple(a); bx, by, bz = v_tuple(b)
    return ax*bx + ay*by + az*bz

def v_cross(a, b):
    ax, ay, az = v_tuple(a); bx, by, bz = v_tuple(b)
    return (ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bx)

def angle_between(u, v):
    uu, vv = v_unit(u), v_unit(v)
    d = max(-1.0, min(1.0, v_dot(uu, vv)))
    return math.acos(d)

def project_onto_plane(v, n):
    # v - (v·n) n
    n_u = v_unit(n)
    dot = v_dot(v, n_u)
    px, py, pz = v_tuple(v)
    nx, ny, nz = n_u
    out = (px - dot*nx, py - dot*ny, pz - dot*nz)
    ln = v_len(out)
    if ln < 1e-8:
        return (0.0, 0.0, 0.0)
    return (out[0]/ln, out[1]/ln, out[2]/ln)

def signed_angle_in_plane(u, v, plane_n):
    ang = angle_between(u, v)
    # sign from plane normal · (u × v)
    cx, cy, cz = v_cross(u, v)
    nx, ny, nz = v_unit(plane_n)
    s = nx*cx + ny*cy + nz*cz
    return ang if s >= 0 else -ang

def roll_pitch_yaw_from_palm(direction, normal):
    dx, dy, dz = v_tuple(direction)
    nx, ny, _  = v_tuple(normal)
    # Leap-style convention; matches legacy sample feel
    pitch = math.atan2(dy, math.sqrt(dx*dx + dz*dz))
    yaw   = math.atan2(dx, -dz)
    roll  = math.atan2(nx, -ny)
    return roll, pitch, yaw

# ------------------- angle computation per digit -------------------
def bone_dir(bone):
    """Unit direction from prev_joint -> next_joint."""
    if not hasattr(bone, "prev_joint") or not hasattr(bone, "next_joint"):
        return (0.0, 0.0, 0.0)
    return v_unit(v_sub(bone.next_joint, bone.prev_joint))

def get_bone_by_name(digit, name):
    # Prefer named attributes if available; else fall back to bones[0..3]
    if hasattr(digit, name):
        return getattr(digit, name)
    if hasattr(digit, "bones"):
        idx_map = {"metacarpal": 0, "proximal": 1, "intermediate": 2, "distal": 3}
        if name in idx_map and len(digit.bones) > idx_map[name]:
            return digit.bones[idx_map[name]]
    return None

def compute_digit_angles(digit, palm_normal, palm_direction):
    meta = get_bone_by_name(digit, "metacarpal")
    prox = get_bone_by_name(digit, "proximal")
    inter= get_bone_by_name(digit, "intermediate")
    dist = get_bone_by_name(digit, "distal")

    # Flexion (angles between adjacent bone directions)
    mcp = angle_between(bone_dir(meta), bone_dir(prox)) if (meta and prox) else float("nan")
    pip = angle_between(bone_dir(prox), bone_dir(inter)) if (prox and inter) else float("nan")
    dip = angle_between(bone_dir(inter), bone_dir(dist)) if (inter and dist) else float("nan")

    # Abduction at MCP (project into palm plane). Prefer meta vs prox; if meta missing (thumb), use lateral reference.
    abd = float("nan")
    if prox:
        if meta:
            u = project_onto_plane(bone_dir(meta), palm_normal)
            v = project_onto_plane(bone_dir(prox), palm_normal)
            abd = signed_angle_in_plane(u, v, palm_normal)
        else:
            # fallback: compare proximal dir to lateral axis (normal × direction)
            lateral = v_unit(v_cross(palm_normal, palm_direction))
            vproj = project_onto_plane(bone_dir(prox), palm_normal)
            abd = signed_angle_in_plane(lateral, vproj, palm_normal)

    # Normalized overall curl in [-1, 1] (open=-1, closed=+1)
    flex_sum = 0.0
    count = 0
    for a in (mcp, pip, dip):
        if not math.isnan(a):
            flex_sum += a
            count += 1
    if count > 0:
        curl = (flex_sum / (math.pi * count)) * 2.0 - 1.0
    else:
        curl = float("nan")

    return mcp, pip, dip, abd, curl

def deg(x):
    return float("nan") if (x is None or isinstance(x, float) and math.isnan(x)) else (x * RAD2DEG)

# ---------------------- listener ----------------------
DIGIT_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

class VisualisingPrintingListener(leap.Listener):
    def __init__(self, canvas: visualiser.Canvas):
        self.canvas = canvas

    def on_connection_event(self, event):
        print("Connected to Ultraleap service.")

    def on_tracking_mode_event(self, event):
        self.canvas.set_tracking_mode(event.current_tracking_mode)
        modes = {
            leap.TrackingMode.Desktop: "Desktop",
            leap.TrackingMode.HMD: "HMD",
            leap.TrackingMode.ScreenTop: "ScreenTop",
        }
        print(f"Tracking mode changed to {modes.get(event.current_tracking_mode, str(event.current_tracking_mode))}")

    def on_device_event(self, event):
        try:
            with event.device.open():
                info = event.device.get_info()
        except leap.LeapCannotOpenDeviceError:
            info = event.device.get_info()
        print(f"Found device {info.serial}")

    def on_tracking_event(self, event):
        # 1) Draw the hands into the canvas image
        self.canvas.render_hands(event)

        # 2) Print a full pose + joint angle dump
        print(f"\nFrame {event.tracking_frame_id} with {len(event.hands)} hands.")
        for h_i, hand in enumerate(event.hands):
            hand_type = "Left" if str(hand.type) == "HandType.Left" else "Right"
            palm = hand.palm
            pos = v_tuple(palm.position)
            if hasattr(palm, "direction") and hasattr(palm, "normal"):
                r, p, y = roll_pitch_yaw_from_palm(palm.direction, palm.normal)
                rpy_deg = (r*RAD2DEG, p*RAD2DEG, y*RAD2DEG)
                rpy_txt = f"RPY(deg):({rpy_deg[0]:.1f},{rpy_deg[1]:.1f},{rpy_deg[2]:.1f})"
            else:
                rpy_txt = "RPY: N/A"

            print(f"{hand_type} hand id:{hand.id}  pos(mm):({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f})  {rpy_txt}")

            # Arm info (supports both Arm struct and Bone)
            arm = getattr(hand, "arm", None)
            if arm:
                if hasattr(arm, "wrist_position") and hasattr(arm, "elbow_position"):
                    wrist = arm.wrist_position; elbow = arm.elbow_position
                elif hasattr(arm, "prev_joint") and hasattr(arm, "next_joint"):
                    elbow = arm.prev_joint; wrist = arm.next_joint
                else:
                    wrist = elbow = None
                if wrist and elbow:
                    wx, wy, wz = v_tuple(wrist); ex, ey, ez = v_tuple(elbow)
                    print(f"  Arm  wrist:({wx:.1f},{wy:.1f},{wz:.1f})  elbow:({ex:.1f},{ey:.1f},{ez:.1f})")

            # Per digit: MCP/PIP/DIP + abduction + curl
            palm_normal = v_tuple(palm.normal) if hasattr(palm, "normal") else (0.0, 1.0, 0.0)
            palm_direction = v_tuple(palm.direction) if hasattr(palm, "direction") else (0.0, 0.0, -1.0)
            for idx, digit in enumerate(hand.digits):
                name = DIGIT_NAMES[idx] if idx < len(DIGIT_NAMES) else f"Digit{idx}"
                mcp, pip, dip, abd, curl = compute_digit_angles(digit, palm_normal, palm_direction)
                mcpd, pipd, dipd, abdd = deg(mcp), deg(pip), deg(dip), deg(abd)
                curl_txt = "nan" if math.isnan(curl) else f"{curl:+.2f}"
                def fmt(x): return "nan" if (x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))) else f"{x:.1f}"
                print(f"  {name:<6} MCP:{fmt(mcpd)}°  PIP:{fmt(pipd)}°  DIP:{fmt(dipd)}°  Abd:{fmt(abdd)}°  Curl:{curl_txt}")

# ---------------------- main ----------------------
def main():
    canvas = visualiser.Canvas()

    print(canvas.name)
    print("")
    print("Press <key> in visualiser window to:")
    print("  x: Exit")
    print("  h: Select HMD tracking mode")
    print("  s: Select ScreenTop tracking mode")
    print("  d: Select Desktop tracking mode")
    print("  f: Toggle hands format between Skeleton/Dots")

    listener = VisualisingPrintingListener(canvas)
    connection = leap.Connection()
    connection.add_listener(listener)

    running = True
    with connection.open():
        connection.set_tracking_mode(leap.TrackingMode.Desktop)
        while running:
            # Display the current canvas image (updated by on_tracking_event)
            cv2.imshow(canvas.name, canvas.output_image)
            key = cv2.waitKey(1)
            if key == ord("x"):
                break
            elif key == ord("h"):
                connection.set_tracking_mode(leap.TrackingMode.HMD)
            elif key == ord("s"):
                connection.set_tracking_mode(leap.TrackingMode.ScreenTop)
            elif key == ord("d"):
                connection.set_tracking_mode(leap.TrackingMode.Desktop)
            elif key == ord("f"):
                canvas.toggle_hands_format()

if __name__ == "__main__":
    main()
