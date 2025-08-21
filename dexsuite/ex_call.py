#!/usr/bin/env python3
import time
from typing import Optional

# your class file should be importable; if it's in the same file, ignore this import.
# from ultraleap_hand_kinematics import UltraleapHandKinematics
from leap_tracking import UltraleapHandKinematics  # <-- if running in the same file as the class

def fmt_angle(x: Optional[float]) -> str:
    return f"{x:6.1f}" if isinstance(x, (int, float)) else "  None"

def print_one_snapshot(k: UltraleapHandKinematics):
    """
    Prints one line of all joint angles for: thumb (mcp,pip,dip,ip), and
    index/middle/ring/pinky (mcp,pip,dip). Also prints abduction (splay).
    """
    joints = k.get_all_joint_angles()           # {finger: {mcp,pip,dip,(ip for thumb)}}
    splay  = k.get_all_abductions()             # {finger: deg}

    # Header
    print("\n=== Joint Angles (deg) ===")
    print(" finger     mcp     pip     dip     ip*   |  abduction")
    print(" -------------------------------------------------------")

    order = ["thumb", "index", "middle", "ring", "pinky"]
    for f in order:
        d = joints.get(f, {})
        mcp = fmt_angle(d.get("mcp"))
        pip = fmt_angle(d.get("pip"))
        dip = fmt_angle(d.get("dip"))
        ip  = fmt_angle(d.get("ip")) if f == "thumb" else "      "
        abd = fmt_angle(splay.get(f))
        print(f" {f:<7} {mcp}  {pip}  {dip}  {ip}   |  {abd}")

    xyz = k.get_xyz()
    rpy = k.get_rpy()
    if xyz is not None and rpy is not None:
        print(f"\n palm XYZ (mm, origin-corrected): ({xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f})")
        print(f" palm RPY (deg, origin-corrected): ({rpy[0]:.1f}, {rpy[1]:.1f}, {rpy[2]:.1f})")
    else:
        print("\n palm pose not available (no hand seen)")

def stream_angles(k: UltraleapHandKinematics, hz: float = 20.0):
    """
    Streams angles at ~hz until Ctrl+C.
    Each line prints: finger:mcp/pip/dip (and thumb ip), plus abductions.
    """
    dt = 1.0 / max(1.0, hz)
    print("\nStreaming angles. Press Ctrl+C to stop.\n")
    try:
        while True:
            joints = k.get_all_joint_angles()
            splay  = k.get_all_abductions()
            parts = []
            # Compact, single-line summary
            for f in ["thumb", "index", "middle", "ring", "pinky"]:
                d = joints.get(f, {})
                if f == "thumb":
                    parts.append(
                        f"T(mcp={fmt_angle(d.get('mcp')).strip()},"
                        f"pip={fmt_angle(d.get('pip')).strip()},"
                        f"dip={fmt_angle(d.get('dip')).strip()},"
                        f"ip={fmt_angle(d.get('ip')).strip()},"
                        f"abd={fmt_angle(splay.get('thumb')).strip()})"
                    )
                else:
                    parts.append(
                        f"{f[0].upper()}(m={fmt_angle(d.get('mcp')).strip()},"
                        f"p={fmt_angle(d.get('pip')).strip()},"
                        f"d={fmt_angle(d.get('dip')).strip()},"
                        f"abd={fmt_angle(splay.get(f)).strip()})"
                    )
            print("  " + "  ".join(parts))
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\nStopped streaming.\n")

def menu():
    # Choose your preferred hand here: 'right', 'left', or 'any'
    hand_pref = 'right'

    with UltraleapHandKinematics(hand=hand_pref) as k:
        print("\nUltraleap Hand Kinematics — quick menu")
        print("--------------------------------------")
        print(f"Hand preference: {hand_pref}")
        print("Tips:")
        print("  - Make sure your chosen hand is visible to the sensor.")
        print("  - Use options 1 & 2 to capture origins from your current pose.\n")

        while True:
            print("Menu:")
            print("  1) Calibrate origin XYZ from current palm position")
            print("  2) Calibrate origin RPY from current palm orientation")
            print("  3) Print ONE snapshot of all finger angles")
            print("  4) STREAM finger angles (Ctrl+C to stop)")
            print("  5) Show current raw palm XYZ & RPY (no origin subtraction)")
            print("  0) Quit")
            choice = input("> ").strip()

            if choice == "1":
                ok = k.set_origin_xyz(use_current=True)
                print("  ✔ XYZ origin captured from current palm" if ok else "  ✖ Failed: no hand visible")
            elif choice == "2":
                ok = k.set_origin_rpy(use_current=True)
                print("  ✔ RPY origin captured from current palm" if ok else "  ✖ Failed: no hand visible")
            elif choice == "3":
                print_one_snapshot(k)
            elif choice == "4":
                stream_angles(k, hz=20.0)
            elif choice == "5":
                xyz = k.get_xyz(raw=True)
                rpy = k.get_rpy(raw=True)
                if xyz is None or rpy is None:
                    print("  ✖ No hand visible")
                else:
                    print(f"  Raw palm XYZ (mm): ({xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f})")
                    print(f"  Raw palm RPY (deg): ({rpy[0]:.1f}, {rpy[1]:.1f}, {rpy[2]:.1f})")
            elif choice == "0":
                print("Bye!")
                break
            else:
                print("  (unknown option)")

if __name__ == "__main__":
    menu()
