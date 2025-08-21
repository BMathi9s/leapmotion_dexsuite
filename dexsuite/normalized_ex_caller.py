#!/usr/bin/env python3
import time
from typing import Optional

# If separate file, use: from ultraleap_hand_kinematics import UltraleapHandKinematics
from raw_hand_tracking import UltraleapHandKinematics
from normalised_tracking import CalibrationAndNormalization

def fmt(x: Optional[float], w: int = 5, p: int = 2) -> str:
    return ("{:"+str(w)+"."+str(p)+"f}").format(x) if isinstance(x, (int, float)) else " " * (w+1)

def run():
    hand_pref = "right"  # 'left'|'right'|'any'
    with UltraleapHandKinematics(hand=hand_pref) as k:
        cal = CalibrationAndNormalization(k)
        

        # --- Calibration wizards ---
        cal.run_worldframe_wizard()         # load or new (1..6 keys for new)
        cal.run_finger_calibration_wizard() # load or new (per-finger sampling)

        
        # --- Stream normalized data ---
        print("\nStreaming NORMALIZED data (Ctrl+C to stop):")
        print("  XYZ: X,Y in [-1,1], Z in [0,1]")
        print("  Joints/Abduction: in [-1,1]\n")

        try:
            while True:
                try:
                    nx, ny, nz = cal.get_normalized_xyz(use_raw_world=True)
                    nr, np ,yaw = cal.get_normalized_rpy(use_raw_world=True)
                    joints = cal.get_all_normalized_joints()
                    abds   = cal.get_all_normalized_abductions()
                except Exception as e:
                    print(f"  ⚠ {e}")
                    time.sleep(0.2)
                    continue

                # Build a compact single-line readout
                parts = [f"XYZ=({fmt(nx)}, {fmt(ny)}, {fmt(nz)})"]
                for f in ("thumb","index","middle","ring","pinky"):
                    jd = joints.get(f, {})
                    if f == "thumb":
                        parts.append(
                            f"T(m={fmt(jd.get('mcp'))},p={fmt(jd.get('pip'))},"
                            f"d={fmt(jd.get('dip'))},ip={fmt(jd.get('ip'))},"
                            f"abd={fmt(abds.get('thumb'))})"
                        )
                    else:
                        parts.append(
                            f"{f[0].upper()}(m={fmt(jd.get('mcp'))},p={fmt(jd.get('pip'))},"
                            f"d={fmt(jd.get('dip'))},abd={fmt(abds.get(f))})"
                        )
                print("  " + "  ".join(parts))
                time.sleep(0.05)  # ~20 Hz
        except KeyboardInterrupt:
            print("\nStopped.")

if __name__ == "__main__":
    run()
