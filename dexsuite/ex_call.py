from leap_tracking import UltraleapHandKinematics
import leap, time

with UltraleapHandKinematics(hand='right', tracking_mode=leap.TrackingMode.Desktop) as hk:
    # Zero pose calibration (optional): capture current pose as origin
    hk.set_origin_xyz(use_current=True)
    hk.set_origin_rpy(use_current=True)

    while True:
        xyz = hk.get_xyz()              # mm, relative to origin
        rpy = hk.get_rpy()              # deg, relative to origin
        idx = {
            'mcp': hk.get_index_mcp(),
            'pip': hk.get_index_pip(),
            'dip': hk.get_index_dip(),
            'abd': hk.get_abduction('index'),
        }
        thumb = hk.get_thumb_angles()   # {'mcp', 'pip', 'dip', 'ip'}
        # TODO: feed xyz/rpy/joint angles to your robot hand controller
        # print(xyz, rpy, idx, thumb)
        time.sleep(0.01)
