import numpy as np


def solid_angle(half_angle):
    cap_hight = 1.0 - np.cos(half_angle)
    return 2.0 * np.pi * cap_hight


def half_angle(solid_angle):
    cap_hight = solid_angle / (2.0 * np.pi)
    return np.arccos(-cap_hight + 1.0)


def half_angle_space(stop_half_angle, num):
    assert num >= 1
    assert stop_half_angle > 0.0

    cone_stop_sr = cone_solid_angle(
        cone_radial_opening_angle_rad=stop_half_angle
    )
    cone_step_sr = cone_stop_sr / num

    edges = [0]
    for i in np.arange(1, num):
        a = cone_radial_opening_angle(i * cone_step_sr)
        edges.append(a)
    edges.append(stop_half_angle)
    return np.array(edges)
