import solid_angle_utils
import numpy as np


def test_cone_zero_solid_angle():
    sa = solid_angle_utils.cone.solid_angle(half_angle_rad=0.0)
    assert sa == 0.0


def test_cone_zero_half_angle():
    oa = solid_angle_utils.cone.half_angle(solid_angle_sr=0.0)
    assert oa == 0


def test_cone_conversion_forth_and_back():
    for half_angle in np.linspace(0, np.pi, 137):
        solid_angle = solid_angle_utils.cone.solid_angle(half_angle)
        half_angle_back = solid_angle_utils.cone.half_angle(solid_angle)

        np.testing.assert_approx_equal(
            actual=half_angle_back, desired=half_angle, significant=7,
        )
