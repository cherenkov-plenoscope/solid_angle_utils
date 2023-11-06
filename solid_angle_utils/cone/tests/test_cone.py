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
            actual=half_angle_back,
            desired=half_angle,
            significant=7,
        )


def test_space():
    NN = 100
    half_angle_deg = 45

    space_rad = solid_angle_utils.cone.half_angle_space(
        start_half_angle_rad=0.0,
        stop_half_angle_rad=np.deg2rad(half_angle_deg),
        num=NN,
    )
    expected_segment_solid_angle = (
        1 / (NN - 1)
    ) * solid_angle_utils.cone.solid_angle(
        half_angle_rad=np.deg2rad(half_angle_deg)
    )

    assert len(space_rad) == NN

    sa_segments = np.zeros(NN - 1)
    for i in range(NN - 1):
        ha_start_rad = space_rad[i]
        ha_stop_rad = space_rad[i + 1]

        sa_stop = solid_angle_utils.cone.solid_angle(
            half_angle_rad=ha_stop_rad
        )
        sa_start = solid_angle_utils.cone.solid_angle(
            half_angle_rad=ha_start_rad
        )

        sa_segments[i] = sa_stop - sa_start

    assert (
        np.abs(np.mean(sa_segments) - expected_segment_solid_angle)
        < 1e-3 * expected_segment_solid_angle
    )
    assert np.std(sa_segments) < 1e-3 * expected_segment_solid_angle
