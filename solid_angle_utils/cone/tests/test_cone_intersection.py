import numpy as np
import solid_angle_utils as sau
from solid_angle_utils.cone.mazonka2012solid import (
    _intersection_of_two_cones_with_debug,
)


def test_cases():
    PI = np.pi
    TAU = 2.0 * PI

    isec, dbg = _intersection_of_two_cones_with_debug(
        half_angle_one_rad=0.5 * PI + 1e-6,
        half_angle_two_rad=0.1,
        angle_between_cones=0.1,
        epsilon_rad=np.deg2rad(1e-3),
    )
    assert dbg == "4.4.2 Inverted cones"
    np.testing.assert_almost_equal(
        isec, sau.cone.solid_angle(half_angle_rad=0.1)
    )

    isec, dbg = _intersection_of_two_cones_with_debug(
        half_angle_one_rad=0.5 * PI + 1e-6,
        half_angle_two_rad=0.1,
        angle_between_cones=np.deg2rad(150),
        epsilon_rad=np.deg2rad(1e-3),
    )
    # assert dbg == "no overlap"
    np.testing.assert_almost_equal(isec, 0.0)

    isec, dbg = _intersection_of_two_cones_with_debug(
        half_angle_one_rad=0.2,
        half_angle_two_rad=0.1,
        angle_between_cones=0.0,
        epsilon_rad=np.deg2rad(1e-3),
    )
    assert dbg == "4.4.3 Co-directed cone"
    np.testing.assert_almost_equal(isec, sau.cone.solid_angle(0.1))

    isec, dbg = _intersection_of_two_cones_with_debug(
        half_angle_one_rad=0.2,
        half_angle_two_rad=0.1,
        angle_between_cones=PI,
        epsilon_rad=np.deg2rad(1e-3),
    )
    # assert dbg == "no overlap"
    np.testing.assert_almost_equal(isec, 0.0)

    isec, dbg = _intersection_of_two_cones_with_debug(
        half_angle_one_rad=PI / 2,
        half_angle_two_rad=0.1,
        angle_between_cones=PI / 2,
        epsilon_rad=np.deg2rad(1e-3),
    )
    assert dbg == "default"
    np.testing.assert_almost_equal(
        isec, 0.5 * sau.cone.solid_angle(half_angle_rad=0.1)
    )

    isec, dbg = _intersection_of_two_cones_with_debug(
        half_angle_one_rad=PI / 2,
        half_angle_two_rad=PI / 2,
        angle_between_cones=PI / 2,
        epsilon_rad=np.deg2rad(1e-3),
    )
    assert dbg == "4.4.6 two hemispheres"
    np.testing.assert_almost_equal(isec, PI)

    isec, dbg = _intersection_of_two_cones_with_debug(
        half_angle_one_rad=np.deg2rad(70),
        half_angle_two_rad=np.deg2rad(1e-6),
        angle_between_cones=np.deg2rad(10),
        epsilon_rad=np.deg2rad(1e-3),
    )
    assert dbg == "4.4.5 Narrow cone, gamma2 < -theta2"
    np.testing.assert_almost_equal(
        isec, sau.cone.solid_angle(np.deg2rad(1e-6))
    )

    isec, dbg = _intersection_of_two_cones_with_debug(
        half_angle_one_rad=np.deg2rad(70),
        half_angle_two_rad=np.deg2rad(1e-4),
        angle_between_cones=np.deg2rad(70),
        epsilon_rad=np.deg2rad(1e-3),
    )
    assert dbg == "4.4.5 Narrow cone, else"
    np.testing.assert_almost_equal(
        isec, 0.5 * sau.cone.solid_angle(np.deg2rad(1e-4)), decimal=12
    )


def test_simple():
    PI = np.pi
    TAU = 2.0 * PI

    isec, _ = _intersection_of_two_cones_with_debug(
        half_angle_one_rad=np.deg2rad(70),
        half_angle_two_rad=np.deg2rad(15),
        angle_between_cones=np.deg2rad(55),
        epsilon_rad=np.deg2rad(1e-3),
    )
    np.testing.assert_almost_equal(isec, sau.cone.solid_angle(np.deg2rad(15)))

    isec, _ = _intersection_of_two_cones_with_debug(
        half_angle_one_rad=np.deg2rad(70),
        half_angle_two_rad=np.deg2rad(15),
        angle_between_cones=np.deg2rad(85),
        epsilon_rad=np.deg2rad(1e-3),
    )
    np.testing.assert_almost_equal(isec, 0.0)

    isec, _ = _intersection_of_two_cones_with_debug(
        half_angle_one_rad=np.deg2rad(70),
        half_angle_two_rad=np.deg2rad(15),
        angle_between_cones=np.deg2rad(70),
        epsilon_rad=np.deg2rad(1e-3),
    )
    np.testing.assert_almost_equal(
        isec, 0.5 * sau.cone.solid_angle(np.deg2rad(15)), decimal=2
    )


def test_sweep_small_cone_over_the_edge_of_a_very_big_cone():
    small_cone_half_angle_rad = np.deg2rad(15)
    large_cone_half_angle_rad = np.deg2rad(70)

    sweep_offsets_rad = np.linspace(
        -small_cone_half_angle_rad,
        small_cone_half_angle_rad,
        1000,
    )

    isecs = []
    approx = []
    for sweep_offset_rad in sweep_offsets_rad:
        isec = sau.cone.intersection_of_two_cones(
            half_angle_one_rad=large_cone_half_angle_rad,
            half_angle_two_rad=small_cone_half_angle_rad,
            angle_between_cones=large_cone_half_angle_rad + sweep_offset_rad,
            epsilon_rad=np.deg2rad(1e-3),
        )
        isecs.append(isec)

        approx.append(
            circular_segment(
                radius=small_cone_half_angle_rad, height=sweep_offset_rad
            )
        )

    isecs = np.asarray(isecs)
    approx = np.asarray(approx)

    np.testing.assert_almost_equal(isecs, approx, decimal=2)


def circular_segment(radius, height):
    R = radius
    d = height
    acos = np.arccos
    sq = np.sqrt
    A = R**2 * acos(d / R) - d * sq(R**2 - d**2)
    return A
