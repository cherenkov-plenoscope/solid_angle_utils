import solid_angle_utils
import pytest
import numpy as np


def normalized(v):
    return v / np.linalg.norm(v)


def test_basics():
    x, y, z = np.eye(3)
    sa = solid_angle_utils.triangle.solid_angle(
        v0=x,
        v1=y,
        v2=z,
    )

    full_sphere_solid_angle = 4 * np.pi
    np.testing.assert_almost_equal(sa, full_sphere_solid_angle / 8)

    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                sa = solid_angle_utils.triangle.solid_angle(
                    v0=(sx, 0, 0),
                    v1=(0, sy, 0),
                    v2=(0, 0, sz),
                )
                np.testing.assert_almost_equal(sa, full_sphere_solid_angle / 8)


def test_bad_vertex_norm_or_radius():
    with pytest.raises(AssertionError):
        sa = solid_angle_utils.triangle.solid_angle(
            v0=(1.1, 0, 0),
            v1=(0, 1, 0),
            v2=(0, 0, 1),
            delta_r=1e-6,
        )


def test_bad_vertex_norm_or_radius_many_combinations():
    with pytest.raises(AssertionError):

        def inc(v, dim):
            v = np.array(v, dtype=np.float64)
            v[dim] = v[dim] * (1 + 1e-2)
            return v

        for dim in range(3):
            sa = solid_angle_utils.triangle.solid_angle(
                v0=inc((1, 0, 0), dim),
                v1=inc((0, 1, 0), dim),
                v2=inc((0, 0, 1), dim),
                delta_r=1e-6,
            )


def test_angle_between():
    x, y, z = np.eye(3)
    almost_equal = np.testing.assert_almost_equal
    angle_between = solid_angle_utils.triangle._angle_between
    PI = np.pi

    almost_equal(angle_between(x, y), PI / 2)
    almost_equal(angle_between(y, z), PI / 2)
    almost_equal(angle_between(z, x), PI / 2)
    almost_equal(angle_between(x, x), 0.0)
    almost_equal(angle_between(y, y), 0.0)
    almost_equal(angle_between(z, z), 0.0)
    almost_equal(angle_between(x, -x), PI)
    almost_equal(angle_between(y, -y), PI)
    almost_equal(angle_between(z, -z), PI)


def test_ray_parameter_for_closest_distance_to_point():
    p = solid_angle_utils.triangle._ray_parameter_for_closest_distance_to_point(
        support_vector=[0, 0, -1],
        direction_vector=normalized([0, 0, 1]),
        point=[0, 0, 1],
    )
    np.testing.assert_almost_equal(p, 2)

    p = solid_angle_utils.triangle._ray_parameter_for_closest_distance_to_point(
        support_vector=[0, 0, 0],
        direction_vector=normalized([1, 1, 0]),
        point=[1, 0, 0],
    )
    np.testing.assert_almost_equal(p, np.sqrt(0.5))


def test_surface_tangent():
    x, y, z = np.eye(3)

    t = solid_angle_utils.triangle._surface_tangent(x, y)
    np.testing.assert_array_almost_equal(t, y)

    t = solid_angle_utils.triangle._surface_tangent(x, z)
    np.testing.assert_array_almost_equal(t, z)

    t = solid_angle_utils.triangle._surface_tangent(y, z)
    np.testing.assert_array_almost_equal(t, z)


def test_tiny_triangles():
    similarity_log10 = []
    arcs_deg = np.geomspace(1e-1, 1e-6, 100)
    for arc_deg in arcs_deg:
        v0 = normalized([0, 0, 1])
        v1 = normalized([np.deg2rad(arc_deg), 0, 1])
        v2 = normalized([0, np.deg2rad(arc_deg), 1])

        spherical_triangle_solid_angle = solid_angle_utils.triangle._solid_angle_of_spherical_triangle_according_to_girad(
            v0, v1, v2
        )
        flat_triangle_area = solid_angle_utils.triangle._area_of_flat_triangle(
            v0, v1, v2
        )

        delta = np.abs(spherical_triangle_solid_angle - flat_triangle_area)
        log10_delta = np.log10(flat_triangle_area / delta)

        similarity_log10.append(log10_delta)

    best_match_arc_deg = arcs_deg[np.argmax(similarity_log10)]

    assert 1e-1 > best_match_arc_deg > 1e-6


def test_provoke_warning():
    x, y, z = np.eye(3)

    # triangle with two large and one tiny angle between vertices
    v0 = x
    v1 = normalized([0, 1e-6, 1])
    v2 = z

    with pytest.warns(RuntimeWarning) as recorded_warnings:
        sa = solid_angle_utils.triangle.solid_angle(v0, v1, v2)
        assert len(recorded_warnings) == 1
