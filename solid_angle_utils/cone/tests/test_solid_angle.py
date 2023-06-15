import solid_angle_utils
import numpy as np


def test_zeros():
    assert solid_angle_utils.sr2squaredeg(0.0) == 0.0
    assert solid_angle_utils.squaredeg2sr(0.0) == 0.0


def test_forth_and_back():
    for sr in np.linspace(-89, 123, 137 * 42):
        sd = solid_angle_utils.sr2squaredeg(sr)
        sr_back = solid_angle_utils.squaredeg2sr(sd)

        np.testing.assert_approx_equal(
            actual=sr_back, desired=sr, significant=7,
        )
