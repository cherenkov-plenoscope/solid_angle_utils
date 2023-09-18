import solid_angle_utils
import numpy as np


def test_convert_sr_squaredeg():
    seed = 42
    prng = np.random.Generator(np.random.PCG64(seed))

    for i in range(1000):
        ii_sr = prng.uniform() * 4 * (4 * np.pi)
        oo_sqdeg = solid_angle_utils.sr2squaredeg(ii_sr)
        oo_sr = solid_angle_utils.squaredeg2sr(oo_sqdeg)
        np.testing.assert_almost_equal(ii_sr, oo_sr)
