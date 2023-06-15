import numpy as np
from . import cone


def sr2squaredeg(solid_angle_sr):
    return solid_angle_sr * (180.0 / np.pi) ** 2


def squaredeg2sr(solid_angle_squaredeg):
    return solid_angle_squaredeg * (np.pi / 180) ** 2