import numpy as np


def addnoise(flux, scale=None):

    noised_spec = np.random.normal(loc=flux, scale=scale, size=len(flux))

    return noised_spec
