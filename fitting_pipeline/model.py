import numpy as np

import os
import sys

class model(object):
    """
    Base class for defining a model.
    """

    def __init__(self, wav, luminosity):

        self.wav = wav
        self.luminosity = luminosity

    def apply_redshift():
        pass

    def apply_dust_attenuation():
        pass

    def apply_igm_attenuation():
        pass

    def apply_stellar_vdisp():
        pass


