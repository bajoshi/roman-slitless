import numpy as np

import os
import sys

class galaxy(object):
    """
    Base class for defining observed data for a galaxy object.
    """

    def __init__(self, wav, fluxes, flux_errors):
        
        #super(galaxy, self).__init__()

        self.wav = wav
        self.fluxes = fluxes
        self.flux_errors = flux_errors




