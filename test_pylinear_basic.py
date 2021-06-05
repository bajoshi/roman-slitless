import numpy as np
from astropy.io import fits

import os
import sys

import matplotlib.pyplot as plt

home = os.getenv('HOME')
basic_testdir = '/Volumes/Joshi_external_HDD/Roman/roman_direct_sims/pylinear_basic_test/'

ext_spec_filename = basic_testdir + 'romansim_grism_basic_test_x1d.fits'

exthdu = fits.open(ext_spec_filename)








