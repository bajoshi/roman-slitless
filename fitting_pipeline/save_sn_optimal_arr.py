import numpy as np
from tqdm import tqdm

import os
import sys

import matplotlib.pyplot as plt

# Assign directories and custom imports
cwd = os.path.dirname(os.path.abspath(__file__))
fitting_utils = cwd + '/utils/'

sys.path.append(fitting_utils)
import dust_utils as du

# Define any required constants/arrays
sn_scalefac = 2.0842526537870818e+48  # see sn_scaling.py 
sn_day_arr = np.arange(-19,51,1)

# Read in SALT2 SN IA file from Lou
salt2_spec = np.genfromtxt(fitting_utils + "salt2_template_0.txt", \
    dtype=None, names=['day', 'lam', 'flam'], encoding='ascii')

# REdshift array
redshift_arr = np.arange(0.01, 3.01, 0.01)

# Av array
av_arr = np.arange(0.5, 5.5, 0.5)

# -------- Now save all models to an npy array
total_models = len(sn_day_arr) * len(redshift_arr) * len(av_arr)  #  total models
print('Total models:', total_models)

# Get the wavelengths for one of the SN spectra
# They're all the same
sn_lam = salt2_spec['lam'][salt2_spec['day'] == 0]

# Empty array to write to
allmods = []

for d in tqdm(range(len(sn_day_arr)), desc='SN Phase'):

    day = sn_day_arr[d]

    day_idx = np.where(salt2_spec['day'] == day)[0]
    spec = salt2_spec['flam'][day_idx]

    # Clip model to observed wavelength range
    # This must be the same range as the clipped range for the extracted spectra
    # Also make sure the wav sampling is the same
    # Currently the pylinear x1d prism spectra have
    #  np.arange(7500.0, 18030.0, 30.0)
    

    for r in range(len(redshift_arr)):

        z = redshift_arr[r]

    adiff = np.abs(dl_z_arr - z)
    z_idx = np.argmin(adiff)
    dl = dl_cm_arr[z_idx]

    redshifted_flux = restframe_lum / (4 * np.pi * dl * dl * (1 + z))

        spec_redshifted = 

        for a in range(len(av_arr)):















