import numpy as np
from scipy.interpolate import griddata

import os
import sys

# Assign directories and custom imports
cwd = os.path.dirname(os.path.abspath(__file__))
fitting_utils = cwd + '/utils/'

sys.path.append(fitting_utils)
import dust_utils as du
import proper_and_lum_dist as cosmo

# Define any required constants/arrays
sn_day_arr = np.arange(-20,51,1)

# Read in SALT2 SN IA file from Lou
salt2_spec = np.genfromtxt(fitting_utils + "salt2_template_0.txt", \
    dtype=None, names=['day', 'lam', 'flam'], encoding='ascii')

def model_sn(x, z, day, sn_av):

    # pull out spectrum for the chosen day
    day_idx_ = np.argmin(abs(sn_day_arr - day))
    day_idx = np.where(salt2_spec['day'] == sn_day_arr[day_idx_])[0]

    sn_spec_llam = salt2_spec['flam'][day_idx]
    sn_spec_lam = salt2_spec['lam'][day_idx]

    # ------ Apply dust extinction
    sn_dusty_llam = du.get_dust_atten_model(sn_spec_lam, sn_spec_llam, sn_av)

    # ------ Apply redshift
    sn_lam_z, sn_flam_z = cosmo.apply_redshift(sn_spec_lam, sn_dusty_llam, z)

    # ------ Calibration polynomial


    # ------ Regrid to Roman wavelength sampling
    sn_mod = griddata(points=sn_lam_z, values=sn_flam_z, xi=x)

    return sn_mod