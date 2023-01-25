import numpy as np
from numpy.random import default_rng

import os
import socket

import dust_utils as du
from apply_redshift import apply_redshift

if 'plffsn2' in socket.gethostname():
    extdir = '/astro/ffsn/Joshi/'
    roman_slitless_dir = extdir + "GitHub/roman-slitless/"
else:
    extdir = '/Volumes/Joshi_external_HDD/Roman/'
    home = os.getenv("HOME")
    roman_slitless_dir = home + "/Documents/GitHub/roman-slitless/"

roman_sims_seds = extdir + 'roman_slitless_sims_seds/'
fitting_utils = roman_slitless_dir + "fitting_pipeline/utils/"

assert os.path.isdir(roman_sims_seds)
assert os.path.isdir(fitting_utils)

# ---------------------------
# Read in SALT2 SN IA file  from Lou
salt2_spec = np.genfromtxt(fitting_utils + "templates/salt2_template_0.txt", 
    dtype=None, names=['day', 'lam', 'llam'], encoding='ascii')

def get_sn_spec_path(cosmo, redshift, day_chosen=None, chosen_av=None):
    """
    This function will assign a random spectrum from the basic SALT2 spectrum form Lou.
    Equal probability is given to any day relative to maximum. This will change for the
    final version. 

    The spectrum file contains a type 1A spectrum from -20 to +50 days relative to max.
    Since the -20 spectrum is essentially empty, I won't choose that spectrum.
    """

    # choose a random day relative to max
    if not day_chosen:
        # Create array for days relative to max
        days_arr = np.arange(-5, 6, 1)
        day_chosen = np.random.choice(days_arr)

    # pull out spectrum for the chosen day
    day_idx = np.where(salt2_spec['day'] == day_chosen)[0]

    sn_spec_lam = salt2_spec['lam'][day_idx]
    sn_spec_llam = salt2_spec['llam'][day_idx]

    # Apply dust extinction
    # Apply Calzetti dust extinction depending on av value chosen
    if not chosen_av:
        rng = default_rng()
        chosen_av = rng.exponential(0.5)
        # the argument above is the scaling factor for the exponential
        # see: https://numpy.org/doc/stable/reference/random/generated/numpy.random.exponential.html
        # higher beta values give "flatter" exponentials
        # I want a fairly steep exponential decline toward high Av values
        if chosen_av > 3.0: chosen_av = 3.0

    sn_dusty_llam = du.get_dust_atten_model(sn_spec_lam, sn_spec_llam, chosen_av)

    # Apply redshift
    pc2cm = 3.086e18  # 1 parsec to centimeter
    dl = cosmo.luminosity_distance(redshift).value * 1e6 * pc2cm  # in cm
    sn_wav_z, sn_flux = apply_redshift(sn_spec_lam, sn_dusty_llam, redshift, dl)

    # Save individual spectrum file if it doesn't already exist
    sn_spec_path = roman_sims_seds \
                   + "salt2_spec_day" + str(day_chosen) \
                   + "_z" + "{:.3f}".format(redshift).replace('.', 'p') \
                   + "_av" + "{:.3f}".format(chosen_av).replace('.', 'p') \
                   + ".txt"

    if not os.path.isfile(sn_spec_path):

        fh_sn = open(sn_spec_path, 'w')
        fh_sn.write("#  lam  flux")
        fh_sn.write("\n")

        for j in range(len(sn_wav_z)):
            fh_sn.write("{:.2f}".format(sn_wav_z[j]) + " " + str(sn_flux[j]))
            fh_sn.write("\n")

        fh_sn.close()

    return sn_spec_path, redshift, day_chosen, chosen_av




