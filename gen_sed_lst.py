import numpy as np
from astropy.io import fits

import os
import sys

home = os.getenv("HOME")
roman_slitless_dir = home + "/Documents/GitHub/roman-slitless/"
roman_direct_dir = home + "/Documents/roman_direct_sims/"
roman_sims_seds = home + "/Documents/roman_slitless_sims_seds/"

def get_sn_spec_path():
    """
    This function will assign a random spectrum from the basic SALT2 spectrum form Lou.
    Equal probability is given to any day relative to maximum. This will change for the
    final version. 

    The spectrum file contains a type 1A spectrum from -20 to +50 days relative to max.
    Since the -20 spectrum is essentially empty, I won't choose that spectrum for now.
    Also, for now, I will restrict this function to -5 to +20 days relative to maximum.
    """

    # Create array for days relative to max
    days_arr = np.arange(-5, 20, 1)

    # Read in SALT2 SN IA file  from Lou
    salt2_spec = np.genfromtxt(roman_sims_seds + "salt2_template_0.txt", \
        dtype=None, names=['day', 'lam', 'flam'], encoding='ascii')

    # choose a random day relative to max
    day_chosen = np.random.choice(days_arr)

    # pull out spectrum for the chosen day
    day_idx = np.where(salt2_spec['day'] == day_chosen)[0]

    sn_spec_lam = salt2_spec['lam'][day_idx]
    sn_spec_flam = salt2_spec['flam'][day_idx]

    # Save individual spectrum file if it doesn't already exist
    sn_spec_path = roman_sims_seds + "salt2_spec_day" + str(day_chosen) + ".txt"
    if not os.path.isfile(sn_spec_path):

        fh_sn = open(sn_spec_path, 'w')
        fh_sn.write("#  lam  flam")
        fh_sn.write("\n")

        for j in range(len(sn_spec_lam)):
            fh_sn.write("{:.2f}".format(sn_spec_lam[j]) + " " + str(sn_spec_flam[j]))
            fh_sn.write("\n")

        fh_sn.close()

    return sn_spec_path

def get_gal_spec_path():
    """
    For now this function will randomly assign one of seven composite
    stellar population SEDs from BC03 to the host galaxy.
    """

    # Assume stellar mass of host galaxy
    log_stellar_mass_arr = np.linspace(10.0, 11.0, 100)
    log_stellar_mass_chosen = np.random.choice(log_stellar_mass_arr)

    log_stellar_mass_str = "{:.2f}".format(log_stellar_mass_chosen).replace('.', 'p')

    # List of possible SEDs
    all_bc03_spec = np.array(['bc03_template_1_gyr.txt',
    'bc03_template_2_gyr.txt',
    'bc03_template_4_gyr.txt',
    'bc03_template_6_gyr.txt',
    'bc03_template_100_myr.txt',
    'bc03_template_300_myr.txt',
    'bc03_template_500_myr.txt'])

    bc03_spec_chosen = np.random.choice(all_bc03_spec)

    # Choose one of the BC03 spectra and multiply flux by stellar mass
    bc03_template = np.genfromtxt(roman_sims_seds + bc03_spec_chosen, dtype=None, names=True, encoding='ascii')
    bc03_lam = bc03_template['wav']
    bc03_llam = bc03_template['llam'] * 10**log_stellar_mass_chosen

    gal_spec_path = roman_sims_seds + bc03_spec_chosen.split(".txt")[0] + \
    "_ms" + log_stellar_mass_str + ".txt"

    # Save individual spectrum file if it doesn't already exist
    if not os.path.isfile(gal_spec_path):

        fh_gal = open(gal_spec_path, 'w')
        fh_gal.write("#  lam  llam")
        fh_gal.write("\n")

        for j in range(len(bc03_llam)):

            fh_gal.write("{:.2f}".format(bc03_lam[j]) + " " + str(bc03_llam[j]))
            fh_gal.write("\n")

        fh_gal.close()

    return gal_spec_path

def main():

    # Set image params
    img_sim_dir = roman_direct_dir + 'K_akari_rotate_subset/'
    img_truth_dir = roman_direct_dir + 'K_akari_rotate_truth/'
    img_basename = 'akari_match_'
    img_suffix = 'Y106_11_1'

    # Open empty file for saving sed.lst
    fh = open(roman_slitless_dir + 'pyLINEARSN/sed.lst', 'w')

    # Write header
    fh.write("# 1: SEGMENTATION ID" + "\n")
    fh.write("# 2: SED FILE" + "\n")
    fh.write("# 3: REDSHIFT" + "\n")
    fh.write("\n")

    # Read in catalog from SExtractor
    cat_header = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'ALPHA_J2000', 'DELTA_J2000', \
    'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_AUTO', 'MAGERR_AUTO', 'FLUX_RADIUS', 'FWHM_IMAGE']
    cat = np.genfromtxt(img_sim_dir + img_basename + img_suffix + '.cat', \
        dtype=None, names=cat_header, skip_header=11, encoding='ascii')

    # Redshift array to choose redshift from
    redshift_arr = np.arange(0.1, 1.0, 0.01)

    # Read in SN truth file
    sn_truth = fits.open(img_truth_dir + img_basename + 'index_' + img_suffix + '_sn.fits')

    # Assign SN ra and dec to arrays
    #sn_ra = sn_truth[1].data['ra']
    #sn_dec = sn_truth[1].data['dec']

    # --------- Dummy stuff for now --------- # 
    # This has to be automated later.
    # Right now the SN truth tables seem to indicate on SNe within 
    # the observed image. The SN coordinates seem to be far away from image coords.
    # Identify some dummy host and SNe segmentation IDs for now.
    # This is eyeballed using ds9 by looking for pairs of close objects.
    #sn_ra = np.array([71.0194361, ])
    #sn_dec = np.array([-53.6042292, ])

    #host_ra = np.array([71.0192822, ])
    #host_dec = np.array([-53.6038719, ])

    host_segids = np.array([475, 755, 548, 207])
    sn_segids = np.array([481, 753, 547, 241])
    
    # --------- End dummy defs --------- #

    # Loop over all objects and assign spectra
    for i in range(len(cat)):

        #mag = cat['MAG_AUTO'][i]

        # Match with the SN positions in the 
        # truth files and assign spectra
        #np.argmin(abs())

        current_id = cat['NUMBER'][i]

        if current_id in sn_segids:
            spec_path = get_sn_spec_path()
        else:
            spec_path = get_gal_spec_path()

        # Choose random redshift
        chosen_redshift = "{:.2f}".format(np.random.choice(redshift_arr))

        print(current_id, spec_path, chosen_redshift)

        # Now write
        fh.write(str(current_id) + " " + spec_path + " " + chosen_redshift)
        fh.write("\n")

    # Close sed.lst file to save
    fh.close()    

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)