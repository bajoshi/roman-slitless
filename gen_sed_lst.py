import numpy as np
from astropy.io import fits

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
astropy_cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)

import os
import sys
import subprocess
from tqdm import tqdm
import time
import socket

from numba import jit

import matplotlib.pyplot as plt

# Define constants
Lsol = 3.826e33

# -----------------
if 'plffsn2' in socket.gethostname():
    extdir = '/astro/ffsn/Joshi/'
    modeldir = extdir + 'bc03_output_dir/'
    
    roman_sims_seds = extdir + 'roman_slitless_sims_seds/'
    pylinear_lst_dir = extdir + 'pylinear_lst_files/'
    roman_direct_dir = extdir + 'roman_direct_sims/sims2021/'

    roman_slitless_dir = extdir + "GitHub/roman-slitless/"
    fitting_utils = roman_slitless_dir + "fitting_pipeline/utils/"

else:
    extdir = '/Volumes/Joshi_external_HDD/Roman/'
    modeldir = extdir + 'bc03_output_dir/m62/'
    
    roman_sims_seds = "/Volumes/Joshi_external_HDD/Roman/roman_slitless_sims_seds/"
    pylinear_lst_dir = "/Volumes/Joshi_external_HDD/Roman/pylinear_lst_files/"
    roman_direct_dir = extdir + 'roman_direct_sims/sims2021/'

    home = os.getenv("HOME")
    roman_slitless_dir = home + "/Documents/GitHub/roman-slitless/"
    fitting_utils = roman_slitless_dir + "fitting_pipeline/utils/"

assert os.path.isdir(modeldir)
assert os.path.isdir(roman_sims_seds)
assert os.path.isdir(pylinear_lst_dir)
assert os.path.isdir(roman_direct_dir)

sys.path.append(fitting_utils)
import proper_and_lum_dist as cosmo
import dust_utils as du

# Read in SALT2 SN IA file  from Lou
salt2_spec = np.genfromtxt(fitting_utils + "salt2_template_0.txt", \
    dtype=None, names=['day', 'lam', 'llam'], encoding='ascii')

model_lam = np.load(extdir + "bc03_output_dir/bc03_models_wavelengths.npy", mmap_mode='r')
model_ages = np.load(extdir + "bc03_output_dir/bc03_models_ages.npy", mmap_mode='r')

all_m62_models = []
tau_low = 0
tau_high = 20
for t in range(tau_low, tau_high, 1):
    tau_str = "{:.3f}".format(t).replace('.', 'p')
    a = np.load(modeldir + 'bc03_all_tau' + tau_str + '_m62_chab.npy', mmap_mode='r')
    all_m62_models.append(a)
    del a

# load models with large tau separately
all_m62_models.append(np.load(modeldir + 'bc03_all_tau20p000_m62_chab.npy', mmap_mode='r'))

# Also load in lookup table for luminosity distance
dl_cat = np.genfromtxt(fitting_utils + 'dl_lookup_table.txt', dtype=None, names=True)
# Get arrays 
dl_z_arr = np.asarray(dl_cat['z'], dtype=np.float64)
dl_cm_arr = np.asarray(dl_cat['dl_cm'], dtype=np.float64)
age_gyr_arr = np.asarray(dl_cat['age_gyr'], dtype=np.float64)

del dl_cat

# This class came from stackoverflow
# SEE: https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-python
class bcolors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_dl_at_z(z):

    adiff = np.abs(dl_z_arr - z)
    z_idx = np.argmin(adiff)
    dl = dl_cm_arr[z_idx]

    return dl

def apply_redshift(restframe_wav, restframe_lum, redshift):

    dl = get_dl_at_z(redshift)

    redshifted_wav = restframe_wav * (1 + redshift)
    redshifted_flux = restframe_lum / (4 * np.pi * dl * dl * (1 + redshift))

    return redshifted_wav, redshifted_flux

def get_sn_spec_path(redshift, day_chosen=None, chosen_av=None):
    """
    This function will assign a random spectrum from the basic SALT2 spectrum form Lou.
    Equal probability is given to any day relative to maximum. This will change for the
    final version. 

    The spectrum file contains a type 1A spectrum from -20 to +50 days relative to max.
    Since the -20 spectrum is essentially empty, I won't choose that spectrum for now.
    Also, for now, I will restrict this function to -5 to +20 days relative to maximum.
    """

    # Create array for days relative to max
    days_arr = np.arange(-19, 51, 1)

    # Define scaling factor
    # Check sn_scaling.py in same folder as this code
    sn_scalefac = 2.0842526537870818e+48

    # choose a random day relative to max
    if not day_chosen:
        day_chosen = np.random.choice(days_arr)

    # pull out spectrum for the chosen day
    day_idx = np.where(salt2_spec['day'] == day_chosen)[0]

    sn_spec_lam = salt2_spec['lam'][day_idx]
    sn_spec_llam = salt2_spec['llam'][day_idx] * sn_scalefac

    # Apply dust extinction
    # Apply Calzetti dust extinction depending on av value chosen
    if not chosen_av:
        av_arr = np.arange(0.0, 5.0, 0.001)  # in mags
        chosen_av = np.random.choice(av_arr)

    sn_dusty_llam = du.get_dust_atten_model(sn_spec_lam, sn_spec_llam, chosen_av)

    # Apply redshift
    sn_wav_z, sn_flux = apply_redshift(sn_spec_lam, sn_dusty_llam, redshift)

    # Save individual spectrum file if it doesn't already exist
    sn_spec_path = roman_sims_seds + "salt2_spec_day" + str(day_chosen) + \
    "_z" + "{:.3f}".format(redshift).replace('.', 'p') + \
    "_av" + "{:.3f}".format(chosen_av).replace('.', 'p') + \
    ".txt"

    if not os.path.isfile(sn_spec_path):

        fh_sn = open(sn_spec_path, 'w')
        fh_sn.write("#  lam  flux")
        fh_sn.write("\n")

        for j in range(len(sn_wav_z)):
            fh_sn.write("{:.2f}".format(sn_wav_z[j]) + " " + str(sn_flux[j]))
            fh_sn.write("\n")

        fh_sn.close()
    #print("Saved:", sn_spec_path)

    return sn_spec_path

def get_bc03_spec(age, logtau):

    tau = 10**logtau  # logtau is log of tau in gyr

    if tau < 20.0:

        tau_int_idx = int((tau - int(np.floor(tau))) * 1e3)
        age_idx = np.argmin(abs(model_ages - age*1e9))
        model_idx = tau_int_idx * len(model_ages)  +  age_idx

        models_taurange_idx = np.argmin(abs(np.arange(tau_low, tau_high, 1) - int(np.floor(tau))))
        models_arr = all_m62_models[models_taurange_idx]

    elif tau >= 20.0:
        
        logtau_arr = np.arange(1.30, 2.01, 0.01)
        logtau_idx = np.argmin(abs(logtau_arr - logtau))

        age_idx = np.argmin(abs(model_ages - age*1e9))
        model_idx = logtau_idx * len(model_ages) + age_idx

        models_arr = all_m62_models[-1]

    model_llam = models_arr[model_idx]

    return np.array(model_llam, dtype=np.float64)

def get_gal_spec_path(redshift):
    """
    This function will generate a template SED assuming
    a composite stellar population using BC03. 
    -- SFH is assumed to be exponential.
       -- where tau is in between 0.01 to 15.0 (in Gyr)
    -- Age is dependent on z and only allows for models that are 
       at least 100 Myr younger than the Universe.
    -- Dust is applied assuming a Calzetti form for the dust extinction law.
    -- Metallicity is one of the six options in BC03.
    -- Log of stellar mass/M_sol is between 10.0 to 11.5
    """

    plot_tocheck = False

    # ---------- Choosing stellar population parameters ----------- #
    # Choose stellar pop parameters at random
    # --------- Stellar mass
    log_stellar_mass_arr = np.linspace(10.0, 11.5, 100)
    log_stellar_mass_chosen = np.random.choice(log_stellar_mass_arr)

    log_stellar_mass_str = "{:.2f}".format(log_stellar_mass_chosen).replace('.', 'p')

    # --------- Age
    age_arr = np.arange(0.1, 13.0, 0.005)  # in Gyr

    # Now choose age consistent with given redshift
    # i.e., make sure model is not older than the Universe
    # Allowing at least 100 Myr for the first galaxies to form after Big Bang
    age_at_z = astropy_cosmo.age(redshift).value  # in Gyr
    age_lim = age_at_z - 0.1  # in Gyr

    chosen_age = np.random.choice(age_arr)
    while chosen_age > age_lim:
        chosen_age = np.random.choice(age_arr)

    # --------- SFH
    # Choose SFH form from a few different models
    # and then choose params for the chosen SFH form
    sfh_forms = ['instantaneous', 'constant', 'exponential', 'linearly_declining']

    # choose_sfh(sfh_forms[2])

    tau_arr = np.arange(0.01, 15.0, 0.005)  # in Gyr
    chosen_tau = np.random.choice(tau_arr)

    # --------- Metallicity
    metals_arr = np.array([0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05])
    # While the newer 2016 version has an additional metallicity
    # referred to as "m82", the documentation never specifies the 
    # actual metallicity associated with it. So I'm ignoring that one.
    metals = 0.02 # np.random.choice(metals_arr)

    # Get hte metallicity string
    if metals == 0.0001:
        metallicity = 'm22'
    elif metals == 0.0004:
        metallicity = 'm32'
    elif metals == 0.004:
        metallicity = 'm42'
    elif metals == 0.008:
        metallicity = 'm52'
    elif metals == 0.02:
        metallicity = 'm62'
    elif metals == 0.05:
        metallicity = 'm72'

    # ----------- CALL BC03 -----------
    # The BC03 generated spectra will always be at redshift=0 and dust free.
    # This code will apply dust extinction and redshift effects manually
    #outdir = home + '/Documents/bc03_output_dir/'
    #gen_bc03_spectrum(chosen_tau, metals, outdir)
    logtau = np.log10(chosen_tau)
    bc03_spec_wav = np.array(model_lam, dtype=np.float64)
    bc03_spec_llam = get_bc03_spec(chosen_age, logtau)

    # Apply Calzetti dust extinction depending on av value chosen
    av_arr = np.arange(0.0, 5.0, 0.001)  # in mags
    chosen_av = np.random.choice(av_arr)

    bc03_dusty_llam = du.get_dust_atten_model(bc03_spec_wav, bc03_spec_llam, chosen_av)

    # Multiply flux by stellar mass
    bc03_dusty_llam = bc03_dusty_llam * 10**log_stellar_mass_chosen

    # Convert to physical units
    bc03_dusty_llam *= Lsol

    # --------------------- CHECK ----------------------
    # ---------------------- TBD -----------------------
    # 1.
    # Given the distribution you have for SFHs here,
    # can you recover the correct cosmic star formation
    # history? i.e., if you took the distribution of models
    # you have and computed the cosmic SFH do you get the 
    # Madau diagram back?
    # 2.
    # Do your model galaxies follow other scaling relations?

    # Apply IGM depending on boolean flag
    #if apply_igm:
    #    pass

    bc03_wav_z, bc03_flux = apply_redshift(bc03_spec_wav, bc03_dusty_llam, redshift)

    if plot_tocheck:
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ax.set_xlabel(r'$\lambda\ \mathrm{[\AA]}$', fontsize=14)
        ax.set_ylabel(r'$f_\lambda\ \mathrm{[erg\, s^{-1}\, cm^{-2}\, \AA]}$', fontsize=14)
        
        #ax.plot(bc03_spec_wav, bc03_spec_llam, label='Orig model')
        #ax.plot(bc03_spec_wav, bc03_dusty_llam, label='Dusty model')
        ax.plot(bc03_wav_z, bc03_flux, label='Redshfited dusty model with chosen Ms')

        ax.legend(loc=0)

        ax.set_xlim(500, 25000)

        plt.show()

    # Save file
    gal_spec_path = roman_sims_seds + 'bc03_template' + \
    "_z" + "{:.3f}".format(redshift).replace('.', 'p') + \
    "_ms" + log_stellar_mass_str + \
    "_age" + "{:.3f}".format(chosen_age).replace('.', 'p') + \
    "_tau" + "{:.3f}".format(chosen_tau).replace('.', 'p') + \
    "_met" + "{:.4f}".format(metals).replace('.', 'p') + \
    "_av" + "{:.3f}".format(chosen_av).replace('.', 'p') + \
    ".txt"

    # Print info to screen
    # print("\n")
    # print("--------------------------------------------------")
    # print("Randomly chosen redshift:", redshift)
    # print("Age limit at redshift [Gyr]:", age_lim)
    # print("\nRandomly chosen stellar population parameters:")
    # print("Age [Gyr]:", chosen_age)
    # print("log(stellar mass) [log(Ms/M_sol)]:", log_stellar_mass_chosen)
    # print("Tau [exp. decl. timescale, Gyr]:", chosen_tau)
    # print("Metallicity (abs. frac.):", metals)
    # print("Dust extinction in V-band (mag):", chosen_av)
    # print("\nWill save to file:", gal_spec_path)

    # Save individual spectrum file if it doesn't already exist
    if not os.path.isfile(gal_spec_path):

        fh_gal = open(gal_spec_path, 'w')
        fh_gal.write("#  lam  flux")
        fh_gal.write("\n")

        for j in range(len(bc03_flux)):

            fh_gal.write("{:.2f}".format(bc03_wav_z[j]) + " " + str(bc03_flux[j]))
            fh_gal.write("\n")

        fh_gal.close()

    return gal_spec_path

def get_match(ra_arr, dec_arr, ra_to_check, dec_to_check, tol_arcsec=0.3):

    # Matching tolerance
    tol = tol_arcsec/3600  # arcseconds expressed in degrees since our ra-decs are in degrees

    radiff = ra_arr - ra_to_check
    decdiff = dec_arr - dec_to_check
    idx = np.where( (np.abs(radiff) < tol) & (np.abs(decdiff) < tol) )[0]

    # Find closest match in case of multiple matches
    if len(idx) > 1:
        #tqdm.write(f"{bcolors.WARNING}")
        #tqdm.write("Multiple matches found. Picking closest one." + f"{bcolors.ENDC}")

        ra_two = ra_to_check
        dec_two = dec_to_check

        dist_list = []
        for v in range(len(idx)):

            ra_one = ra_arr[idx][v]
            dec_one = ra_arr[idx][v]

            dist = np.arccos(np.cos(dec_one*np.pi/180) * \
                np.cos(dec_two*np.pi/180) * np.cos(ra_one*np.pi/180 - ra_two*np.pi/180) + \
                np.sin(dec_one*np.pi/180) * np.sin(dec_two*np.pi/180))
            dist_list.append(dist)

        dist_list = np.asarray(dist_list)
        idx = np.argmin(dist_list)

    elif len(idx) == 1:
        idx = int(idx)

    # Increase tolerance if no match found at first
    elif len(idx) == 0:
        #tqdm.write(f"{bcolors.WARNING}" + "No matches found.")
        #tqdm.write("Redoing search within greater radius.")
        #tqdm.write("CAUTION: This method needs a LOT more testing." + f"{bcolors.ENDC}")
        #tqdm.write("Search centered on " + str(ra_to_check) + "   " + str(dec_to_check))
        #idx = get_match(ra_arr, dec_arr, ra_to_check, dec_to_check, tol_arcsec=tol_arcsec+0.1)
        idx = -99

    return idx

def get_sn_z(snmag):

    # Neglecting K-corr for now. When needed, the code to compute K-corr
    # is already written in massive-galaxies/grismz_pipeline/intg_lum_func.py

    g_absmag = -19.5  # assumed abs mag of SN Ia in HST ACS/F435W

    dist_mod = snmag - g_absmag
    dl = 10 * np.power(10, dist_mod/5.0)  # in parsecs
    dl *= 3.086e18  # convert to cm # this is the unit in the lookup table

    # Now reverse lookup z corresponding to dl in lookup table
    z_idx = np.argmin(abs(dl_cm_arr - dl))

    sn_z = dl_z_arr[z_idx]

    return sn_z

def gen_sed_lst():

    print(f"{bcolors.WARNING}")
    print("TODO: 1. What about the case in Y106_0_3 where there are 3 SN")
    print("but only two end up in the SED lst file? Probably the code checking")
    print("<if id_fetch in hostids> should be <for i in hostids> and loop instead of checking.")
    print("2. Make this code run on multiple cores.")
    print("3. How is the simulated magnitude taken into account?")
    print("i.e., if the SN mag is known and the redshift is known then")
    print("the SN phase is also a known quantity (so it shouldn't be randomized)?")
    print("4. Why are so many SNe matches not found?")
    print("5. Ask Kevin why some objects are missing from the truth files.")
    print(f"{bcolors.ENDC}")

    # Set image and truth params
    dir_img_part = 'part1'

    img_sim_dir = roman_direct_dir + 'K_5degimages_' + dir_img_part + '/'
    img_basename = '5deg_'
    img_filt = 'Y106_'

    truth_dir = roman_direct_dir + 'K_5degtruth/'
    truth_basename = '5deg_index_'

    # Read in the large truth arrays
    truth_match = fits.open(roman_direct_dir + '5deg_truth_gal.fits')

    # Arrays to loop over
    pointings = np.arange(1, 2)
    detectors = np.arange(11, 19, 1)

    for pt in tqdm(pointings, desc="Pointing"):
        for det in tqdm(detectors, desc="Detector", leave=False):

            img_suffix = img_filt + str(pt) + '_' + str(det)
            dir_img_name = img_basename + img_suffix + '_SNadded.fits'

            # Because some direct images are missing
            try:
                assert os.path.isfile(img_sim_dir + dir_img_name)
            except AssertionError:
                tqdm.write(f"{bcolors.FAIL}")
                tqdm.write("Missing image file for: " + dir_img_name)
                tqdm.write("Moving to next direct image.")
                tqdm.write(f"{bcolors.ENDC}")
                continue

            # Open empty file for saving sed.lst
            sed_filename = pylinear_lst_dir + \
            'sed_' + img_filt + str(pt) + '_' + str(det) + '.lst'
            tqdm.write(f"{bcolors.CYAN}" + "\nWill generate SED file: " + \
                sed_filename + f"{bcolors.ENDC}")

            # Check if the file exists 
            #if os.path.isfile(sed_filename):
            #    # Now check that it isn't empty
            #    sed_filesize = os.stat(sed_filename).st_size / 1000  # KB
            #    if sed_filesize > 30:  
            #        # I chose this limit after looking at file sizes by eye
            #        continue

            fh = open(sed_filename, 'w')

            # Write header
            fh.write("# 1: SEGMENTATION ID" + "\n")
            fh.write("# 2: SED FILE" + "\n")

            # Read in catalog from SExtractor
            cat_filename = img_sim_dir + img_basename + img_filt + \
                           str(pt) + '_' + str(det) + '_SNadded.cat'
            tqdm.write("Checking for catalog: " + cat_filename)

            if not os.path.isfile(cat_filename):
                tqdm.write("Cannot find object catalog. SExtractor will be run automatically.")

                # Now run SExtractor automatically
                # Set up sextractor
                img_filename = img_basename + img_filt + \
                               str(pt) + '_' + str(det) + '_SNadded.fits'
                checkimage = img_basename + img_filt + \
                               str(pt) + '_' + str(det) + '_segmap.fits'

                # Change directory to images directory
                os.chdir(img_sim_dir)

                tqdm.write(f"{bcolors.GREEN}" + "Running: " + "sex " + \
                    img_filename + " -c" + " roman_sims_sextractor_config.txt" + \
                    " -CATALOG_NAME " + os.path.basename(cat_filename) + \
                    " -CHECKIMAGE_NAME " + checkimage + f"{bcolors.ENDC}")

                # Use subprocess to call sextractor.
                # The args passed MUST be passed in this way.
                # i.e., args that would be separated by a space on the 
                # command line must be passed here separated by commas.
                # It will not work if you join all of these args in a 
                # string with spaces where they are supposed to be; 
                # even if the command looks right when printed out.
                sextractor = subprocess.run(['sex', img_filename, \
                    '-c', 'roman_sims_sextractor_config.txt', \
                    '-CATALOG_NAME', os.path.basename(cat_filename), \
                    '-CHECKIMAGE_NAME', checkimage], check=True)
                
                tqdm.write("Finished SExtractor run. Check cat and segmap if needed.")

                # Go back to roman-slitless directory
                os.chdir(roman_slitless_dir)

            cat_header = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'ALPHA_J2000', 'DELTA_J2000', \
            'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_AUTO', 'MAGERR_AUTO', 'FLUX_RADIUS', 'FWHM_IMAGE']
            cat = np.genfromtxt(cat_filename, dtype=None, names=cat_header, encoding='ascii')
            tqdm.write(f"{bcolors.GREEN}" + str(len(cat)) + " objects in catalog." + f"{bcolors.ENDC}")

            # Loop over all objects and assign spectra
            # Read in the truth files first
            truth_hdu_gal = fits.open(truth_dir + truth_basename + \
                            img_filt + str(pt) + '_' + str(det) + '.fits')
            truth_hdu_sn = fits.open(truth_dir + truth_basename + \
                            img_filt + str(pt) + '_' + str(det) + '_sn.fits')

            #print(repr(truth_hdu_gal[1].header))
            #print("-------------------------------------")
            #print(repr(truth_hdu_sn[1].header))
            
            #print("-------------------------------------")
            #print(truth_hdu_sn[1].data)

            #print("SN RAs:", truth_hdu_sn[1].data['ra'] * 180/np.pi)
            #print("SN DECs:", truth_hdu_sn[1].data['dec'] * 180/np.pi)

            #hostids = truth_hdu_sn[1].data['hostid']
            #print("Host galaxy IDs (corresponding to Kevin's IDs NOT SExtractor ID):")
            #print(hostids)

            #for hi in hostids:
            #    t = np.where(truth_hdu_gal[1].data['ind'] == hi)[0]
            #    print("Host index:", t)
            #    print("Host RA", truth_hdu_gal[1].data['ra'][t] * 180/np.pi)
            #    print("Host DEC", truth_hdu_gal[1].data['dec'][t] * 180/np.pi)

            #sys.exit(0)

            hostids = truth_hdu_sn[1].data['hostid']

            # assign arrays
            ra_gal  = truth_hdu_gal[1].data['ra']  * 180/np.pi
            dec_gal = truth_hdu_gal[1].data['dec'] * 180/np.pi

            # Also assign SN spectra to our added SN
            snadd_cat = np.load(cat_filename.replace('.cat', '.npy'))
            xi = snadd_cat[:, 0]
            yi = snadd_cat[:, 1]

            """
            # -----------
            ***** Short explaination of the code flow below. *****
            # -----------

            GOAL: Every object in the SExtractor catalog must be assigned a spectrum.

            Steps:
              1. Check for match with truth galaxy catalog.

              2. IF MATCH IS NOT FOUND:

                2A: Pick a random z

                2B: Is it one of the fake SNe that we added?
                    Yes --> Assign SN spectrum
                    No  --> Assign GALAXY spectrum

              3. IF MATCH IS FOUND:

                3A: Get truth-z

                3B: Is it a host galaxy?
                    Yes --> Find corresponding SN and assign SN and host galaxy
                            spectrum to the same redshift.
                        --> Ensure that when the SN ID is encountered in the loop
                            it is skipped.
                    No  --> Assign galaxy spectrum
            """

            assigned_sne = []

            for i in tqdm(range(len(cat)), desc="Object SegID", leave=False):

                # -------------- First match object -------------- #
                current_sextractor_id = int(cat['NUMBER'][i])

                if current_sextractor_id in assigned_sne:
                    tqdm.write("\nSN spectrum already assigned to " + str(current_sextractor_id))
                    tqdm.write("Skipping.")
                    continue
    
                # The -1 in the index is needed because the SExtractor 
                # catalog starts counting object ids from 1.
                ra_to_check = cat['ALPHA_J2000'][current_sextractor_id - 1]
                dec_to_check = cat['DELTA_J2000'][current_sextractor_id - 1]
    
                # Now match and get corresponding entry in the larger truth file
                idx = get_match(ra_gal, dec_gal, ra_to_check, dec_to_check)
                #tqdm.write("Matched idx: " + str(idx))

                if idx == -99:
                    tqdm.write("\nObjID:" + str(current_sextractor_id))
                    tqdm.write("No matches found in truth file.")

                    z_nomatch_gal = np.random.uniform(low=0.0, high=3.0)

                    # There are some galaxies that have no matches in the truth
                    # files. I'm assigning a random redshift and spectrum to them.
                    # They need to be given a spectrum otherwise the extraction
                    # is likely to be messed up since we'd then have objects that
                    # should have had dispersed light on the detector but didn't.
                    # Not sure what that would do to the extraction.
                    # ie. can't skip them like before.
                    # The SNe added through insert_sne.py should however be given
                    # SNe spectra.
                    current_x = cat['X_IMAGE'][i]
                    current_y = cat['Y_IMAGE'][i]
                    added_match = np.where((np.abs(xi - current_x) <= 3.0) & \
                                           (np.abs(yi - current_y) <= 3.0))[0]
                    if len(added_match) < 1:
                        tqdm.write('Assigning galaxy spectrum to object with no match in truth')
                        tqdm.write('and is not an object added through insert_sne.py')
                        spec_path = get_gal_spec_path(z_nomatch_gal)
                        fh.write(str(current_sextractor_id) + " " + spec_path + "\n")
                        continue

                    else:
                        tqdm.write("Assigning random redshift to added fake SN.")
                        # SN z must be consistent with cosmological dimming
                        z_nomatch_sn = get_sn_z(cat['MAG_AUTO'][i])
                        sn_spec_path = get_sn_spec_path(z_nomatch_sn)
                        fh.write(str(current_sextractor_id) + " " + sn_spec_path + "\n")
                        continue

                id_fetch = int(truth_hdu_gal[1].data['ind'][idx])
                #tqdm.write("ID to fetch from truth file: " + str(id_fetch))

                truth_idx = np.where(truth_match[1].data['gind'] == id_fetch)[0]
                # -------------- Matching done -------------- #

                # -------------- Get object redshift
                z = float(truth_match[1].data['z'][truth_idx])
                #tqdm.write("Object z: " + str(z))

                # -------------- Check if it is a SN host galaxy or SN itself
                # If it is then also call the sn SED path generation
                if id_fetch in hostids:
                    # Now you must find the corresponding SExtractor ID for the SN

                    sn_idx0 = np.where(hostids == id_fetch)[0]

                    sn_ra = truth_hdu_sn[1].data['ra'][sn_idx0] * 180/np.pi
                    sn_dec = truth_hdu_sn[1].data['dec'][sn_idx0] * 180/np.pi

                    sn_idx = get_match(cat['ALPHA_J2000'], cat['DELTA_J2000'], sn_ra, sn_dec)
                    if sn_idx == -99:
                        tqdm.write(f"{bcolors.FAIL}")
                        tqdm.write("Matching SN not found for hostid " + str(id_fetch))
                        tqdm.write("Assigning GALAXY spectrum.")
                        tqdm.write(f"{bcolors.ENDC}")
                        spec_path = get_gal_spec_path(z)
                        fh.write(str(current_sextractor_id) + " " + spec_path + "\n")
                        continue

                    snid = cat['NUMBER'][sn_idx]

                    # This means that the SN and host matched to the same location
                    # i.e., the SN is bright enough that it outshines the host 
                    if snid == current_sextractor_id:
                        sn_spec_path = get_sn_spec_path(z)
                        fh.write(str(snid) + " " + sn_spec_path + "\n")
                        tqdm.write("Only SN detected. SN SExtractor ID: " + str(snid))
                        tqdm.write("SN mag: " + str(cat['MAG_AUTO'][sn_idx]))
                        
                    elif snid != current_sextractor_id:
                        sn_spec_path = get_sn_spec_path(z)
                        gal_spec_path = get_gal_spec_path(z)

                        fh.write(str(snid) + " " + sn_spec_path + "\n")
                        fh.write(str(current_sextractor_id) + " " + gal_spec_path + "\n")

                        assigned_sne.append(snid)

                        tqdm.write("SN SExtractor ID: " + str(snid))
                        tqdm.write("HOST SExtractor ID: " + str(current_sextractor_id))
                        tqdm.write("SN and HOST mags respectively: " + \
                                str(cat['MAG_AUTO'][sn_idx]) + "   " + str(cat['MAG_AUTO'][i]))

                else:  # i.e., for a generic galaxy
                    spec_path = get_gal_spec_path(z)
                    fh.write(str(current_sextractor_id) + " " + spec_path + "\n")

            fh.close()

    return None

def main():
    
    gen_sed_lst()

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)


