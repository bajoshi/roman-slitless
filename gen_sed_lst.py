import numpy as np
from numpy.random import default_rng
from astropy.io import fits

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

import os
import sys
import subprocess
from tqdm import tqdm
import socket
import pickle

import matplotlib.pyplot as plt

# Define constants
Lsol = 3.826e33
# Define scaling factor
# Check sn_scaling.py in same folder as this code
sn_scalefac = 1.734e40

astropy_cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc,
                              Tcmb0=2.725 * u.K, Om0=0.3)

# -----------------
if 'plffsn2' in socket.gethostname():
    extdir = '/astro/ffsn/Joshi/'
    modeldir = extdir + 'bc03_output_dir/'

    roman_sims_seds = extdir + 'roman_slitless_sims_seds/'
    pylinear_lst_dir = extdir + 'pylinear_lst_files/'
    roman_direct_dir = extdir + 'roman_direct_sims/sims2021/'

    roman_slitless_dir = extdir + "GitHub/roman-slitless/"
    fitting_utils = roman_slitless_dir + "fitting_pipeline/utils/"

    pickles_path = extdir + 'Pickles_stellar_library/for_romansim/'

else:
    extdir = '/Volumes/Joshi_external_HDD/Roman/'
    modeldir = extdir + 'bc03_output_dir/m62/'

    roman_sims_seds = extdir + "roman_slitless_sims_seds/"
    pylinear_lst_dir = extdir + "pylinear_lst_files/"
    roman_direct_dir = extdir + 'roman_direct_sims/sims2021/'

    home = os.getenv("HOME")
    roman_slitless_dir = home + "/Documents/GitHub/roman-slitless/"
    fitting_utils = roman_slitless_dir + "fitting_pipeline/utils/"

    pickles_path = '/Volumes/Joshi_external_HDD/' + \
        'Pickles_stellar_library/for_romansim/'

assert os.path.isdir(modeldir)
assert os.path.isdir(roman_sims_seds)
assert os.path.isdir(pylinear_lst_dir)
assert os.path.isdir(roman_direct_dir)

sys.path.append(fitting_utils)
import dust_utils as du  # noqa: E402
from get_obj_pix import get_obj_pix  # noqa: E402

# Read in SALT2 SN IA file  from Lou
salt2_spec = np.genfromtxt(fitting_utils + "templates/salt2_template_0.txt",
                           dtype=None, names=['day', 'lam', 'llam'],
                           encoding='ascii')

model_lam = np.load(extdir + "bc03_output_dir/bc03_models_wavelengths.npy",
                    mmap_mode='r')
model_ages = np.load(extdir + "bc03_output_dir/bc03_models_ages.npy",
                     mmap_mode='r')

all_m62_models = []
tau_low = 0
tau_high = 20
for t in range(tau_low, tau_high, 1):
    tau_str = "{:.3f}".format(t).replace('.', 'p')
    a = np.load(modeldir + 'bc03_all_tau' + tau_str + '_m62_chab.npy',
                mmap_mode='r')
    all_m62_models.append(a)
    del a

# load models with large tau separately
all_m62_models.append(np.load(modeldir + 'bc03_all_tau20p000_m62_chab.npy',
                              mmap_mode='r'))

# Also load in lookup table for luminosity distance
dl_cat = np.genfromtxt(fitting_utils + 'dl_lookup_table.txt',
                       dtype=None, names=True, encoding='ascii')

# Get arrays
dl_z_arr = np.asarray(dl_cat['z'], dtype=np.float64)
dl_cm_arr = np.asarray(dl_cat['dl_cm'], dtype=np.float64)

del dl_cat

# And load table for SN Ia mF106 to z conversion
sn_mag_z = np.genfromtxt(fitting_utils + 'sn_mag_z_lookup.txt',
                         dtype=None, names=True, encoding='ascii')


class bcolors:
    # This class came from stackoverflow
    # SEE:
    # https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-python
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


def get_sn_spec_path(redshift, day_chosen=-99, chosen_av=None):
    """
    This function will assign a random spectrum from
    the basic SALT2 spectrum form Lou.
    Equal probability is given to any day relative
    to maximum. This will change for the final version.

    The spectrum file contains a type 1A spectrum from
    -20 to +50 days relative to max. Since the -20 spectrum
    is essentially empty, I won't choose that spectrum.
    """

    # choose a random day relative to max
    if day_chosen == -99:
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
        # see:
        # https://numpy.org/doc/stable/reference/random/generated/numpy.random.exponential.html
        # higher beta values give "flatter" exponentials
        # I want a fairly steep exponential decline toward high Av values
        if chosen_av > 3.0:
            chosen_av = 3.0

    sn_dusty_llam = du.get_dust_atten_model(sn_spec_lam, sn_spec_llam,
                                            chosen_av)

    # Apply redshift
    sn_wav_z, sn_flux = apply_redshift(sn_spec_lam, sn_dusty_llam, redshift)

    # Save individual spectrum file if it doesn't already exist
    sn_spec_path = roman_sims_seds \
        + "salt2_spec_day" + str(day_chosen) \
        + "_z" + "{:.4f}".format(redshift).replace('.', 'p') \
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

    return sn_spec_path


def get_bc03_spec(age, logtau):

    tau = 10**logtau  # logtau is log of tau in gyr

    if tau < 20.0:

        tau_int_idx = int((tau - int(np.floor(tau))) * 1e3)
        age_idx = np.argmin(abs(model_ages - age * 1e9))
        model_idx = tau_int_idx * len(model_ages) + age_idx

        models_taurange_idx = np.argmin(abs(np.arange(tau_low, tau_high, 1)
                                            - int(np.floor(tau))))
        models_arr = all_m62_models[models_taurange_idx]

    elif tau >= 20.0:

        logtau_arr = np.arange(1.30, 2.01, 0.01)
        logtau_idx = np.argmin(abs(logtau_arr - logtau))

        age_idx = np.argmin(abs(model_ages - age * 1e9))
        model_idx = logtau_idx * len(model_ages) + age_idx

        models_arr = all_m62_models[-1]

    model_llam = models_arr[model_idx]

    return np.array(model_llam, dtype=np.float64)


def get_gal_spec_path(redshift, log_stellar_mass_chosen=None,
                      chosen_age=None,
                      chosen_tau=None, chosen_av=None):
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
    if not log_stellar_mass_chosen:
        log_stellar_mass_arr = np.linspace(10.0, 11.5, 100)
        log_stellar_mass_chosen = np.random.choice(log_stellar_mass_arr)

    log_stellar_mass_str = \
        "{:.2f}".format(log_stellar_mass_chosen).replace('.', 'p')

    # --------- Age
    if not chosen_age:
        age_arr = np.arange(0.1, 13.0, 0.005)  # in Gyr

        # Now choose age consistent with given redshift
        # i.e., make sure model is not older than the Universe
        # Allowing at least 100 Myr for the first
        # galaxies to form after Big Bang
        age_at_z = astropy_cosmo.age(redshift).value  # in Gyr
        age_lim = age_at_z - 0.1  # in Gyr

        chosen_age = np.random.choice(age_arr)
        while chosen_age > age_lim:
            chosen_age = np.random.choice(age_arr)

    # --------- SFH
    # Choose SFH form from a few different models
    # and then choose params for the chosen SFH form
    # sfh_forms = \
    #     ['instantaneous', 'constant', 'exponential', 'linearly_declining']

    # choose_sfh(sfh_forms[2])

    if not chosen_tau:
        tau_arr = np.arange(0.01, 15.0, 0.005)  # in Gyr
        chosen_tau = np.random.choice(tau_arr)

    # --------- Metallicity
    # metals_arr = np.array([0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05])
    # While the newer 2016 version has an additional metallicity
    # referred to as "m82", the documentation never specifies the
    # actual metallicity associated with it. So I'm ignoring that one.
    metals = 0.02  # np.random.choice(metals_arr)

    # Get hte metallicity string
    # if metals == 0.0001:
    #     metallicity = 'm22'
    # elif metals == 0.0004:
    #     metallicity = 'm32'
    # elif metals == 0.004:
    #     metallicity = 'm42'
    # elif metals == 0.008:
    #     metallicity = 'm52'
    # elif metals == 0.02:
    #     metallicity = 'm62'
    # elif metals == 0.05:
    #     metallicity = 'm72'

    # ----------- CALL BC03 -----------
    # The BC03 generated spectra will always be at redshift=0 and dust free.
    # This code will apply dust extinction and redshift effects manually
    # outdir = home + '/Documents/bc03_output_dir/'
    # gen_bc03_spectrum(chosen_tau, metals, outdir)
    logtau = np.log10(chosen_tau)
    bc03_spec_wav = np.array(model_lam, dtype=np.float64)
    bc03_spec_llam = get_bc03_spec(chosen_age, logtau)

    # Apply Calzetti dust extinction depending on av value chosen
    if not chosen_av:
        av_arr = np.arange(0.0, 5.0, 0.001)  # in mags
        chosen_av = np.random.choice(av_arr)

    bc03_dusty_llam = du.get_dust_atten_model(bc03_spec_wav, bc03_spec_llam,
                                              chosen_av)

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
    # if apply_igm:
    #     pass

    bc03_wav_z, bc03_flux = apply_redshift(bc03_spec_wav, bc03_dusty_llam,
                                           redshift)

    if plot_tocheck:

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.set_xlabel(r'$\lambda\ \mathrm{[\AA]}$', fontsize=14)
        ax.set_ylabel(r'$f_\lambda\ \mathrm{[erg\, s^{-1}\, cm^{-2}\, \AA]}$',
                      fontsize=14)

        # ax.plot(bc03_spec_wav, bc03_spec_llam, label='Orig model')
        # ax.plot(bc03_spec_wav, bc03_dusty_llam, label='Dusty model')
        ax.plot(bc03_wav_z, bc03_flux,
                label='Redshfited dusty model with chosen Ms')

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

            fh_gal.write("{:.2f}".format(bc03_wav_z[j]) + " "
                         + str(bc03_flux[j]))
            fh_gal.write("\n")

        fh_gal.close()

    return gal_spec_path


def get_match(ra_arr, dec_arr, ra_to_check, dec_to_check, tol_arcsec=0.3):

    # Matching tolerance
    tol = tol_arcsec / 3600
    # arcseconds expressed in degrees since our ra-decs are in degrees

    radiff = ra_arr - ra_to_check
    decdiff = dec_arr - dec_to_check
    idx = np.where((np.abs(radiff) < tol) & (np.abs(decdiff) < tol))[0]

    # Find closest match in case of multiple matches
    if len(idx) > 1:

        ra_two = ra_to_check
        dec_two = dec_to_check

        dist_list = []
        for v in range(len(idx)):

            ra_one = ra_arr[idx][v]
            dec_one = ra_arr[idx][v]

            dist = np.arccos(np.cos(dec_one * np.pi / 180)
                             * np.cos(dec_two * np.pi / 180)
                             * np.cos(ra_one * np.pi / 180
                                      - ra_two * np.pi / 180)
                             + np.sin(dec_one * np.pi / 180)
                             * np.sin(dec_two * np.pi / 180))
            dist_list.append(dist)

        dist_list = np.asarray(dist_list)
        idx = np.argmin(dist_list)

    elif len(idx) == 1:
        idx = int(idx)

    # Increase tolerance if no match found at first
    elif len(idx) == 0:
        idx = -99

    return idx


def get_sn_z(snmag):

    # This function assumes that the utility code kcorr.py
    # has been run on its own. kcorr.py will do a couple
    # tests and print out two cols to the terminal --
    # redshift and mF106 which are used here.

    mag_arr = sn_mag_z['mF106']
    z_idx = np.argmin(abs(snmag - mag_arr))

    sn_z = sn_mag_z['Redshift'][z_idx]

    return sn_z


def gen_sed_lst_with_truth():

    print(f"{bcolors.WARNING}")
    print("TODO: ")
    print("1. Make this code run on multiple cores.")
    print("2. Use the improved algorithm in the survey gen_sed_lst.")
    print("3. Ensure that SN and host get unique SegIDs.")
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
    pointings = np.arange(0, 1)
    detectors = np.arange(1, 2, 1)

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
            tqdm.write(f"{bcolors.CYAN}" + "\nWill generate SED file: "
                       + sed_filename + f"{bcolors.ENDC}")

            fh = open(sed_filename, 'w')

            # Write header
            fh.write("# 1: SEGMENTATION ID" + "\n")
            fh.write("# 2: SED FILE" + "\n")

            # Read in catalog from SExtractor
            cat_filename = img_sim_dir + img_basename + img_filt + \
                str(pt) + '_' + str(det) + '_SNadded.cat'
            tqdm.write("Read catalog: " + cat_filename)
            tqdm.write("Checking for catalog: " + cat_filename)

            if not os.path.isfile(cat_filename):
                tqdm.write("Cannot find object catalog."
                           + "SExtractor will be run automatically.")

                # Now run SExtractor automatically
                # Set up sextractor
                img_filename = img_basename + img_filt + \
                    str(pt) + '_' + str(det) + '_SNadded.fits'
                checkimage = img_basename + img_filt + \
                    str(pt) + '_' + str(det) + '_segmap.fits'

                # Change directory to images directory
                os.chdir(img_sim_dir)

                tqdm.write(f"{bcolors.GREEN}" + "Running: " + "sex "
                           + img_filename + " -c"
                           + " roman_sims_sextractor_config.txt"
                           + " -CATALOG_NAME " + os.path.basename(cat_filename)
                           + " -CHECKIMAGE_NAME " + checkimage
                           + f"{bcolors.ENDC}")

                # Use subprocess to call sextractor.
                # The args passed MUST be passed in this way.
                # i.e., args that would be separated by a space on the
                # command line must be passed here separated by commas.
                # It will not work if you join all of these args in a
                # string with spaces where they are supposed to be;
                # even if the command looks right when printed out.
                subprocess.run(['sex', img_filename,
                                '-c', 'roman_sims_sextractor_config.txt',
                                '-CATALOG_NAME',
                                os.path.basename(cat_filename),
                                '-CHECKIMAGE_NAME', checkimage], check=True)

                tqdm.write("Finished SExtractor run."
                           + "Check cat and segmap if needed.")

                # Go back to roman-slitless directory
                os.chdir(roman_slitless_dir)

            cat_header = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'ALPHA_J2000',
                          'DELTA_J2000', 'FLUX_AUTO', 'FLUXERR_AUTO',
                          'MAG_AUTO', 'MAGERR_AUTO', 'FLUX_RADIUS',
                          'FWHM_IMAGE', 'CLASS_STAR']
            cat = np.genfromtxt(cat_filename, dtype=None, names=cat_header,
                                encoding='ascii')
            tqdm.write(f"{bcolors.GREEN}" + str(len(cat))
                       + " objects in catalog." + f"{bcolors.ENDC}")

            # Loop over all objects and assign spectra
            # Read in the truth files first
            truth_hdu_gal = fits.open(truth_dir + truth_basename
                                      + img_filt + str(pt) + '_'
                                      + str(det) + '.fits')
            truth_hdu_sn = fits.open(truth_dir + truth_basename
                                     + img_filt + str(pt) + '_'
                                     + str(det) + '_sn.fits')

            hostids = truth_hdu_sn[1].data['hostid']

            # assign arrays
            ra_gal = truth_hdu_gal[1].data['ra'] * 180 / np.pi
            dec_gal = truth_hdu_gal[1].data['dec'] * 180 / np.pi

            # Also assign SN spectra to our added SN
            snadd_cat = np.load(cat_filename.replace('.cat', '.npy'))
            xi = snadd_cat[:, 0]
            yi = snadd_cat[:, 1]

            """
            # -----------
            ***** Short explanation of the code flow below. *****
            # -----------

            GOAL: Every object in the SExtractor catalog
                  must be assigned a spectrum.

            Steps:
              1. Check for match with truth galaxy catalog.

              2. IF MATCH IS NOT FOUND:

                2A: Pick a random z; separately for galaxy and SN

                2B: Is it one of the fake SNe that we added?
                    Yes --> Assign SN spectrum
                    No  --> Assign GALAXY spectrum

              3. IF MATCH IS FOUND:

                3A: Get truth-z

                3B: Is it a host galaxy?
                    Yes --> Find corresponding SN and assign SN and host galaxy
                            spectrum to the same redshift.
                        --> Ensure that when the SN ID is encountered in
                            the loop it is skipped.
                        --> Must also ensure that the SN ID wasn't previously
                            assigned a galaxy spectrum when
                            a match wasn't found.
                    No  --> Assign galaxy spectrum
            """

            assigned_sne = []
            assigned_gal = []
            assigned_z = []

            for i in tqdm(range(len(cat)), desc="Object SegID", leave=False):

                # -------------- First match object -------------- #
                current_sextractor_id = int(cat['NUMBER'][i])

                if current_sextractor_id in assigned_sne:
                    tqdm.write("\nSN spectrum already assigned to "
                               + str(current_sextractor_id))
                    tqdm.write("Skipping.")
                    continue

                # The -1 in the index is needed because the SExtractor
                # catalog starts counting object ids from 1.
                ra_to_check = cat['ALPHA_J2000'][current_sextractor_id - 1]
                dec_to_check = cat['DELTA_J2000'][current_sextractor_id - 1]

                # Now match and get corresponding entry in the
                # larger truth file
                idx = get_match(ra_gal, dec_gal, ra_to_check, dec_to_check)
                # tqdm.write("Matched idx: " + str(idx))

                if idx == -99:
                    tqdm.write("\nObjID:" + str(current_sextractor_id))
                    tqdm.write("No matches found in truth file.")

                    z_nomatch_gal = np.random.uniform(low=0.0, high=3.0)

                    # There are some galaxies that have no matches in the truth
                    # files. I'm assigning a random redshift and
                    # spectrum to them.
                    # They need to be given a spectrum otherwise the extraction
                    # is likely to be messed up since we'd then have
                    # objects that should have had dispersed light
                    # on the detector but didn't.
                    # Not sure what that would do to the extraction.
                    # ie. can't skip them like before.
                    # The SNe added through insert_sne.py should
                    # however be given SNe spectra.
                    current_x = cat['X_IMAGE'][i]
                    current_y = cat['Y_IMAGE'][i]
                    added_match = np.where((np.abs(xi - current_x) <= 3.0)
                                           & (np.abs(yi - current_y)
                                              <= 3.0))[0]
                    if len(added_match) < 1:
                        tqdm.write('Assigning galaxy spectrum to object'
                                   + ' with no match in truth')
                        tqdm.write('and is not an object added through '
                                   + 'insert_sne.py')
                        spec_path = get_gal_spec_path(z_nomatch_gal)
                        fh.write(str(current_sextractor_id) + " "
                                 + spec_path + "\n")

                        assigned_gal.append(current_sextractor_id)
                        assigned_z.append(z_nomatch_gal)
                        continue

                    else:
                        tqdm.write(f'{bcolors.CYAN}'
                                   + 'Assigning random redshift'
                                   + 'to inserted SN.'
                                   + f'{bcolors.ENDC}')
                        # SN z must be consistent with cosmological dimming
                        z_nomatch_sn = get_sn_z(cat['MAG_AUTO'][i])
                        sn_spec_path = get_sn_spec_path(z_nomatch_sn)
                        fh.write(str(current_sextractor_id) + " "
                                 + sn_spec_path + "\n")

                        assigned_sne.append(current_sextractor_id)
                        assigned_z.append(z_nomatch_sn)
                        continue

                id_fetch = int(truth_hdu_gal[1].data['ind'][idx])

                truth_idx = np.where(truth_match[1].data['gind']
                                     == id_fetch)[0]
                # -------------- Matching done -------------- #

                # -------------- Get object redshift
                z = float(truth_match[1].data['z'][truth_idx])

                # -------------- Check if it is a SN host galaxy or SN itself
                # If it is then also call the sn SED path generation
                if id_fetch in hostids:
                    # Now you must find the corresponding
                    # SExtractor ID for the SN

                    sn_idx0 = np.where(hostids == id_fetch)[0]

                    sn_ra = truth_hdu_sn[1].data['ra'][sn_idx0] * 180 / np.pi
                    sn_dec = truth_hdu_sn[1].data['dec'][sn_idx0] * 180 / np.pi

                    sn_idx = get_match(cat['ALPHA_J2000'], cat['DELTA_J2000'],
                                       sn_ra, sn_dec)
                    if sn_idx == -99:
                        tqdm.write(f"{bcolors.FAIL}")
                        tqdm.write("Matching SN not found for hostid "
                                   + str(id_fetch))
                        tqdm.write("Assigning GALAXY spectrum.")
                        tqdm.write(f"{bcolors.ENDC}")
                        spec_path = get_gal_spec_path(z)
                        fh.write(str(current_sextractor_id) + " "
                                 + spec_path + "\n")

                        assigned_gal.append(current_sextractor_id)
                        assigned_z.append(z)
                        continue

                    snid = cat['NUMBER'][sn_idx]

                    # This means that the SN and host matched to
                    # the same location
                    # i.e., the SN is bright enough that it outshines the host
                    z_sn = get_sn_z(cat['MAG_AUTO'][i])
                    if snid == current_sextractor_id:
                        sn_spec_path = get_sn_spec_path(z_sn)
                        fh.write(str(snid) + " " + sn_spec_path + "\n")
                        tqdm.write("Only SN detected. SN SExtractor ID: "
                                   + str(snid))
                        tqdm.write("SN mag: " + str(cat['MAG_AUTO'][sn_idx]))

                        assigned_sne.append(current_sextractor_id)
                        assigned_z.append(z_sn)

                    elif snid != current_sextractor_id:
                        sn_spec_path = get_sn_spec_path(z_sn)
                        gal_spec_path = get_gal_spec_path(z_sn)

                        fh.write(str(snid) + " " + sn_spec_path + "\n")
                        fh.write(str(current_sextractor_id) + " "
                                 + gal_spec_path + "\n")

                        assigned_sne.append(snid)
                        assigned_gal.append(current_sextractor_id)
                        assigned_z.append(z_sn)

                        tqdm.write("SN SExtractor ID: " + str(snid))
                        tqdm.write("HOST SExtractor ID: "
                                   + str(current_sextractor_id))
                        tqdm.write("SN and HOST mags respectively: "
                                   + str(cat['MAG_AUTO'][sn_idx]) + "   "
                                   + str(cat['MAG_AUTO'][i]))

                else:  # i.e., for a generic galaxy
                    spec_path = get_gal_spec_path(z)
                    fh.write(str(current_sextractor_id) + " "
                             + spec_path + "\n")

                    assigned_gal.append(current_sextractor_id)
                    assigned_z.append(z)

            fh.close()

            # ----- Assert that each object got a spectrum
            sedlst = np.genfromtxt(sed_filename, dtype=None,
                                   names=['SegID', 'sed_path'],
                                   encoding='ascii', skip_header=2)
            try:
                assert len(cat) == len(sedlst)
            except AssertionError:
                tqdm.write(f'{bcolors.FAIL}')

                tqdm.write('Lengths of catalog and SED lst not consistent.')
                tqdm.write('Need to manually remove one of the '
                           + 'following repeated SegIDs:')

                assigned_gal = np.asarray(assigned_gal)
                assigned_sne = np.asarray(assigned_sne)
                print(np.intersect1d(assigned_sne, assigned_gal))

                tqdm.write(f'{bcolors.ENDC}')

    return None


def remove_duplicates():

    img_filt = 'Y106_'

    # Arrays to loop over
    pointings = np.arange(0, 1)
    detectors = np.arange(1, 19, 1)

    for pt in pointings:
        for det in tqdm(detectors, desc="Removing duplicates"):

            sed_filename = (pylinear_lst_dir
                            + 'sed_' + img_filt + str(pt) + '_'
                            + str(det) + '.lst')

            # Get all lines
            alllines = open(sed_filename, 'r').readlines()

            # First check for duplicates and then rewrite the
            # file without duplicates if there are any.

            # Gather all IDs and check for uniques
            all_ids = []
            all_spectra = []
            linecount = 0
            for line in alllines:
                if linecount > 1:  # skip first two lines of header
                    lsp = line.split()
                    all_ids.append(int(lsp[0]))
                    all_spectra.append(lsp[1])
                linecount += 1

            all_ids = np.array(all_ids)
            all_spectra = np.array(all_spectra)

            all_unique_ids, counts = np.unique(all_ids, return_counts=True)

            # Now if there are more total IDS than unique IDs
            # then we have duplicates. Otherwise skip to hte next file.
            if len(all_unique_ids) != len(all_ids):

                assert len(all_unique_ids) < len(all_ids)

                duplicate_idx = np.where(counts > 1)[0]
                # This could be either the SN or the galaxy
                # spectrum that numpy counted (with counts>1)
                # i.e, this corresponds to the index of
                # the second appearance of the ID
                # Therefore, we'll look for both IDs that match and
                # then decide to keep the SN spectrum
                assert len(duplicate_idx) == 1
                # typically only one duplicate # we could turn
                # code below into a for loop if needed

                duplicate_id_idx = np.where(all_ids
                                            == all_ids[duplicate_idx[0]])[0]

                # Find duplicate spectra and choose the SN spectra
                # duplicate_IDs = all_ids[duplicate_id_idx]
                duplicate_spectra = all_spectra[duplicate_id_idx]
                for s, spec in enumerate(duplicate_spectra):
                    if 'bc03' in spec:
                        idx_to_delete = duplicate_id_idx[s]
                        break

                ids_to_write = np.delete(all_ids, idx_to_delete)
                spectra_to_write = np.delete(all_spectra, idx_to_delete)

                with open(sed_filename, 'w') as fh:
                    fh.write(alllines[0])
                    fh.write(alllines[1])

                    for i in range(len(ids_to_write)):
                        fh.write(str(ids_to_write[i]) + ' '
                                 + spectra_to_write[i] + '\n')

            else:
                continue

    return None


def add_faint_sne_sedlst():

    # Set image and truth params
    dir_img_part = 'part1'

    img_sim_dir = roman_direct_dir + 'K_5degimages_' + dir_img_part + '/'
    img_basename = '5deg_'
    img_filt = 'Y106_'

    faint_mag_lim = 24.5

    # Arrays to loop over
    pointings = np.arange(0, 1)
    detectors = np.arange(1, 19, 1)

    for pt in pointings:
        for det in tqdm(detectors, desc="Adding faint SNe manually"):

            # Read in segmentation map
            segmap = img_sim_dir + img_basename + img_filt + \
                str(pt) + '_' + str(det) + '_segmap.fits'
            segdata, seghdr = fits.getdata(segmap, header=True)

            # Name of direct image
            dir_img_name = segmap.replace('_segmap.fits', '_SNadded.fits')

            # Name of catalog
            cat_filename = img_sim_dir + img_basename + img_filt + \
                str(pt) + '_' + str(det) + '_SNadded.cat'

            # Read in current sedlst
            sed_filename = (pylinear_lst_dir
                            + 'sed_' + img_filt + str(pt) + '_'
                            + str(det) + '.lst')
            sedlst = np.genfromtxt(sed_filename, dtype=None,
                                   names=['SegID', 'sed_path'],
                                   encoding='ascii', skip_header=2)

            # Read in list of artificially inserted SNe
            snadd_cat = np.load(segmap.replace('_segmap.fits', '_SNadded.npy'))
            xi = snadd_cat[:, 0]
            yi = snadd_cat[:, 1]
            ins_mag = snadd_cat[:, 2]

            # Find all SNe fainter than 24.5
            # Loop over all faint SNe and manually add them
            # 1. Get the x and y pos of the inserted SN
            # 2. Within the segmap, now add a Gaussian
            # at the position whose pix sum up to the
            # required flux. I'm ignoring that the other SNe
            # added in have a different "PSF" but this
            # should be okay for now.
            # 3. Give this new segmap object a new ID and
            # also assign a SN spectrum to it with a redshift
            # that is consistent with the inserted mag.
            faint_mag_idx = np.where(ins_mag >= faint_mag_lim)[0]

            max_id_in_sedlst = np.max(sedlst['SegID'])

            # Now loop
            for i in range(len(faint_mag_idx)):
                # Put the object in the SED LST
                faint_mag = ins_mag[faint_mag_idx][i]
                faint_z = get_sn_z(faint_mag)

                new_spectrum = get_sn_spec_path(faint_z)
                new_id = max_id_in_sedlst + i + 1

                with open(sed_filename, 'a') as fh:
                    fh.write(str(int(new_id)) + ' ' + new_spectrum + '\n')

                # Now put the object in the segmap
                # Get the x and y position first
                xpos = xi[faint_mag_idx][i]
                ypos = yi[faint_mag_idx][i]

                # print(faint_mag, faint_z, new_id,
                #       os.path.basename(new_spectrum), xpos, ypos)

                # Get all pix to associate with the SN
                faint_sn_pix = get_obj_pix(xpos, ypos, dir_img_name)

                # Add it to the segmap
                segdata[faint_sn_pix[0], faint_sn_pix[1]] = new_id

                # Get obj counts
                # This is inferred from flam and NOT summed
                # from the direct image because the direct image is
                # too shallow. If you try to sum the counts in the
                # direct image then it'll just be summing background.
                faint_sn_counts = 10**(-0.4 * (faint_mag - 26.264))

                # Also add to catalog
                with open(cat_filename, 'a') as fc:
                    fc.write('      ' + str(new_id)
                             + '   ' + '{:.4f}'.format(xpos)
                             + '   ' + '{:.4f}'.format(ypos)
                             + '   ' + str(-99.999999)
                             + '   ' + str(-99.999999)
                             + '   ' + str(faint_sn_counts)
                             + '   ' + str(-99.9999)
                             + '   ' + '{:.4f}'.format(faint_mag)
                             + '   ' + str(-99.9999)
                             + '   ' + str(-99.9999)
                             + '   ' + str(-99.9999) + '\n'
                             )

            # Save new segmap
            phdu = fits.PrimaryHDU(header=seghdr, data=segdata)
            phdu.writeto(segmap, overwrite=True)

    return None


def get_stellar_spec_path():
    # Randomly choses a stellar spectrum
    all_stars = ['a5v', 'b5iii', 'f0v', 'g2v', 'k3i', 'm4v', 'o5v']
    star_chosen = np.random.choice(all_stars)

    pickles_spec_path = pickles_path + 'uk' + star_chosen + '.dat'

    # Rewrite the SED to the pylinear SED dir.
    # The pickles spectra have a wav col and 4 other cols.
    # We need the first col (normalized flux) and wav.
    # Also truncate the spectra to somewhat closer to prism
    # coverage.
    star_spec_path = roman_sims_seds + star_chosen + '_pickles.txt'

    if not os.path.isfile(star_spec_path):

        # First read in the spectrum from pickles
        stellar_spec = np.genfromtxt(pickles_spec_path, dtype=None,
                                     names=['wav', 'flux'], usecols=(0, 1))

        # Now truncate
        stellar_wav = stellar_spec['wav']
        stellar_flux = stellar_spec['flux']

        wav_idx = np.where((stellar_wav >= 7000) & (stellar_wav <= 19000))[0]

        stellar_wav = stellar_wav[wav_idx]
        stellar_flux = stellar_flux[wav_idx]

        # Write to file
        with open(star_spec_path, 'w') as fh:

            for i in range(len(stellar_wav)):
                fh.write("{:.2f}".format(stellar_wav[i])
                         + " " + str(stellar_flux[i]))
                fh.write("\n")

    return star_spec_path


def gen_sed_lst():

    # Set image and truth params
    dir_img_part = 'part1'

    img_sim_dir = roman_direct_dir + 'K_5degimages_' + dir_img_part + '/'
    img_basename = '5deg_'
    img_filt = 'Y106_'

    # Arrays to loop over
    pointings = np.arange(0, 1)
    detectors = np.arange(1, 2, 1)

    for pt in pointings:
        for det in tqdm(detectors, desc="Detector", leave=False):

            img_suffix = img_filt + str(pt) + '_' + str(det)
            dir_img_path = img_sim_dir + \
                img_basename + img_suffix + '_SNadded.fits'

            sedlst_filename = pylinear_lst_dir + 'sed_' + img_suffix + '.lst'

            # Open an empty file for writing sed lst
            with open(sedlst_filename, 'w') as fh:

                # ------------ Write header
                fh.write("# 1: SEGMENTATION ID" + "\n")
                fh.write("# 2: SED FILE" + "\n")

                # ------------
                # Read in the Segmentation map and
                # get the total number of objects
                segmap = dir_img_path.replace('.fits', '_segmap.fits')
                segdata = fits.getdata(segmap)

                total_objects = np.max(segdata)

                # ------------
                # Also assign SN spectra to our added SN
                insert_cat = np.load(dir_img_path.replace('.fits', '.npy'))
                insert_segid = np.array(insert_cat[:, -1], dtype=np.int64)
                host_segids = np.array(insert_cat[:, 4])
                host_segids = host_segids.astype(np.float64)
                host_segids = host_segids.astype(np.int64)

                # Also load in the host and SN segpix arrays
                host_segpix_fl = dir_img_path.replace('.fits', 
                                                      '_hostsegpix.pkl')
                sn_segpix_fl = dir_img_path.replace('.fits', '_snsegpix.pkl')
                with open(host_segpix_fl, 'rb') as fh_host:
                    host_segpix_arr = pickle.load(fh_host)
                with open(sn_segpix_fl, 'rb') as fh_sn:
                    sn_segpix_arr = pickle.load(fh_sn)

                # Keep track of assigned redshifts
                all_redshifts = np.zeros(total_objects)

                # ------------ Now loop over all objects
                for i in tqdm(range(total_objects), desc="Object SegID"):

                    current_segid = i + 1

                    # ------------ First get hte type of the object
                    if current_segid in insert_segid:
                        obj_idx = int(np.where(insert_segid
                                               == current_segid)[0])
                        object_type = insert_cat[:, -2][obj_idx]
                    else:
                        object_type = 'GLXY'

                    # ------------ Now assign the spectrum
                    # depending on the type
                    if object_type == 'GLXY':

                        # You also need to know if it is a host-galaxy
                        # because if it is then we need to ensure
                        # that the SN that it hosts follows the
                        # correct cosmology.
                        if current_segid in host_segids:
                            insert_sn_idx = \
                                np.where(host_segids == current_segid)[0]
                            snmag = float(insert_cat[insert_sn_idx, 2])
                            z = get_sn_z(snmag)
                        else:
                            z = np.random.uniform(low=0.2, high=3.0)

                        spec_path = get_gal_spec_path(z)
                        # Append redshift
                        all_redshifts[i] = z

                    elif object_type == 'SNIa':
                        # Check the host id and get its redshift
                        current_host_segid = host_segids[obj_idx]
                        host_idx = int(current_host_segid - 1)
                        host_z = all_redshifts[host_idx]

                        # Now compute overlap between host-galaxy and SN pixels
                        host_segpix = host_segpix_arr[obj_idx]
                        sn_segpix = sn_segpix_arr[obj_idx]

                        overlap, overlap_pix = isoverlapping(sn_segpix, 
                                                             host_segpix)

                        sys.exit(0)

                        if overlap:
                            # i.e., generate a SN spectrum
                            # contaminated by host light
                            spec_path = \
                                get_sn_spec_path_hostoverlap(host_z, 
                                                             overlap_pix)
                        else:
                            # i.e., generate a pure SN spectrum
                            spec_path = get_sn_spec_path(host_z)

                        # Append redshift
                        all_redshifts[i] = host_z

                    elif object_type == 'STAR':
                        spec_path = get_stellar_spec_path()

                    # ------------ Write to file
                    fh.write(str(current_segid) + " " + spec_path + "\n")

    return None


def isoverlapping(sn_segpix, host_segpix):

    # The shape for these segpix arrays should 
    # always be (2, n) where n is the number of
    # pixels in the object.

    assert sn_segpix.ndim == 2
    assert host_segpix.ndim == 2

    print(sn_segpix.shape)
    print(host_segpix.shape)

    # There is probably a faster numpy way to do this
    # but this is easiest for now. I'm going to check each
    # (row, col) coordinate tuple in the SN segpix against 
    # the host segpix to determine pixel overlap.
    for i in range(sn_segpix.shape[1]):

        current_sn_segpix = sn_segpix[:, i]

        print(current_sn_segpix)

        if (current_sn_segpix[0], current_sn_segpix[1]) in all_host_coords:
            overlap = True

    return overlap


if __name__ == '__main__':

    gen_sed_lst()
    # remove_duplicates()
    # add_faint_sne_sedlst()

    sys.exit(0)
