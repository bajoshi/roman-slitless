import numpy as np
from astropy.io import fits
from scipy.interpolate import griddata
from numba import jit
import emcee

import os
import sys
import time
import datetime as dt
from multiprocessing import Pool

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
astropy_cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)

# Assign directories and custom imports
cwd = os.path.dirname(os.path.abspath(__file__))
fitting_utils = cwd + '/utils/'

roman_slitless_dir = os.path.dirname(cwd)
ext_spectra_dir = "/Volumes/Joshi_external_HDD/Roman/roman_slitless_sims_results/"
results_dir = ext_spectra_dir + 'fitting_results/'

sys.path.append(roman_slitless_dir)
sys.path.append(fitting_utils)
from get_snr import get_snr
from get_template_inputs import get_template_inputs
import dust_utils as du
from snfit_plots import read_pickle_make_plots_sn

#### ------ DONE WITH IMPORTS ------ ####
start = time.time()

# Define any required constants/arrays
sn_day_arr = np.arange(-20,51,1)

# Load in all models
# ------ THIS HAS TO BE GLOBAL!

# Read in SALT2 SN IA file from Lou
salt2_spec = np.genfromtxt(fitting_utils + "salt2_template_0.txt", \
    dtype=None, names=['day', 'lam', 'flam'], encoding='ascii')

# Also load in lookup table for luminosity distance
dl_cat = np.genfromtxt(fitting_utils + 'dl_lookup_table.txt', dtype=None, names=True)
# Get arrays 
dl_z_arr = np.asarray(dl_cat['z'], dtype=np.float64)
dl_cm_arr = np.asarray(dl_cat['dl_cm'], dtype=np.float64)

del dl_cat

print("Done loading all models. Time taken:", "{:.3f}".format(time.time()-start), "seconds.")

# Set pylinear f_lambda scaling factor
pylinear_flam_scale_fac = 1e-17

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

@jit(nopython=True)
def apply_redshift(restframe_wav, restframe_lum, redshift):

    adiff = np.abs(dl_z_arr - redshift)
    z_idx = np.argmin(adiff)
    dl = dl_cm_arr[z_idx]

    redshifted_wav = restframe_wav * (1 + redshift)
    redshifted_flux = restframe_lum / (4 * np.pi * dl * dl * (1 + redshift))

    return redshifted_wav, redshifted_flux

def loglike_sn(theta, x, data, err):
    
    z, day, av = theta

    y = model_sn(x, z, day, av)

    # ------- Vertical scaling factor
    y = get_y_alpha(y, data, err)

    lnLike = get_lnLike(y, data, err)

    #print("Chi2 term:", np.sum((y-data)**2/err**2))
    #print("Second loglikelihood term:", np.nansum( np.log(2 * np.pi * err**2)) )
    #print("ln(likelihood) SN", lnLike)

    """
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'$\lambda\, [\mathrm{\AA}]$', fontsize=14)
    ax.set_ylabel(r'$f_\lambda\, [\mathrm{cgs}]$', fontsize=14)

    ax.plot(x, data, color='k')
    ax.fill_between(x, data - err, data + err, color='gray', alpha=0.5)

    ax.plot(x, y, color='firebrick')

    plt.show()
    #sys.exit(0)
    """

    return lnLike

def logprior_sn(theta):

    z, day, av = theta

    if ( 0.0001 <= z <= 3.0  and  -19 <= day <= 50  and  0.0 <= av <= 5.0):
        return 0.0
    
    return -np.inf

def logpost_sn(theta, x, data, err):

    lp = logprior_sn(theta)
    
    if not np.isfinite(lp):
        return -np.inf
    
    lnL = loglike_sn(theta, x, data, err)
    
    return lp + lnL

#@jit(nopython=True)
# griddata is an issue for jit
# maybe a manually written 'griddate' would be okay
def model_sn(x, z, day, sn_av):

    # pull out spectrum for the chosen day
    day_idx_ = np.argmin(abs(sn_day_arr - day))
    day_idx = np.where(salt2_spec['day'] == sn_day_arr[day_idx_])[0]

    sn_spec_llam = salt2_spec['flam'][day_idx]
    sn_spec_lam = salt2_spec['lam'][day_idx]

    # ------ Apply dust extinction
    sn_dusty_llam = du.get_dust_atten_model(sn_spec_lam, sn_spec_llam, sn_av)

    # ------ Apply redshift
    sn_lam_z, sn_flam_z = apply_redshift(sn_spec_lam, sn_dusty_llam, z)

    # ------ Regrid to Roman wavelength sampling
    sn_mod = griddata(points=sn_lam_z, values=sn_flam_z, xi=x)

    return sn_mod

@jit(nopython=True)
def get_y_alpha(y, data, err):

    alpha = np.nansum(data * y / err**2) / np.nansum(y**2 / err**2)

    ya = y * alpha

    return ya

@jit(nopython=True)
def get_lnLike(y, data, err):

    lnLike = -0.5 * np.nansum( (y-data)**2/ err**2 ) 

    return lnLike

def main():

    # ----------------------- Preliminary stuff ----------------------- #
    ext_root = "romansim_prism_"

    img_basename = '5deg_'
    img_filt = 'Y106_'

    exptime1 = '_900s'
    exptime2 = '_1800s'
    exptime3 = '_3600s'

    all_exptimes = [exptime1, exptime2, exptime3]

    # ----------------------- Using emcee ----------------------- #
    # Labels for corner and trace plots
    label_list_sn = [r'$z$', r'$Day$', r'$A_V [mag]$']

    # Set jump sizes # ONLY FOR INITIAL POSITION SETUP
    jump_size_z = 0.01
    jump_size_av = 0.1  # magnitudes
    jump_size_day = 2  # days

    zprior = 0.5
    zprior_sigma = 0.02

    #get_optimal_position()
    rsn_init = np.array([zprior, 0, 0.0])  # redshift, day relative to peak, and dust extinction

    # Setup dims and walkers
    nwalkers = 300
    ndim_sn  = 3

    # generating ball of walkers about initial position defined above
    pos_sn = np.zeros(shape=(nwalkers, ndim_sn))

    for i in range(nwalkers):

        # ---------- For SN
        rsn0 = float(rsn_init[0] + jump_size_z * np.random.normal(size=1))
        rsn1 = int(rsn_init[1] + jump_size_day * np.random.normal(size=1))
        rsn2 = float(rsn_init[2] + jump_size_av * np.random.normal(size=1))

        rsn = np.array([rsn0, rsn1, rsn2])

        pos_sn[i] = rsn

    # ----------------------- Loop over all simulated and extracted SN spectra ----------------------- #
    # Arrays to loop over
    pointings = np.arange(0, 191)
    detectors = np.arange(1, 19, 1)

    for pt in pointings:
        for det in detectors:

            img_suffix = img_filt + str(pt) + '_' + str(det)

            # --------------- Read in sed.lst
            sedlst_header = ['segid', 'sed_path']
            sedlst_path = roman_slitless_dir + '/pylinear_lst_files/' + 'sed_' + img_suffix + '.lst'
            sedlst = np.genfromtxt(sedlst_path, dtype=None, names=sedlst_header, encoding='ascii')
            print("Read in sed.lst from:", sedlst_path)

            print("Number of spectra in file:", len(sedlst))

            # --------------- Read in the extracted spectra
            # For all exposure times
            ext_spec_filename1 = ext_spectra_dir + ext_root + img_suffix + exptime1 + '_x1d.fits'
            ext_hdu1 = fits.open(ext_spec_filename1)
            print("Read in extracted spectra from:", ext_spec_filename1)

            ext_spec_filename2 = ext_spectra_dir + ext_root + img_suffix + exptime2 + '_x1d.fits'
            ext_hdu2 = fits.open(ext_spec_filename2)
            print("Read in extracted spectra from:", ext_spec_filename2)

            ext_spec_filename3 = ext_spectra_dir + ext_root + img_suffix + exptime3 + '_x1d.fits'
            ext_hdu3 = fits.open(ext_spec_filename3)
            print("Read in extracted spectra from:", ext_spec_filename3)
            print('\n')

            all_hdus = [ext_hdu1, ext_hdu2, ext_hdu3]

            # --------------- Loop over all extracted files and SN in each file
            expcount = 0
            for ext_hdu in all_hdus:

                for i in range(len(sedlst)):

                    template_name = os.path.basename(sedlst['sed_path'][i])
                    if 'salt' not in template_name:
                        continue

                    # Get spectrum
                    segid = sedlst['segid'][i]

                    print("\nFitting SegID:", segid)

                    wav = ext_hdu[('SOURCE', segid)].data['wavelength']
                    flam = ext_hdu[('SOURCE', segid)].data['flam'] * pylinear_flam_scale_fac

                    # Check SNR
                    snr = get_snr(wav, flam)

                    print("SNR for this spectrum:", "{:.2f}".format(snr))

                    if snr < 3.0:
                        print("Skipping due to low SNR.")
                        continue

                    # Set noise level based on snr
                    noise_lvl = 1/snr

                    # Create ferr array
                    ferr = noise_lvl * flam

                    # Clip data at the ends
                    wav_idx = np.where((wav > 7600) & (wav < 18000))[0]

                    sn_wav = wav[wav_idx]
                    sn_flam = flam[wav_idx]
                    sn_ferr = ferr[wav_idx]

                    # Set up args
                    args_sn = [sn_wav, sn_flam, sn_ferr]

                    print("logpost at starting position for SN:")
                    print(logpost_sn(rsn_init, sn_wav, sn_flam, sn_ferr))

                    # Now run on SN
                    snstr = str(segid) + '_' + img_suffix + all_exptimes[expcount]
                    emcee_savefile = results_dir + \
                                     'emcee_sampler_sn' + snstr + '.h5'
                    if not os.path.isfile(emcee_savefile):
                        backend = emcee.backends.HDFBackend(emcee_savefile)
                        backend.reset(nwalkers, ndim_sn)
                        
                        with Pool(2) as pool:
                            sampler = emcee.EnsembleSampler(nwalkers, ndim_sn, logpost_sn,
                                args=args_sn, pool=pool, backend=backend,
                                moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2),])
                            sampler.run_mcmc(pos_sn, 2000, progress=True)

                        print(f"{bcolors.GREEN}")
                        print("Finished running emcee.")
                        print("Mean acceptance Fraction:", np.mean(sampler.acceptance_fraction), "\n")
                        print(f"{bcolors.ENDC}")

                        # ---------- Stuff needed for plotting
                        template_inputs = get_template_inputs(template_name)
                        truth_dict = {}
                        truth_dict['z']     = template_inputs[0]
                        truth_dict['phase'] = template_inputs[1]
                        truth_dict['Av']    = template_inputs[2]

                        read_pickle_make_plots_sn('sn' + snstr, 
                            ndim_sn, args_sn, label_list_sn, truth_dict, results_dir)

                        print("Finished plotting results.")

                expcount += 1

            # --------------- close all open fits files
            ext_hdu1.close()
            ext_hdu2.close()
            ext_hdu3.close()

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)