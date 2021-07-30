import numpy as np
from astropy.io import fits
from scipy.interpolate import griddata
from numba import jit
import emcee
from astropy.convolution import convolve, Box1DKernel

import os
import sys
import time
import datetime as dt
from multiprocessing import Pool

import matplotlib.pyplot as plt

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
sn_scalefac = 2.0842526537870818e+48  # see sn_scaling.py 
sn_day_arr = np.arange(-19,51,1)

av_optfindarr = np.arange(0.5, 5.5, 0.5)
redshift_optfindarr = np.arange(0.01, 3.01, 0.01)

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

sn_opt_arr = np.load('/Volumes/Joshi_external_HDD/Roman/allsnmodspec.npy')

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

    zp = 1.0
    zps = 0.5

    z, day, av = theta

    if ( 0.0001 <= z <= 3.0  and  -19 <= day <= 50  and  0.0 <= av <= 5.0):

        # Gaussian prior on redshift
        ln_pz = np.log( 1.0 / (np.sqrt(2*np.pi)*zps) ) - 0.5*(z - zp)**2/zps**2

        return ln_pz
    
    return -np.inf

def logpost_sn(theta, x, data, err):

    lp = logprior_sn(theta)
    
    if not np.isfinite(lp):
        return -np.inf
    
    lnL = loglike_sn(theta, x, data, err)
    
    return lp + lnL

#@jit(nopython=True)
# griddata is an issue for jit
# maybe a manually written 'griddata' would be okay
def model_sn(x, z, day, sn_av):

    # pull out spectrum for the chosen day
    day_idx_ = np.argmin(abs(sn_day_arr - day))
    day_idx = np.where(salt2_spec['day'] == sn_day_arr[day_idx_])[0]

    sn_spec_llam = salt2_spec['flam'][day_idx] * sn_scalefac
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

def retrieve_sn_optpars(big_index):

    av_subidx, z_idx = np.divmod(big_index, len(redshift_optfindarr))
    trash, av_idx    = np.divmod(av_subidx, len(av_optfindarr))
    phase_idx, trash = np.divmod(big_index, len(av_optfindarr)*len(redshift_optfindarr))

    #print(z_idx, av_subidx, phase_idx, trash)

    z = redshift_optfindarr[z_idx]
    av = av_optfindarr[av_idx]
    phase = sn_day_arr[phase_idx]

    del trash

    return z, phase, av

def retrieve_sn_optpars_inverse(z, phase, av):

    k = np.argmin(abs(redshift_optfindarr - z))
    j = np.argmin(abs(av_optfindarr - av))
    i = np.argmin(abs(sn_day_arr - phase))

    big_index = (i * len(av_optfindarr) * len(redshift_optfindarr)) + \
                (j * len(redshift_optfindarr)) + k

    return big_index

def get_optimal_position(wav, flam, ferr):

    verbose = False

    if verbose: print('\nGetting optimal starting position...')

    model_a = np.sum(flam * sn_opt_arr / ferr**2, axis=1) / np.sum(sn_opt_arr**2 / ferr**2, axis=1)

    optmod_eff = sn_opt_arr.T * model_a
    optmod_eff = optmod_eff.T

    chi2_opt = ((flam - optmod_eff) / ferr )**2
    chi2_opt = np.sum(chi2_opt, axis=1)

    big_index = np.argmin(chi2_opt)

    z_prior, phase_prior, av_prior = retrieve_sn_optpars(big_index)

    if verbose:

        print('Data shape:', flam.shape)
        print('Opt find array shape:', sn_opt_arr.shape)
        print('A shape:', model_a.shape)
        print('Effective model shape:', optmod_eff.shape)
        print('Chi2 array shape:', chi2_opt.shape)
        print('Chi2 array:', chi2_opt)
        print('Min chi2:', chi2_opt[big_index], np.min(chi2_opt))
        print('Big index:', big_index)
        print('Retrieved priors:', z_prior, phase_prior, av_prior)
        print('---------------------------\n')

        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ax.plot(wav, flam, color='k')
        ax.plot(wav, model_a[big_index] * sn_opt_arr[big_index], color='crimson')
        ax.plot(wav, model_a[true_big_index] * sn_opt_arr[true_big_index], color='orchid')

        ax.fill_between(wav, flam-ferr_lo,flam+ferr_hi, color='gray', alpha=0.5)

        ax.axhline(y=0.0, ls='--', color='navy')

        odiff = 0
        tdiff = 0

        for i in range(len(flam)):

            od = (flam[i] - model_a[big_index] * sn_opt_arr[big_index][i])**2 / ferr[i]**2
            td = (flam[i] - model_a[true_big_index] * sn_opt_arr[true_big_index][i])**2 / ferr[i]**2

            odiff += od
            tdiff += td

            #print('\n')
            #print(wav[i], '{:.3e}'.format(flam[i]), \
            #    '{:.3e}'.format(model_a[big_index] * sn_opt_arr[big_index][i]), \
            #    '{:.3e}'.format(ferr[i]), '{:.3f}'.format(od), odiff)
            #print(wav[i], '{:.3e}'.format(flam[i]), \
            #    '{:.3e}'.format(model_a[true_big_index] * sn_opt_arr[true_big_index][i]), \
            #    '{:.3e}'.format(ferr[i]), '{:.3f}'.format(td), tdiff)

        chi2o = np.sum((flam - model_a[big_index] * sn_opt_arr[big_index])**2/ferr**2, axis=None)
        chi2t = np.sum((flam - model_a[true_big_index]* sn_opt_arr[true_big_index])**2/ferr**2, axis=None)
        print('Manual chi2 opt:',  chi2o)
        print('Manual chi2 true:', chi2t)
        print(model_a[big_index], model_a[true_big_index])
        print(odiff, tdiff)
        
        ax.text(x=0.75, y=0.2,  s='z = ' + '{:.3f}'.format(z_prior),
            verticalalignment='top', horizontalalignment='left',
            transform=ax.transAxes, color='royalblue', size=12)
        ax.text(x=0.75, y=0.15, s='Phase = ' + '{:d}'.format(phase_prior),
            verticalalignment='top', horizontalalignment='left',
            transform=ax.transAxes, color='royalblue', size=12)
        ax.text(x=0.75, y=0.1,  s='Av = ' + '{:.3f}'.format(av_prior),
            verticalalignment='top', horizontalalignment='left',
            transform=ax.transAxes, color='royalblue', size=12)

        plt.show()
        fig.clear()
        plt.close(fig)

    del optmod_eff

    return z_prior, phase_prior, av_prior

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
    jump_size_z   = 0.1
    jump_size_av  = 0.5  # magnitudes
    jump_size_day = 3  # days

    # Setup dims and walkers
    nwalkers = 500
    niter    = 1000
    ndim_sn  = 3

    # ----------------------- Loop over all simulated and extracted SN spectra ----------------------- #
    # Arrays to loop over
    pointings = np.arange(2, 3)
    detectors = np.arange(1, 5, 1)

    for pt in pointings:
        for det in detectors:

            img_suffix = img_filt + str(pt) + '_' + str(det)

            # --------------- Read in sed.lst
            sedlst_header = ['segid', 'sed_path']
            sedlst_path = '/Volumes/Joshi_external_HDD/Roman/pylinear_lst_files/' + 'sed_' + img_suffix + '.lst'
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

            # --------------- loop and find all SN segids
            all_sn_segids = []
            for i in range(len(sedlst)):
                if 'salt' in sedlst['sed_path'][i]:
                    all_sn_segids.append(sedlst['segid'][i])

            print('ALL SN segids in this file:', all_sn_segids)

            # --------------- Loop over all extracted files and SN in each file
            expcount = 0
            for ext_hdu in all_hdus:

                for segid in all_sn_segids:

                    #if segid == 188 and img_suffix == 'Y106_0_17':
                    #    print('Skipping SN', segid, 'in img_suffix', img_suffix)
                    #    continue

                    print("\n-----------------")
                    print("Fitting SegID:", segid, "with exposure time:", all_exptimes[expcount])
 
                    # ----- Get spectrum
                    segid_idx = int(np.where(sedlst['segid'] == segid)[0])

                    template_name = os.path.basename(sedlst['sed_path'][segid_idx])
                    template_inputs = get_template_inputs(template_name)  # needed for plotting

                    wav = ext_hdu[('SOURCE', segid)].data['wavelength']
                    flam = ext_hdu[('SOURCE', segid)].data['flam'] * pylinear_flam_scale_fac

                    ferr_lo = ext_hdu[('SOURCE', segid)].data['flounc'] * pylinear_flam_scale_fac
                    ferr_hi = ext_hdu[('SOURCE', segid)].data['fhiunc'] * pylinear_flam_scale_fac

                    # Smooth with boxcar
                    smoothing_width_pix = 5
                    sf = convolve(flam, Box1DKernel(smoothing_width_pix))

                    # ----- Check SNR
                    snr = get_snr(wav, flam)
                    smoothed_snr = get_snr(wav, sf)

                    print("SNR for this spectrum:", "{:.2f}".format(snr), "{:.2f}".format(smoothed_snr))

                    if snr < 3.0:
                        print("Skipping due to low SNR.")
                        continue

                    if smoothed_snr > 2 * snr:
                        flam = sf
                        print(f'{bcolors.HEADER}')
                        print("------> Fitting smoothed spectrum.")
                        print(f'{bcolors.ENDC}')

                    # ----- Set noise level based on snr
                    #noise_lvl = 1/snr
                    # Create ferr array
                    #ferr = noise_lvl * flam

                    ferr = (ferr_lo + ferr_hi)/2.0

                    # ----- Get optimal starting position
                    z_prior, phase_prior, av_prior = get_optimal_position(wav, flam, ferr)
                    rsn_init = np.array([z_prior, phase_prior, av_prior])
                    # redshift, day relative to peak, and dust extinction

                    # generating ball of walkers about optimal position defined above
                    pos_sn = np.zeros(shape=(nwalkers, ndim_sn))

                    for i in range(nwalkers):

                        # ---------- For SN
                        rsn0 = float(rsn_init[0] + jump_size_z * np.random.normal(size=1))
                        rsn1 = int(rsn_init[1] + jump_size_day * np.random.normal(size=1))
                        rsn2 = float(rsn_init[2] + jump_size_av * np.random.normal(size=1))

                        rsn = np.array([rsn0, rsn1, rsn2])

                        pos_sn[i] = rsn

                    # ----- Clip data at the ends
                    wav_idx = np.where((wav > 7600) & (wav < 18000))[0]

                    sn_wav = wav[wav_idx]
                    sn_flam = flam[wav_idx]
                    sn_ferr = ferr[wav_idx]

                    # ----- Set up args
                    args_sn = [sn_wav, sn_flam, sn_ferr]

                    print("logpost at starting position for SN:")
                    print(logpost_sn(rsn_init, sn_wav, sn_flam, sn_ferr))
                    print("Starting position:", rsn_init)

                    # ----- Now run emcee on SN
                    snstr = str(segid) + '_' + img_suffix + all_exptimes[expcount]
                    emcee_savefile = results_dir + \
                                     'emcee_sampler_sn' + snstr + '.h5'
                    if not os.path.isfile(emcee_savefile):

                        backend = emcee.backends.HDFBackend(emcee_savefile)
                        backend.reset(nwalkers, ndim_sn)
                            
                        with Pool(6) as pool:
                            sampler = emcee.EnsembleSampler(nwalkers, ndim_sn, logpost_sn,
                                args=args_sn, pool=pool, backend=backend)
                            sampler.run_mcmc(pos_sn, niter, progress=True)

                        print(f"{bcolors.GREEN}")
                        print("Finished running emcee.")
                        print("Mean acceptance Fraction:", np.mean(sampler.acceptance_fraction), "\n")
                        print(f"{bcolors.ENDC}")

                        # ---------- Stuff needed for plotting
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