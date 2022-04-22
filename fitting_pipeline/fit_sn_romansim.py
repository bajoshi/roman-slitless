import numpy as np
from astropy.io import fits
from scipy.interpolate import griddata
from numba import jit
import emcee
from astropy.convolution import convolve, Box1DKernel  # noqa

import os
import sys
import time
from multiprocessing import Pool

import matplotlib.pyplot as plt

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
astropy_cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K,
                              Om0=0.3)

# Assign directories and custom imports
cwd = os.path.dirname(os.path.abspath(__file__))
fitting_utils = cwd + '/utils/'

roman_slitless_dir = os.path.dirname(cwd)
extdir = "/Volumes/Joshi_external_HDD/Roman/"
ext_spectra_dir = extdir + "roman_slitless_sims_results/run1/"
results_dir = ext_spectra_dir + 'fitting_results/refitting/'
pylinear_lst_dir = extdir + 'pylinear_lst_files/run1/'
dirimg_dir = extdir + 'roman_direct_sims/sims2021/K_5degimages_part1/'

sys.path.append(roman_slitless_dir)
sys.path.append(fitting_utils)
from get_snr import get_snr # noqa
from get_template_inputs import get_template_inputs # noqa
import dust_utils as du # noqa
from snfit_plots import read_pickle_make_plots_sn # noqa
from gen_sed_lst import get_sn_z  # noqa


# ### ------ DONE WITH IMPORTS ------ ### #
start = time.time()

# Define any required constants/arrays
sn_scalefac = 1.734e40  # see sn_scaling.py
sn_day_arr = np.arange(-19, 51, 1)

# Set pylinear f_lambda scaling factor
pylinear_flam_scale_fac = 1e-17

av_optfindarr = np.arange(0.5, 5.5, 0.5)
redshift_optfindarr = np.arange(0.01, 3.01, 0.01)

# Load in all models
# ------ THIS HAS TO BE GLOBAL!

# Read in SALT2 SN IA file from Lou
salt2_spec = np.genfromtxt(fitting_utils + "templates/salt2_template_0.txt",
                           dtype=None, names=['day', 'lam', 'flam'],
                           encoding='ascii')

# Also load in lookup table for luminosity distance
dl_cat = np.genfromtxt(fitting_utils + 'dl_lookup_table.txt',
                       dtype=None, names=True)
# Get arrays
dl_z_arr = np.asarray(dl_cat['z'], dtype=np.float64)
dl_cm_arr = np.asarray(dl_cat['dl_cm'], dtype=np.float64)

del dl_cat

sn_opt_arr = np.load('/Volumes/Joshi_external_HDD/Roman/allsnmodspec.npy')
print("Done loading all models. Time taken:",
      "{:.3f}".format(time.time() - start), "seconds.")
# --------------------------------------

# Check directories
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)


class bcolors:
    """
    This class came from stackoverflow
    SEE:
    https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-python
    """
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

    # print("Chi2 term:", np.sum((y-data)**2/err**2))
    # print("Second loglikelihood term:",
    #       np.nansum( np.log(2 * np.pi * err**2)) )
    # print("ln(likelihood) SN", lnLike)

    """
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'$\lambda\, [\mathrm{\AA}]$', fontsize=14)  # noqa
    ax.set_ylabel(r'$f_\lambda\, [\mathrm{cgs}]$', fontsize=14)  # noqa

    ax.plot(x, data, color='k')
    ax.fill_between(x, data - err, data + err, color='gray', alpha=0.5)

    ax.plot(x, y, color='firebrick')

    plt.show()
    #sys.exit(0)
    """

    return lnLike


def logprior_sn(theta):

    zp = 1.2
    zps = 0.7

    z, day, av = theta

    if (0.0001 <= z <= 3.0 and -19 <= day <= 50 and 0.0 <= av <= 5.0):

        # Gaussian prior on redshift
        ln_pz = np.log(1.0 / (np.sqrt(2*np.pi)*zps)) - 0.5*(z - zp)**2/zps**2

        return ln_pz

    return -np.inf


def logpost_sn(theta, x, data, err):

    lp = logprior_sn(theta)

    if not np.isfinite(lp):
        return -np.inf

    lnL = loglike_sn(theta, x, data, err)

    return lp + lnL


# @jit(nopython=True)
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
    lnLike = -0.5 * np.nansum((y-data)**2/err**2)
    return lnLike


def retrieve_sn_optpars(big_index):

    av_subidx, z_idx = np.divmod(big_index, len(redshift_optfindarr))
    trash, av_idx = np.divmod(av_subidx, len(av_optfindarr))
    phase_idx, trash = np.divmod(big_index, 
                                 len(av_optfindarr)*len(redshift_optfindarr))

    # print(z_idx, av_subidx, phase_idx, trash)

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


def get_optimal_position(wav, flam, ferr, opt_args=None):

    if opt_args is not None:
        verbose = opt_args['verbose']
        ztrue = opt_args['ztrue']
        phasetrue = opt_args['phasetrue']
        avtrue = opt_args['avtrue']
        ferr_lo = opt_args['ferr_lo']
        ferr_hi = opt_args['ferr_hi']
    else:
        verbose = False

    if verbose:
        print('\nGetting optimal starting position...')

    # -------
    # Clip and only fit for optimal position in the central part of the data
    # I think because the sensitivity of the prism is lower in the blue end
    # compared to the red end, the blue end needs to be brought in more than
    # the red.
    # These MUST be the same as the limits in save_sn_optimal_arr.py
    clip_idx = np.where((wav >= 10000) & (wav <= 16000))[0]
    wav = wav[clip_idx]
    flam = flam[clip_idx]
    ferr = ferr[clip_idx]

    # -------
    # Get vertical scaling factor for all models
    a_num = np.sum(flam * sn_opt_arr / ferr**2, axis=1)
    a_den = np.sum(sn_opt_arr**2 / ferr**2, axis=1)
    model_a = a_num / a_den

    # get model array to correct shape and compute chi2
    optmod_eff = sn_opt_arr.T * model_a
    optmod_eff = optmod_eff.T

    chi2_opt = ((flam - optmod_eff) / ferr)**2
    chi2_opt = np.sum(chi2_opt, axis=1)

    # -------
    # Find the global minimum and retrieve corresponding params
    big_index = np.argmin(chi2_opt)
    z_prior, phase_prior, av_prior = retrieve_sn_optpars(big_index)

    # Also find other local minima which could potentially
    # be very close to the global minimum (in chi2) but far
    # in parameter space.
    # Finally collect global and other close local minima and return
    # sort_idx = np.argsort(chi2_opt)
    # print(big_index)
    # print(z_prior, phase_prior, av_prior, '\n')
    # for i in range(30):
    #     idx = sort_idx[i]
    #     print(idx, retrieve_sn_optpars(idx), chi2_opt[idx])

    if verbose:

        print('\n---------------------------')
        # print('Data shape:', flam.shape)
        # print('Opt find array shape:', sn_opt_arr.shape)
        # print('A shape:', model_a.shape)
        # print('Effective model shape:', optmod_eff.shape)
        # print('Chi2 array shape:', chi2_opt.shape)
        print('Chi2 array:', chi2_opt)
        print('Min chi2:', chi2_opt[big_index], np.min(chi2_opt))
        print('Big index:', big_index)
        print('Retrieved priors:', z_prior, phase_prior, av_prior)

        # True index
        true_big_index = retrieve_sn_optpars_inverse(ztrue, phasetrue, avtrue)
        print('True big index:', true_big_index)
        print('---------------------------\n')

        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ax.plot(wav, flam, color='k')
        ax.plot(wav, model_a[big_index] * sn_opt_arr[big_index], 
                color='crimson', label='Model at optimal pos')
        ax.plot(wav, model_a[true_big_index] * sn_opt_arr[true_big_index], 
                color='orchid', label='Model at true pos')

        # Clipping ferr. Wont be done before because ferr(lo/hi) is optional
        ferr_lo = ferr_lo[clip_idx]
        ferr_hi = ferr_hi[clip_idx]
        ax.fill_between(wav, flam-ferr_lo, flam+ferr_hi, 
                        color='gray', alpha=0.5)

        ax.axhline(y=0.0, ls='--', color='navy')

        odiff = 0
        tdiff = 0

        for i in range(len(flam)):

            od = (flam[i] - model_a[big_index] * 
                  sn_opt_arr[big_index][i])**2 / ferr[i]**2
            td = (flam[i] - model_a[true_big_index] * 
                  sn_opt_arr[true_big_index][i])**2 / ferr[i]**2

            odiff += od
            tdiff += td

            # print('\n')
            # print(wav[i], '{:.3e}'.format(flam[i]), \
            #     '{:.3e}'.format(model_a[big_index] *
            #                     sn_opt_arr[big_index][i]),
            #     '{:.3e}'.format(ferr[i]), '{:.3f}'.format(od), odiff)
            # print(wav[i], '{:.3e}'.format(flam[i]), \
            #     '{:.3e}'.format(model_a[true_big_index] *
            #                     sn_opt_arr[true_big_index][i]),
            #     '{:.3e}'.format(ferr[i]), '{:.3f}'.format(td), tdiff)

        chi2o = np.sum((flam - model_a[big_index]
                        * sn_opt_arr[big_index])**2 / ferr**2, axis=None)
        chi2t = np.sum((flam - model_a[true_big_index]
                        * sn_opt_arr[true_big_index])**2 / ferr**2, axis=None)
        print('Manual chi2 opt:', chi2o)
        print('Manual chi2 true:', chi2t)
        print(model_a[big_index], model_a[true_big_index])
        print(odiff, tdiff)

        ax.text(x=0.75, y=0.25, s='z = ' + '{:.3f}'.format(z_prior),
                verticalalignment='top', horizontalalignment='left',
                transform=ax.transAxes, color='royalblue', size=14)
        ax.text(x=0.75, y=0.2, s='Phase = ' + '{:d}'.format(phase_prior),
                verticalalignment='top', horizontalalignment='left',
                transform=ax.transAxes, color='royalblue', size=14)
        ax.text(x=0.75, y=0.15, s='Av = ' + '{:.3f}'.format(av_prior),
                verticalalignment='top', horizontalalignment='left',
                transform=ax.transAxes, color='royalblue', size=14)

        ax.legend(loc=0, fontsize=14, frameon=False)

        plt.show()
        fig.clear()
        plt.close(fig)

    del optmod_eff

    return z_prior, phase_prior, av_prior


def main():

    # ----------------------- Preliminary stuff ----------------------- #
    ext_root = "romansim_prism_"
    img_filt = 'Y106_'

    # ext_root = "shortsim_"
    # shortsim_dir = extdir + \
    #     "roman_direct_sims/sims2021/K_5degimages_part1/shortsim/"

    # exptime1 = '_10800s'
    exptime2 = '_3600s'
    exptime3 = '_1200s'
    exptime4 = '_400s'

    all_exptimes = [exptime2, exptime3, exptime4]

    # ----------------------- Using emcee ----------------------- #
    # Labels for corner and trace plots
    label_list_sn = [r'$z$', r'$Day$', r'$A_V [mag]$']

    # Set jump sizes # ONLY FOR INITIAL POSITION SETUP
    jump_size_z = 0.1
    jump_size_av = 0.5  # magnitudes
    jump_size_day = 3  # days

    # Setup dims and walkers
    nwalkers = 500
    niter = 1000
    ndim_sn = 3

    # ---------- Loop over all simulated and extracted SN spectra ---------- #
    # Arrays to loop over
    pointings = np.arange(0, 1)
    detectors = np.arange(1, 19, 1)

    for pt in pointings:
        for det in detectors:

            img_suffix = img_filt + str(pt) + '_' + str(det)
            # img_suffix = 'shortsim'

            # --------------- Read in sed.lst
            sedlst_header = ['segid', 'sed_path']
            sedlst_path = pylinear_lst_dir + 'sed_' + img_suffix + '.lst'
            # sedlst_path = shortsim_dir + 'sed_' + img_suffix + '.lst'
            sedlst = np.genfromtxt(sedlst_path, dtype=None,
                                   names=sedlst_header, encoding='ascii')
            print("Read in sed.lst from:", sedlst_path)

            print("Number of spectra in file:", len(sedlst))

            # --------------- loop and find all SN segids
            all_sn_segids = []
            for i in range(len(sedlst)):
                if ('salt' in sedlst['sed_path'][i])\
                        or ('contam' in sedlst['sed_path'][i]):
                    all_sn_segids.append(sedlst['segid'][i])

            print('ALL SN segids in this file:', all_sn_segids)

            # --------------- Loop over all extracted files
            for e in range(len(all_exptimes)):

                exptime = all_exptimes[e]

                # --------------- Read in the extracted spectra
                # for full sim
                ext_spec_filename = (ext_spectra_dir + ext_root + img_suffix
                                     + exptime + '_x1d.fits')
                # for shortsim
                # ext_spec_filename = ext_spectra_dir + ext_root + '_x1d.fits'
                ext_hdu = fits.open(ext_spec_filename)
                print("Read in extracted spectra from:", ext_spec_filename)

                # Loop over each SN in x1d file
                for segid in all_sn_segids:

                    # replace = False
                    fitsmooth = False

                    print("\n#####################################")
                    print("Fitting SegID:", segid,
                          "with exposure time:", exptime)

                    # ----- Get spectrum
                    segid_idx = int(np.where(sedlst['segid'] == segid)[0])

                    template_name = \
                        os.path.basename(sedlst['sed_path'][segid_idx])
                    # Get template inputs needed for plotting
                    template_inputs = get_template_inputs(template_name)
                    print('Template inputs:', template_inputs)
                    if 'contam' in template_name:
                        print('---Contains host-galaxy OVERLAP---')

                    # This is get to faster results to show in
                    # the schematic figure, i.e., only fit the
                    # SNe in the redshift ranges you want.
                    # ztrue = template_inputs[0]
                    # proceed = False
                    # if (0.5 < ztrue < 0.6) or (0.9 < ztrue < 1.1) or \
                    #    (1.4 < ztrue < 1.6) or (1.9 < ztrue < 2.1):
                    #     proceed = True
                    # if not proceed:
                    #     continue

                    wav = ext_hdu[('SOURCE', segid)].data['wavelength']
                    flam = ext_hdu[('SOURCE', segid)].data['flam'] * \
                        pylinear_flam_scale_fac

                    ferr_lo = ext_hdu[('SOURCE', segid)].data['flounc'] * \
                        pylinear_flam_scale_fac
                    ferr_hi = ext_hdu[('SOURCE', segid)].data['fhiunc'] * \
                        pylinear_flam_scale_fac

                    # ----- Smooth with boxcar
                    # smoothing_width_pix = 3
                    # sf = convolve(flam, Box1DKernel(smoothing_width_pix))

                    # ----- Get noise level
                    ferr = (ferr_lo + ferr_hi) / 2.0
                    # noise_correct = ferr * 5
                    # sf_noised = np.zeros(len(sf))
                    # for w in range(len(wav)):
                    #     sf_noised[w] = \
                    #         np.random.normal(loc=sf[w],
                    #                          scale=noise_correct[w], size=1)

                    # flam = sf_noised
                    # ferr = noise_correct

                    # ----- Check SNR
                    snr = get_snr(wav, flam)
                    # smoothed_snr = get_snr(wav, sf)
                    # snr = np.nanmean(flam / ferr)
                    # print("Avg SNR per pix for this spectrum:",
                    #       "{:.2f}".format(snr))
                    print('SNR from Stoehr et al. algorithm:',
                          "{:.2f}".format(snr))

                    # Check if file exists and continue if all okay
                    snstr = str(segid) + '_' + img_suffix + exptime
                    emcee_savefile = (results_dir + 'emcee_sampler_sn'
                                      + snstr + '.h5')

                    if snr < 3.0:
                        # if os.path.isfile(emcee_savefile):
                        #     os.remove(emcee_savefile)
                        #     print('Removed:',
                        #           os.path.basename(emcee_savefile))
                        continue
                        # if (smoothed_snr > 2 * snr) and (smoothed_snr > 3.0):
                        #     #fitsmooth = True
                        #     #flam = sf
                        #     #ferr /= np.sqrt(smoothing_width_pix)
                        #     #print(f'{bcolors.HEADER}',
                        #     #      "------> Fitting smoothed spectrum.",
                        #     #      f'{bcolors.ENDC}')
                        #     replace = True
                        # else:
                        #     continue

                    # if replace:
                    #     if os.path.isfile(emcee_savefile):
                    #         os.remove(emcee_savefile)

                    if not os.path.isfile(emcee_savefile):
                        # ----- Get optimal starting position
                        # Fix the crazy flam and ferr values
                        # before getting optimal pos
                        # snr_array = flam / ferr
                        # nan_idx = np.where(np.isnan(snr_array))[0]

                        # opt_dict = {'verbose': True,
                        #             'ztrue': template_inputs[0],
                        #             'phasetrue': template_inputs[1],
                        #             'avtrue': template_inputs[2],
                        #             'ferr_lo': ferr_lo,
                        #             'ferr_hi': ferr_hi}

                        z_prior, phase_prior, av_prior = \
                            get_optimal_position(wav, flam, ferr)
                        rsn_init = np.array([z_prior, phase_prior, av_prior])
                        # redshift, day relative to peak, and dust extinction

                        print("logpost at starting position for SN:")
                        print(logpost_sn(rsn_init, wav, flam, ferr))
                        print("Starting position:", rsn_init)

                        # generating ball of walkers about optimal
                        # position defined above
                        pos_sn = np.zeros(shape=(nwalkers, ndim_sn))

                        for i in range(nwalkers):

                            # ---------- For SN
                            rsn0 = float(rsn_init[0] + jump_size_z
                                         * np.random.normal(size=1))
                            rsn1 = int(rsn_init[1] + jump_size_day
                                       * np.random.normal(size=1))
                            rsn2 = float(rsn_init[2] + jump_size_av
                                         * np.random.normal(size=1))

                            rsn = np.array([rsn0, rsn1, rsn2])

                            pos_sn[i] = rsn

                        # ----- Clip data at the ends
                        wav_idx = np.where((wav > 7800) & (wav < 18000))[0]

                        wav = wav[wav_idx]
                        flam = flam[wav_idx]
                        ferr = ferr[wav_idx]

                        # ----- Set up args
                        args_sn = [wav, flam, ferr]

                        # ----- Now run emcee on SN
                        backend = emcee.backends.HDFBackend(emcee_savefile)
                        backend.reset(nwalkers, ndim_sn)

                        with Pool(6) as pool:
                            sampler = emcee.EnsembleSampler(nwalkers, ndim_sn,
                                                            logpost_sn,
                                                            args=args_sn,
                                                            pool=pool,
                                                            backend=backend)
                            sampler.run_mcmc(pos_sn, niter, progress=True)

                        print(f"{bcolors.GREEN}")
                        print("Finished running emcee.")
                        print("Mean acceptance Fraction:",
                              np.mean(sampler.acceptance_fraction), "\n")
                        print(f"{bcolors.ENDC}")

                        # ---------- Stuff needed for plotting
                        truth_dict = {}
                        truth_dict['z'] = template_inputs[0]
                        truth_dict['phase'] = template_inputs[1]
                        truth_dict['Av'] = template_inputs[2]

                        if fitsmooth:
                            orig_wav = \
                                ext_hdu[('SOURCE', segid)].data['wavelength']
                            orig_spec = \
                                ext_hdu[('SOURCE', segid)].data['flam'] * \
                                pylinear_flam_scale_fac
                            orig_ferr = (ferr_lo + ferr_hi) / 2.0
                            read_pickle_make_plots_sn('sn' + snstr,
                                                      ndim_sn, args_sn,
                                                      label_list_sn,
                                                      truth_dict, results_dir,
                                                      fitsmooth=True,
                                                      orig_wav=orig_wav,
                                                      orig_spec=orig_spec,
                                                      orig_ferr=orig_ferr)

                        else:
                            read_pickle_make_plots_sn('sn' + snstr,
                                                      ndim_sn, args_sn,
                                                      label_list_sn,
                                                      truth_dict, results_dir)

                        print("Finished plotting results.")

                # --------------- close all open fits files
                ext_hdu.close()

    return None


if __name__ == '__main__':
    main()
    sys.exit(0)
