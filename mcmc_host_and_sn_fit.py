import numpy as np
from astropy.io import fits
import emcee
import corner
import scipy
from scipy.interpolate import griddata
from astropy.cosmology import Planck15
from multiprocessing import Pool
import pickle

import os
import sys
from functools import reduce
import time
import datetime

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec

home = os.getenv('HOME')
stacking_utils = home + '/Documents/GitHub/stacking-analysis-pears/util_codes/'
pears_figs_dir = home + '/Documents/pears_figs_data/'

roman_slitless_dir = home + "/Documents/GitHub/roman-slitless/"
ext_spectra_dir = home + "/Documents/roman_slitless_sims_results/"
template_dir = home + "/Documents/roman_slitless_sims_seds/"
roman_sims_seds = home + "/Documents/roman_slitless_sims_seds/"

modeldir = "/Volumes/Heather_extdrive/bc03_output_dir/"  # home + '/Documents/bc03_output_dir/'

grism_sens_cat = np.genfromtxt(home + '/Documents/pylinear_ref_files/pylinear_config/Roman/roman_throughput_20190325.txt', \
    dtype=None, names=True, skip_header=3)

grism_sens_wav = grism_sens_cat['Wave'] * 1e4  # the text file has wavelengths in microns # needed in angstroms
grism_sens = grism_sens_cat['BAOGrism_1st']
grism_wav_idx = np.where(grism_sens > 0.25)

sys.path.append(stacking_utils)
import proper_and_lum_dist as cosmo
from dust_utils import get_dust_atten_model
from bc03_utils import get_bc03_spectrum

# Read in SALT2 SN IA file from Lou
salt2_spec = np.genfromtxt(roman_sims_seds + "salt2_template_0.txt", \
    dtype=None, names=['day', 'lam', 'flam'], encoding='ascii')

# Read in all models and parameters
model_lam_grid = np.load(pears_figs_dir + 'model_lam_grid_withlines_chabrier.npy', mmap_mode='r')
model_grid = np.load(pears_figs_dir + 'model_comp_spec_llam_withlines_chabrier.npy', mmap_mode='r')

log_age_arr = np.load(pears_figs_dir + 'log_age_arr_chab.npy', mmap_mode='r')
metal_arr = np.load(pears_figs_dir + 'metal_arr_chab.npy', mmap_mode='r')
tau_gyr_arr = np.load(pears_figs_dir + 'tau_gyr_arr_chab.npy', mmap_mode='r')
tauv_arr = np.load(pears_figs_dir + 'tauv_arr_chab.npy', mmap_mode='r')

"""
Array ranges are:
1. Age: 7.02 to 10.114 (this is log of the age in years)
2. Metals: 0.0001 to 0.05 (absolute fraction of metals. All CSP models although are fixed at solar = 0.02)
3. Tau: 0.01 to 63.095 (this is in Gyr. SSP models get -99.0)
4. TauV: 0.0 to 2.8 (Visual dust extinction in magnitudes. SSP models get -99.0)
"""

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


def get_template(age, tau, tauv, metallicity, \
    log_age_arr, metal_arr, tau_gyr_arr, tauv_arr, \
    model_lam_grid_withlines_mmap, model_comp_spec_withlines_mmap):

    """
    print("\nFinding closest model to --")
    print("Age [Gyr]:", 10**age / 1e9)
    print("Tau [Gyr]:", tau)
    print("Tau_v:", tauv)
    print("Metallicity [abs. frac.]:", metallicity)
    """

    # First find closest values and then indices corresponding to them
    # It has to be done this way because you typically wont find an exact match
    closest_age_idx = np.argmin(abs(log_age_arr - age))
    closest_tau_idx = np.argmin(abs(tau_gyr_arr - tau))
    closest_tauv_idx = np.argmin(abs(tauv_arr - tauv))

    # Now get indices
    age_idx = np.where(log_age_arr == log_age_arr[closest_age_idx])[0]
    tau_idx = np.where(tau_gyr_arr == tau_gyr_arr[closest_tau_idx])[0]
    tauv_idx = np.where(tauv_arr   ==    tauv_arr[closest_tauv_idx])[0]
    metal_idx = np.where(metal_arr == metallicity)[0]

    model_idx = int(reduce(np.intersect1d, (age_idx, tau_idx, tauv_idx, metal_idx)))

    model_llam = model_comp_spec_withlines_mmap[model_idx]

    chosen_age = 10**log_age_arr[model_idx] / 1e9
    chosen_tau = tau_gyr_arr[model_idx]
    chosen_av = 1.086 * tauv_arr[model_idx]
    chosen_metallicity = metal_arr[model_idx]

    """
    print("\nChosen model index:", model_idx)
    print("Chosen model parameters -- ")
    print("Age [Gyr]:", chosen_age)
    print("Tau [Gyr]:", chosen_tau)
    print("A_v:", chosen_av)
    print("Metallicity [abs. frac.]:", chosen_metallicity)
    """

    return model_llam

def loglike_sn(theta, x, data, err, host_flam):
    
    z, day, host_frac = theta

    y = model_sn(x, z, day, host_frac, host_flam)
    #print("Model SN func result:", y)

    # ------- Vertical scaling factor
    alpha = np.nansum(data * y / err**2) / np.nansum(y**2 / err**2)
    #print("Alpha SN:", "{:.2e}".format(alpha))

    y = y * alpha

    lnLike = -0.5 * np.nansum( (y-data)**2/err**2 ) #  +  np.log(2 * np.pi * err**2))
    #print("ln(likelihood) SN", lnLike)

    #print("Chi2 array:", (y-data)**2/err**2)
    #print("Chi2 array sum:", np.sum((y-data)**2/err**2))
    #print("Second loglikelihood term:", np.log(2 * np.pi * err**2))
    
    return lnLike

def loglike_host(theta, x, data, err):
    
    z, ms, age, tau, av, lsf_sigma = theta

    y = model_host(x, z, ms, age, tau, av, lsf_sigma)
    #print("Model func result:", y)

    # ------- Clip all arrays to where grism sensitivity is >= 25%
    # then get the log likelihood
    x0 = np.where( (x >= grism_sens_wav[grism_wav_idx][0]  ) &
                   (x <= grism_sens_wav[grism_wav_idx][-1] ) )[0]

    y = y[x0]
    data = data[x0]
    err = err[x0]

    # ------- Vertical scaling factor
    #alpha = np.nansum(data * y / err**2) / np.nansum(y**2 / err**2)
    #print("Alpha HOST:", "{:.2e}".format(alpha))
    #y = y * alpha

    # ------- log likelihood
    lnLike = -0.5 * np.nansum( (y-data)**2/err**2  +  np.log(2 * np.pi * err**2))
    #print("Pure chi2 term:", np.nansum( (y-data)**2/err**2 ))
    #print("Second error term:", np.nansum(np.log(2 * np.pi * err**2)))

    return lnLike

def logprior_sn(theta):

    z, day, host_frac = theta

    if ( 0.0001 <= z <= 6.0  and  -19 <= day <= 50  and  0.0 <= host_frac <= 0.5):
        return 0.0
    
    return -np.inf

def logprior_host(theta):

    z, ms, age, tau, av, lsf_sigma = theta
    #print("Parameter vector given:", theta)
    
    # Make sure model is not older than the Universe
    # Allowing at least 100 Myr for the first galaxies to form after Big Bang
    age_at_z = Planck15.age(z).value  # in Gyr
    age_lim = age_at_z - 0.1  # in Gyr

    if ((0.0001 <= z <= 6.0) and \
        (9.0 <= ms <= 12.0) and \
        (0.01 <= age <= age_lim) and \
        (0.001 <= tau <= 20.0) and \
        (0.0 <= av <= 5.0) and \
        (0.5 <= lsf_sigma <= 20.0)):
        return 0.0
    
    return -np.inf

def logpost_sn(theta, x, data, err, host_flam):

    lp = logprior_sn(theta)

    #print("SN prior:", lp)
    
    if not np.isfinite(lp):
        return -np.inf
    
    lnL = loglike_sn(theta, x, data, err, host_flam)

    #print("SN log(likelihood):", lnL)
    
    return lp + lnL

def logpost_host(theta, x, data, err):

    lp = logprior_host(theta)
    #print("Prior HOST:", lp)
    
    if not np.isfinite(lp):
        return -np.inf
    
    lnL = loglike_host(theta, x, data, err)

    #print("Likelihood HOST:", lnL)
    
    return lp + lnL

def model_sn(x, z, day, sn_av):

    # pull out spectrum for the chosen day
    day_idx = np.where(salt2_spec['day'] == day)[0]

    sn_spec_llam = salt2_spec['flam'][day_idx]
    sn_spec_lam = salt2_spec['lam'][day_idx]

    # ------ Apply dust extinction
    sn_dusty_llam = get_dust_atten_model(sn_spec_lam, sn_spec_llam, sn_av)

    # ------ Apply redshift
    sn_lam_z, sn_flam_z = cosmo.apply_redshift(sn_spec_lam, sn_dusty_llam, z)

    # ------ Apply some LSF. 
    # This is a NUISANCE FACTOR ONLY FOR NOW
    # When we use actual SNe they will be point sources.
    #lsf_sigma = 0.5
    #sn_flam_z = scipy.ndimage.gaussian_filter1d(input=sn_flam_z, sigma=lsf_sigma)

    # ------ Downgrade to grism resolution
    sn_mod = np.zeros(len(x))

    ### Zeroth element
    lam_step = x[1] - x[0]
    idx = np.where((sn_lam_z >= x[0] - lam_step) & (sn_lam_z < x[0] + lam_step))[0]
    sn_mod[0] = np.mean(sn_flam_z[idx])

    ### all elements in between
    for j in range(1, len(x) - 1):
        idx = np.where((sn_lam_z >= x[j-1]) & (sn_lam_z < x[j+1]))[0]
        sn_mod[j] = np.mean(sn_flam_z[idx])
    
    ### Last element
    lam_step = x[-1] - x[-2]
    idx = np.where((sn_lam_z >= x[-1] - lam_step) & (sn_lam_z < x[-1] + lam_step))[0]
    sn_mod[-1] = np.mean(sn_flam_z[idx])

    # ------ combine host light
    # some fraction to account for host contamination
    # This fraction is a free parameter
    #sn_flam_hostcomb = sn_mod  +  host_frac * host_flam

    return sn_mod

def model_host(x, z, ms, age, tau, av, lsf_sigma):
    """
    Expects to get the following arguments
    
    x: observed wavelength grid
    
    z: redshift to apply to template
    ms: log of the stellar mass
    age: age of SED in Gyr
    tau: exponential SFH timescale in Gyr
    metallicity: absolute fraction of metals
    av: visual dust extinction
    """

    #met = 0.0001
    #model_lam, model_llam = get_bc03_spectrum(age, tau, met, modeldir)

    tauv = 0.0
    metallicity = 0.02
    model_llam = get_template(np.log10(age * 1e9), tau, tauv, metallicity, \
        log_age_arr, metal_arr, tau_gyr_arr, tauv_arr, \
        model_lam_grid, model_grid)

    model_lam = model_lam_grid

    model_lam, model_llam = remove_emission_lines(model_lam, model_llam)

    # ------ Apply dust extinction
    model_dusty_llam = get_dust_atten_model(model_lam, model_llam, av)

    # ------ Multiply luminosity by stellar mass
    model_dusty_llam = model_dusty_llam * 10**ms

    # ------ Apply redshift
    model_lam_z, model_flam_z = cosmo.apply_redshift(model_lam, model_dusty_llam, z)

    # ------ Apply LSF
    model_lsfconv = scipy.ndimage.gaussian_filter1d(input=model_flam_z, sigma=lsf_sigma)

    # ------ Downgrade to grism resolution
    model_mod = np.zeros(len(x))

    ### Zeroth element
    lam_step = x[1] - x[0]
    idx = np.where((model_lam_z >= x[0] - lam_step) & (model_lam_z < x[0] + lam_step))[0]
    model_mod[0] = np.mean(model_lsfconv[idx])

    ### all elements in between
    for j in range(1, len(x) - 1):
        idx = np.where((model_lam_z >= x[j-1]) & (model_lam_z < x[j+1]))[0]
        model_mod[j] = np.mean(model_lsfconv[idx])
    
    ### Last element
    lam_step = x[-1] - x[-2]
    idx = np.where((model_lam_z >= x[-1] - lam_step) & (model_lam_z < x[-1] + lam_step))[0]
    model_mod[-1] = np.mean(model_lsfconv[idx])

    # ------ Apply sensitivity curve of the dispersive element
    #grism_sens_modelgrid = griddata(points=grism_sens_wav, values=grism_sens, xi=x, method='linear')
    #grism_sens_modelgrid[grism_sens_modelgrid == 0] = np.nan
    #model_mod /= grism_sens_modelgrid

    return model_mod

def remove_emission_lines(l, ll):
    """
    This function ISN'T trying to detect any lines and then
    remove then. It is only possible for me to do it this way
    because I know what lines I added to the models.

    Copy pasted the line wavelengths from function used to 
    add the lines in. 
    In >> grismz_pipeline/fullfitting_grism_broadband_emlines.py
    """

    # ------------------ Line Wavelengths in Vacuum ------------------ #
    # Hydrogen
    h_alpha = 6564.61
    h_beta = 4862.68
    h_gamma = 4341.68
    h_delta = 4102.89

    # Metals
    # I'm defining only the important ones that I want to include here
    # Not using the wavelengths given in the Anders & Alvensleben 2003 A&A paper
    # since those are air wavelengths.
    MgII = 2799.117
    OII_1 = 3727.092
    OIII_1 = 4960.295
    OIII_2 = 5008.240
    NII_1 = 6549.86
    NII_2 = 6585.27
    SII_1 = 6718.29
    SII_2 = 6732.67

    all_line_wav = np.array([MgII, OII_1, OIII_1, OIII_2, NII_1, NII_2, SII_1, SII_2, h_alpha, h_beta, h_gamma, h_delta])
    all_line_names = np.array(['MgII', 'OII_1', 'OIII_1', 'OIII_2', 'NII_1', 'NII_2', 'SII_1', 'SII_2', 'h_alpha', 'h_beta', 'h_gamma', 'h_delta'])

    # get list of indices to be deleted in the model wavelength and spectrum arrays
    lidx = []

    for i in range(len(all_line_wav)):
        lidx.append(np.where(l == all_line_wav[i])[0])

    l_new = np.delete(arr=l, obj=lidx)
    ll_new = np.delete(arr=ll, obj=lidx)

    return l_new, ll_new

def add_noise(sig_arr, noise_level):
    """
    This function will vary the flux randomly within 
    the noise level specified. It assumes the statistical
    noise is Gaussian.
    The final error array returned is Poissonian noise.
    """
    # Poisson noise: does the signal have to be in 
    # units of photons or electrons for sqrt(N) to 
    # work? like I cannot use sqrt(signal) in physical 
    # units and call it Poisson noise?

    sigma_arr = noise_level * sig_arr

    spec_noise = np.zeros(len(sig_arr))
    err_arr = np.zeros(len(sig_arr))

    for k in range(len(sig_arr)):

        mu = sig_arr[k]
        sigma = sigma_arr[k]

        # Now vary flux using numpy random.normal
        # mu and the resulting new flux value HAVE TO BE POSITIVE!
        if mu >= 0:
            spec_noise[k] = np.random.normal(mu, sigma, 1)
        elif mu < 0:
            spec_noise[k] = np.nan

        if spec_noise[k] < 0:
            max_iters = 10
            iter_count = 0
            while iter_count < max_iters:
                spec_noise[k] = np.random.normal(mu, sigma, 1)
                iter_count += 1
                if spec_noise[k] > 0:
                    break
            # if it still hasn't given a positive number after max_iters
            # then revert it back to whatever the signal was before randomly varying
            if (iter_count >= max_iters) and (spec_noise[k] < 0):
                spec_noise[k] = sig_arr[k]

        # err_arr[k] = np.sqrt(spec_noise[k])
        err_arr[k] = 4 * noise_level * spec_noise[k]

    return spec_noise, err_arr

def get_autocorr_time(sampler):

    try:
        tau = sampler.get_autocorr_time()
    except emcee.autocorr.AutocorrError as errmsg:
        print(errmsg)
        print("\n")
        print("Emcee AutocorrError occured.")
        print("The chain is shorter than 50 times the integrated autocorrelation time for 5 parameter(s).")
        print("Use this estimate with caution and run a longer chain!")
        print("\n")

        tau_list_str = str(errmsg).split('tau:')[-1]
        tau_list = tau_list_str.split()
        print("Tau list:", tau_list)

        tau = []
        for j in range(len(tau_list)):
            curr_elem = tau_list[j]
            if ('[' in curr_elem) and (len(curr_elem) > 1):
                tau.append(float(curr_elem.lstrip('[')))
            elif (']' in curr_elem) and (len(curr_elem) > 1):
                tau.append(float(curr_elem.rstrip(']')))
            elif len(curr_elem) > 1:
                tau.append(float(tau_list[j]))

    print("Tau:", tau)

    return tau

def run_emcee(object_type, nwalkers, ndim, logpost, pos, args_obj, objid):

    print("Running on:", object_type)

    # ----------- Emcee 
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logpost, args=args_obj, pool=pool)
        sampler.run_mcmc(pos, 1000, progress=True)

    #pickle.dump(sampler.chain, open(object_type + '_' + str(objid) + '_emcee_chains.pkl', 'wb'))
    pickle.dump(sampler, open(object_type + '_' + str(objid) + '_emcee_sampler.pkl', 'wb'))

    print("Done with fitting.")

    return None

def read_pickle_make_plots(object_type, ndim, args_obj, truth_arr, label_list, objid, img_suffix, verbose=False):

    #pkl_path = roman_slitless_dir + '/emcee_diagnostics/run1/' + object_type + '_' + str(objid) + '_emcee_sampler.pkl'
    pkl_path = roman_slitless_dir + object_type + '_' + str(objid) + '_emcee_sampler.pkl'
    sampler = pickle.load(open(pkl_path, 'rb'))
    samples = sampler.get_chain()

    print(f"{bcolors.CYAN}\nRead in pickle:", pkl_path, f"{bcolors.ENDC}")
    print("Samples shape:", samples.shape)

    # plot trace
    fig1, axes1 = plt.subplots(ndim, figsize=(10, 6), sharex=True)

    for i in range(ndim):
        ax1 = axes1[i]
        ax1.plot(samples[:, :, i], "k", alpha=0.1)
        ax1.set_xlim(0, len(samples))
        ax1.set_ylabel(label_list[i])
        ax1.yaxis.set_label_coords(-0.1, 0.5)

    axes1[-1].set_xlabel("Step number")

    fig1.savefig(roman_slitless_dir + 'emcee_trace_' + object_type + '_' + str(objid) + '_' + img_suffix + '.pdf', \
        dpi=200, bbox_inches='tight')

    # Get autocorrelation time
    # Discard burn-in. You do not want to consider the burn in the corner plots/estimation.
    tau = get_autocorr_time(sampler)
    if not np.isnan(tau[0]):
        burn_in = int(3 * tau[0])
        thinning_steps = int(0.5 * tau[0])
    else:
        burn_in = 400
        thinning_steps = 67
        print("Burn-in:", burn_in)
        print("Thinning steps:", thinning_steps)

    flat_samples = sampler.get_chain(discard=burn_in, thin=thinning_steps, flat=True)
    print("Flat samples shape:", flat_samples.shape)

    # plot corner plot
    # compute weights for the samples first
    # only doing this because I noticed that 
    # some LSF values in the samples are negative
    corner_weights = np.zeros(len(flat_samples))

    for j in range(5):#len(flat_samples)):

        s = flat_samples[j]
        w = logprior_host(s)
        if w == 0.0:
            corner_weights[j] = 1.0

        if verbose:
            if corner_weights[j] == 1.0:
                wcol = bcolors.GREEN
            else:
                print("\nSkipping invalid sample... negative LSF.")
                continue

            print("\nSample number:", j)
            print(f"{wcol}Weight: ", corner_weights[j], f"{bcolors.ENDC}")
            if object_type == 'host':
                lnL = logpost_host(s, args_obj[0], args_obj[1], args_obj[2])
            else:
                lnL = logpost_sn(s, args_obj[0], args_obj[1], args_obj[2], args_obj[3])
            print("ln(likelihood): ", lnL)

            # Now plot model and data to compare
            fig = plt.figure()
            ax = fig.add_subplot(111)
            
            x = args_obj[0]
            d = args_obj[1]
            e = args_obj[2]
            y = model_host(x, s[0], s[1], s[2], s[3], s[4], s[5])

            # ------- Clip all arrays to where grism sensitivity is >= 25%
            # then get the log likelihood
            x0 = np.where( (x >= grism_sens_wav[grism_wav_idx][0]  ) &
                           (x <= grism_sens_wav[grism_wav_idx][-1] ) )[0]

            x = x[x0]
            y = y[x0]
            d = d[x0]
            e = e[x0]

            # ------- Vertical scaling factor
            a = np.nansum(d * y / e**2) / np.nansum(y**2 / e**2)
            print("Alpha HOST recomputed:", "{:.2e}".format(a))

            y = y * a

            print(f"{bcolors.CYAN}Reduced chi2 (pure chi2 NOT ln(likelihood)):", np.nansum( (y-d)**2/e**2 ) / len(y), f"{bcolors.ENDC}")
            chi2_expl = 0
            for w in range(len(x)):
                chi2_expl += (y[w]-d[w])**2/e[w]**2
            print("Explicitly summed chi2:", chi2_expl)

            ax.plot(x, d)
            ax.plot(x, y)

            plt.show()
            plt.clf()
            plt.close()

    fig = corner.corner(flat_samples, quantiles=[0.16, 0.5, 0.84], labels=label_list, weights=corner_weights, \
        label_kwargs={"fontsize": 14}, show_titles='True', title_kwargs={"fontsize": 14}, truths=truth_arr, \
        verbose=True, truth_color='tab:red')
    #range=[(1.9, 2.0), (9.0, 12.0), (0.4, 1.8), (1, 15.0), (0.0, 1.0), (0.0, 5.0)], \
    fig.savefig(roman_slitless_dir + 'corner_' + object_type + '_' + str(objid) + '_' + img_suffix + '.pdf', \
        dpi=200, bbox_inches='tight')

    # ------------ Plot 100 random models from the parameter space
    inds = np.random.randint(len(flat_samples), size=100)

    # first pull out required stuff from args
    wav = args_obj[0]
    flam = args_obj[1]
    ferr = args_obj[2]

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)

    ax3.set_xlabel(r'$\mathrm{\lambda\ [\AA]}$', fontsize=15)
    ax3.set_ylabel(r'$\mathrm{f_\lambda\ [cgs]}$', fontsize=15)

    ax3.plot(wav, flam, color='k')
    ax3.fill_between(wav, flam - ferr, flam + ferr, color='gray', alpha=0.5, zorder=3)

    for ind in inds:
        sample = flat_samples[ind]
        #print("\nAt random index:", ind)
        #print("With sample:", sample)

        # Check that LSF is not negative
        if sample[5] < 0.0:
            sample[5] = 1.0

        if object_type == 'host':
            m = model_host(wav, sample[0], sample[1], sample[2], sample[3], sample[4], sample[5])
        elif object_type == 'sn':
            m = model_sn(wav, sample[0], sample[1], sample[2], args_obj[-1])

        # ------- Clip all arrays to where grism sensitivity is >= 25%
        x0 = np.where( (wav >= grism_sens_wav[grism_wav_idx][0]  ) &
                       (wav <= grism_sens_wav[grism_wav_idx][-1] ) )[0]

        m = m[x0]
        wav = wav[x0]
        flam = flam[x0]
        ferr = ferr[x0]

        a = np.nansum(flam * m / ferr**2) / np.nansum(m**2 / ferr**2)
        ax3.plot(wav, a * m, color='tab:red', alpha=0.2, zorder=2)

    fig3.savefig(roman_slitless_dir + 'emcee_overplot_' + object_type + '_' + str(objid) + '_' + img_suffix + '.pdf', \
        dpi=200, bbox_inches='tight')

    return None

def get_optimal_fit(args_obj, object_type):

    return np.array([best_z, 10.5, best_age, best_tau, best_av, 1.0])

def main():

    print("\n * * * *    [WARNING]: model has worse resolution than data in NIR. np.mean() will result in nan. Needs fixing.    * * * *")
    print("\n * * * *    [WARNING]: check vertical scaling.    * * * *")
    print("\n * * * *    [WARNING]: use FlatLambdaCDM cosmology from astropy consistently.    * * * *")

    ext_root = "romansim1"
    img_suffix = 'Y106_11_1'

    # Read in sed.lst
    sedlst_header = ['segid', 'sed_path']
    sedlst_path = roman_slitless_dir + 'sed_' + img_suffix + '_edit.lst'
    sedlst = np.genfromtxt(sedlst_path, dtype=None, names=sedlst_header, encoding='ascii')
    print("Read in sed.lst from:", sedlst_path)

    # Read in the extracted spectra
    ext_spec_filename = ext_spectra_dir + ext_root + '_ext_x1d.fits'
    ext_hdu = fits.open(ext_spec_filename)
    print("Read in extracted spectra from:", ext_spec_filename)

    # Set pylinear f_lambda scaling factor
    pylinear_flam_scale_fac = 1e-17

    # This will come from detection on the direct image
    # For now this comes from the sedlst generation code
    # For Y106_11_1
    host_segids = np.array([475, 755, 548, 207])
    sn_segids = np.array([481, 753, 547, 241])

    for i in range(len(sedlst)):

        # Get info
        segid = sedlst['segid'][i]

        # Read in the dummy template passed to pyLINEAR
        template_name = os.path.basename(sedlst['sed_path'][i])

        if 'salt' not in template_name:
            continue
        else:
            print("\nSegmentation ID:", segid, "is a SN. Will begin fitting.")

            # Get corresponding host ID
            hostid = int(host_segids[np.where(sn_segids == segid)[0]])
            print("I have the following SN and HOST IDs:", segid, hostid)

            # ---------------------------- Set up input params dict ---------------------------- #
            print("INPUTS:")

            # Read in template file names
            print("Template name SN:", template_name)
            
            input_dict = {}

            # ---- SN
            t = template_name.split('.txt')[0].split('_')

            sn_av = float(t[-1].replace('p', '.').replace('av',''))
            sn_z = float(t[-2].replace('p', '.').replace('z',''))
            sn_day = int(t[-3].replace('day',''))
            print("Supernova input Av:", sn_av)
            print("Supernova input z:", sn_z)
            print("Supernova day:", sn_day, "\n")

            # ---- HOST
            h_idx = int(np.where(sedlst['segid'] == hostid)[0])
            h_path = sedlst['sed_path'][h_idx]
            th = os.path.basename(h_path)
            print("Template name HOST:", th)

            th = th.split('.txt')[0].split('_')

            host_av = float(th[-1].replace('p', '.').replace('av',''))
            host_met = float(th[-2].replace('p', '.').replace('met',''))
            host_tau = float(th[-3].replace('p', '.').replace('tau',''))
            host_age = float(th[-4].replace('p', '.').replace('age',''))
            host_ms = float(th[-5].replace('p', '.').replace('ms',''))
            host_z = float(th[-6].replace('p', '.').replace('z',''))

            print("Host input z:", host_z)
            print("Host input stellar mass [log(Ms/Msol)]:", host_ms)
            print("Host input age [Gyr]:", host_age)
            print("Host input tau:", host_tau)
            print("Host input metallicity:", host_met)
            print("Host input Av:", host_av)

            # Put into input dict
            input_dict['host_z'] = host_z
            input_dict['host_ms'] = host_ms
            input_dict['host_age'] = host_age
            input_dict['host_tau'] = host_tau
            input_dict['host_met'] = host_met
            input_dict['host_av'] = host_av

            input_dict['sn_z'] = sn_z
            input_dict['sn_day'] = sn_day
            input_dict['sn_av'] = sn_av

            # ---------------------------- FITTING ---------------------------- #
            # ---------- Get spectrum for host and sn
            host_wav = ext_hdu[('SOURCE', hostid)].data['wavelength']
            host_flam = ext_hdu[('SOURCE', hostid)].data['flam'] * pylinear_flam_scale_fac
        
            sn_wav = ext_hdu[('SOURCE', segid)].data['wavelength']
            sn_flam = ext_hdu[('SOURCE', segid)].data['flam'] * pylinear_flam_scale_fac

            # ---- Apply noise and get dummy noisy spectra
            noise_level = 0.03  # relative to signal
            # First assign noise to each point
            #host_flam_noisy, host_ferr = add_noise(host_flam, noise_level)
            #sn_flam_noisy, sn_ferr = add_noise(sn_flam, noise_level)

            host_ferr = noise_level * host_flam
            sn_ferr = noise_level * sn_flam

            # -------- Test figure for HOST
            """
            snr_host = host_flam / host_ferr
            print("Mean of signal to noise array:", np.mean(snr_host))

            fig = plt.figure()
            ax = fig.add_subplot()

            ax.set_xlabel(r'$\mathrm{\lambda\ [\AA]}$', fontsize=15)
            ax.set_ylabel(r'$\mathrm{f_\lambda\ [cgs]}$', fontsize=15)

            # plot extracted spectrum
            ax.plot(host_wav, host_flam, color='tab:brown', lw=2.5, \
                label='pyLINEAR extraction (sky noise added; no stat noise)')
            ax.fill_between(host_wav, host_flam - host_ferr, host_flam + host_ferr, \
                color='grey', alpha=0.5, zorder=1)

            m = model_host(host_wav, host_z, host_ms, host_age, host_tau, host_av, 4.0)

            # Only consider wavelengths where sensitivity is above 20%
            host_x0 = np.where( (host_wav >= grism_sens_wav[grism_wav_idx][0]  ) &
                                (host_wav <= grism_sens_wav[grism_wav_idx][-1] ) )[0]
            m = m[host_x0]

            a = np.nansum(host_flam[host_x0] * m / host_ferr[host_x0]**2) / np.nansum(m**2 / host_ferr[host_x0]**2)
            print("HOST a:", a)
            m = a*m
            print("HOST base model reduced chi2:", np.nansum( (m - host_flam[host_x0])**2 / host_ferr[host_x0]**2 ) / len(m))

            ax.plot(host_wav[host_x0], m, lw=1.0, color='tab:red', zorder=2, label='Downgraded model from mcmc code')

            # plot actual template passed into pylinear
            host_template = np.genfromtxt(h_path, dtype=None, names=True, encoding='ascii')
            ax.plot(host_template['lam'], 1e2 * host_template['flux'], lw=1.0, \
                color='tab:green', zorder=1, label='model given to pyLINEAR (+constant)')

            ax.set_xlim(9000, 20000)
            host_fig_ymin = np.min(host_flam)
            host_fig_ymax = np.max(host_flam)
            #ax.set_ylim(host_fig_ymin * 0.4, host_fig_ymax * 1.2)

            ax.legend(loc=0)

            plt.show()
            sys.exit(0)
            """

            # test figure for SN
            """
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)

            ax1.set_xlabel(r'$\mathrm{\lambda\ [\AA]}$', fontsize=15)
            ax1.set_ylabel(r'$\mathrm{f_\lambda\ [cgs]}$', fontsize=15)

            ax1.plot(sn_wav, sn_flam, lw=1.0, color='k', label='Obs SN data', zorder=2)
            ax1.fill_between(sn_wav, sn_flam - sn_ferr, sn_flam + sn_ferr, \
                color='grey', alpha=0.5, zorder=2)

            # Add host light
            #host_frac = 0.4  # some fraction to account for host contamination

            msn = model_sn(sn_wav, sn_z, sn_day, sn_av)

            sn_x0 = np.where( (sn_wav >= grism_sens_wav[grism_wav_idx][0]  ) &
                              (sn_wav <= grism_sens_wav[grism_wav_idx][-1] ) )[0]
            msn = msn[sn_x0]

            asn = np.nansum(sn_flam[sn_x0] * msn / sn_ferr[sn_x0]**2) / np.nansum(msn**2 / sn_ferr[sn_x0]**2)
            print("SN a:", asn)
            msn = asn * msn
            print("SN base model reduced chi2:", np.nansum( (msn - sn_flam[sn_x0])**2 / sn_ferr[sn_x0]**2 ) / len(msn))

            # plot spectrum without host light addition
            ax1.plot(sn_wav[sn_x0], msn, color='tab:green', label='SN template only', zorder=2)

            # Some dummy line
            #msn_and_line = msn + (sn_wav[sn_x0] * (5e-17 / 2500)  +  1e-17)
            #ax1.plot(sn_wav[sn_x0], msn_and_line, color='tab:red', label='SN template and line', zorder=2)

            #print("\nSN downgraded model spectrum mean:", np.nanmean(msn))
            #print("Obs host galaxy spectrum mean:", np.mean(host_flam))
            #print("Obs SN spectrum mean:", np.mean(sn_flam))
            #print("Host fraction manual:", host_frac)

            ax1.legend(loc=0)

            plt.show()
            sys.exit(0)

            # ADD host light in
            # The observed spectrum has the supernova light COMBINED with the host galaxy light
            new_mm = mm + host_frac * host_flam
            an = np.nansum(sn_flam * new_mm / sn_ferr**2) / np.nansum(new_mm**2 / sn_ferr**2)
            print("Vertical factor for manually scaled host light added SN model:", an)
            ax1.plot(sn_wav, an * new_mm, color='tab:cyan', label='Scaled host light added SN model', zorder=2)

            # Accomodate some shift in the host light
            # This is implemented by by giving a range of wavelengths 
            # within the SN spectrum that the host galaxy's light can 
            # contaminate. 
            # First plot observed host light
            ax1.plot(host_wav, host_flam_norm, color='darkblue', label='Obs host galaxy', alpha=0.8, zorder=1)

            # Shift is proportional to the "impact parameter" between the 
            # host and SN projected along the dispersion direction.
            # Define the shift as a fraction
            shift = -0.23  # this number is between -1.0 to approx +1.0
            # Now find initial and end wavelengths for shift
            shift_init_wav = sn_wav[0]  * (1 + shift)
            shift_end_wav  = sn_wav[-1] * (1 + shift)
            print("\nInitial and end wavelengths of shift:", shift_init_wav, shift_end_wav)
            print("Shift factor:", shift)
            print("These are the starting and ending SN spectrum wavelengths", end='\n')
            print("that are affected by the host light. If either wavelength is", end='\n')
            print("outside the grism/prism wavelength coverage then those wavelengths", end='\n')
            print("are, of course, superceded by the grism/prism wavelength coverage.", end='\n')
            shift_idx = np.where((sn_wav >= shift_init_wav) & (sn_wav <= shift_end_wav))[0]

            print("\nShift indices (i.e., host light contaminated indices in SN spectrum):", shift_idx)

            if shift > 0.0:
                host_end_wav = sn_wav[-1] / (1 + shift)
                print("Host end wav:", host_end_wav)
                host_shift_idx = np.where(host_wav <= host_end_wav)[0]
            elif shift < 0.0:
                host_init_wav = sn_wav[0] / (1 + shift)
                print("Host start wav:", host_init_wav)
                host_shift_idx = np.where(host_wav >= host_init_wav)[0]
            elif shift == 0.0:
                host_shift_idx = np.arange(len(host_wav))

            print("Host shift indices:", host_shift_idx)

            sn_fin = np.zeros(len(sn_wav))

            #mm_norm *= 5.0

            count = 0
            for v in range(len(sn_wav)):

                if v in shift_idx:
                    if shift > 0.0:
                    #if (sn_wav[v] >= shift_init_wav) and (sn_wav[v] <= shift_end_wav):
                        sn_fin[v] = mm_norm[v] + host_frac * host_flam_norm[count]
                    elif shift < 0.0:
                        sn_fin[v] = mm_norm[v] + host_frac * host_flam_norm[host_shift_idx[count]]
                else:
                    sn_fin[v] = mm_norm[v]

                count += 1

            print("Final SN spectrum", sn_fin)
            print("len final sn spec:", len(sn_fin))

            a_fin = np.nansum(sn_flam_norm * sn_fin / sn_ferr_norm**2) / np.nansum(sn_fin**2 / sn_ferr_norm**2)
            ax1.plot(sn_wav, a_fin * sn_fin, color='tab:red', label='Shifted host light added SN model', zorder=3)

            ax1.legend(loc=0)

            ax1.set_xlim(9000, 20000)

            plt.show()

            sys.exit(0)
            """

            # ----------------------- Test with explicit Metropolis-Hastings  ----------------------- #
            """
            print("\nRunning explicit Metropolis-Hastings...")
            N = 1000   #number of "timesteps"

            #host_frac_init = 0.05
            #r = np.array([0.01, 20, host_frac_init])  # initial position

            r = np.array([0.01, 10.0, 1.0, 2.0, 0.0, 2.0])
            print("Initial parameter vector:", r)

            #logp = logpost_sn(r, sn_wav, sn_flam, sn_ferr, host_flam)  # evaluating the probability at the initial guess
            jump_size_z = 0.01
            #jump_size_day = 1  # days
            #jump_size_host_frac = 0.01

            jump_size_ms = 0.1  # log(ms)
            jump_size_age = 0.1  # in gyr
            jump_size_tau = 0.1  # in gyr
            jump_size_av = 0.1  # magnitudes
            jump_size_lsf = 0.2

            logp = logpost_host(r, host_wav, host_flam, host_ferr)
            print("Initial guess log(probability):", logp)

            samples = []  #creating array to hold parameter vector with time
            accept = 0.
            logl_list = []
            chi2_list = []

            for i in range(N): #beginning the iteratitive loop

                print("\nMH Iteration", i, end='\n')
                #print("MH Iteration", i, end='\r')

                #rn0 = float(r[0] + jump_size_z * np.random.normal(size=1))
                #rn1 = int(r[1] + jump_size_day * np.random.choice([-1, 1]))
                #rn2 = float(r[2] + jump_size_host_frac * np.random.normal(size=1))
                #rn = np.array([rn0, rn1, rn2])

                rh0 = float(r[0] + jump_size_z * np.random.normal(size=1))
                rh1 = float(r[1] + jump_size_ms * np.random.normal(size=1))
                rh2 = float(r[2] + jump_size_age * np.random.normal(size=1))
                rh3 = float(r[3] + jump_size_tau * np.random.normal(size=1))
                rh4 = float(r[4] + jump_size_av * np.random.normal(size=1))
                rh5 = float(r[5] + jump_size_lsf * np.random.normal(size=1))

                rn = np.array([rh0, rh1, rh2, rh3, rh4, rh5])

                print("Proposal parameter vector", rn)
                
                #logpn = logpost_sn(rn, sn_wav, sn_flam, sn_ferr, host_flam)  #evaluating probability of proposal vector
                logpn = logpost_host(rn, host_wav, host_flam, host_ferr)  #evaluating probability of proposal vector
                print("Proposed parameter vector log(probability):", logpn)
                dlogL = logpn - logp
                print("dlogL:", dlogL)

                a = np.exp(dlogL)

                print("Ratio of probabilities at proposed to current position:", a)

                if a >= 1:   #always keep it if probability got higher
                    print("Will accept point since probability increased.")
                    logp = logpn
                    r = rn
                    accept+=1
        
                else:  #only keep it based on acceptance probability
                    print("Probability decreased. Will decide whether to keep point or not.")
                    u = np.random.rand()  #random number between 0 and 1
                    if u < a:  #only if proposal prob / previous prob is greater than u, then keep new proposed step
                        logp = logpn
                        r = rn
                        accept+=1
                        print("Point kept.")

                samples.append(r)  #update
                
                # -------
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(sn_wav, sn_flam, color='k')
                ax.fill_between(sn_wav, sn_flam - sn_ferr, sn_flam + sn_ferr, \
                    color='grey', alpha=0.5)

                m = model_sn(sn_wav, rn0, rn1, rn2, rn3, host_flam)
                print("Meam model:", np.mean(m))
                a = np.nansum(sn_flam * m / sn_ferr**2) / np.nansum(m**2 / sn_ferr**2)

                chi2 = np.nansum( (m-sn_flam)**2 / sn_ferr**2 )
                lnLike = -0.5 * np.nansum( (m-sn_flam)**2 / sn_ferr**2 )
                print("Chi2 for this position:", chi2)

                ax.text(x=0.65, y=0.95, s=r"$\chi^2 \,=\, $" + "{:.2e}".format(chi2), \
                    verticalalignment='top', horizontalalignment='left', transform=ax.transAxes, color='k', size=12)
                ax.text(x=0.65, y=0.87, s=r"$ln(L) \,=\, $"  + "{:.2e}".format(lnLike), \
                    verticalalignment='top', horizontalalignment='left', transform=ax.transAxes, color='k', size=12)

                ax.plot(sn_wav, a * m, color='tab:red')

                plt.show()
                plt.clf()
                plt.cla()
                plt.close()

                logl_list.append(logp)
                chi2_list.append(chi2)

            print("Finished explicit Metropolis-Hastings.\n")

            # Plotting results from explicit MH
            samples = np.array(samples)

            # plot trace
            #fig = plt.figure()
            #ax = fig.add_subplot(111)
            #ax.plot(samples[:,0], color='tab:red', label=r'$z$')
            #ax.plot(samples[:,1], color='tab:blue', label=r'$\mathrm{Day}$')
            #ax.plot(samples[:,2], color='tab:brown', label=r'$\mathrm{Host\ frac}$')
            #ax.legend(loc=0)
            #plt.show()

            # plot trace
            fig1, axes1 = plt.subplots(6, figsize=(10, 6), sharex=True)
            label_list_host = [r'$z$', r'$log(Ms/M_\odot)$', r'$Age [Gyr]$', r'$\tau [Gyr]$', r'$A_V [mag]$', 'LSF']

            for i in range(6):
                ax1 = axes1[i]
                ax1.plot(samples[:, i], "k", alpha=0.2)
                ax1.set_xlim(0, len(samples))
                ax1.set_ylabel(label_list_host[i])
                ax1.yaxis.set_label_coords(-0.1, 0.5)

            axes1[-1].set_xlabel("Step number")

            # using corner
            corner.corner(samples, bins=30, labels=label_list_host, \
                show_titles='True', plot_contours='True')#, truths=np.array([sn_z, sn_day, 0.0005]))
            plt.show()

            print("Acceptance Rate:", accept/N)

            sys.exit(0)
            """

            # ----------------------- Using MCMC to fit ----------------------- #
            print("\nRunning emcee...")

            # Set jump sizes # ONLY FOR INITIAL POSITION SETUP
            jump_size_z = 0.002
            jump_size_ms = 0.1  # log(ms)
            jump_size_age = 0.1  # in gyr
            jump_size_tau = 0.1  # in gyr
            jump_size_av = 0.1  # magnitudes
            jump_size_lsf = 0.2

            jump_size_day = 1  # days
            jump_size_host_frac = 0.02
            jump_size_lsf = 1.0  # angstrom

            # Define initial position and arguments required for emcee
            # The parameter vector is (redshift, ms, age, tau, av, lsf_sigma)
            # ms is actually log(ms/msol)
            # age in gyr and tau in gyr
            # dust parameter is av not tauv
            # lsf_sigma in angstroms

            args_sn = [sn_wav, sn_flam, sn_ferr, host_flam]
            args_host = [host_wav, host_flam, host_ferr]

            rhost_init = np.array([host_z, 10.5, host_age, host_tau, host_av, 1.0])  #get_optimal_fit(args_host, object_type='host')

            host_frac_init = 0.005
            rsn_init = np.array([0.01, 1, host_frac_init])  # redshift, day relative to peak, and fraction of host contamination

            # Setup dims and walkers
            nwalkers = 50
            ndim_host = 6
            ndim_sn  = 3

            # generating ball of walkers about initial position defined above
            pos_host = np.zeros(shape=(nwalkers, ndim_host))
            pos_sn = np.zeros(shape=(nwalkers, ndim_sn))

            z_init = np.linspace(0.001, 3.0, nwalkers)

            for i in range(nwalkers):

                # ---------- For HOST
                rh0 = float(rhost_init[0] + jump_size_z * np.random.normal(size=1))
                rh1 = float(rhost_init[1] + jump_size_ms * np.random.normal(size=1))
                rh2 = float(rhost_init[2] + jump_size_age * np.random.normal(size=1))
                rh3 = float(rhost_init[3] + jump_size_tau * np.random.normal(size=1))
                rh4 = float(rhost_init[4] + jump_size_av * np.random.normal(size=1))
                rh5 = float(rhost_init[5] + jump_size_lsf * np.random.normal(size=1))

                rh = np.array([rh0, rh1, rh2, rh3, rh4, rh5])

                pos_host[i] = rh

                # ---------- For SN
                rsn0 = float(rsn_init[0] + jump_size_z * np.random.normal(size=1))
                rsn1 = int(rsn_init[1] + jump_size_day * np.random.choice([-1, 1]))
                rsn2 = float(rsn_init[2] + jump_size_host_frac * np.random.normal(size=1))

                rsn = np.array([rsn0, rsn1, rsn2])

                pos_sn[i] = rsn
            
            # Set up truth arrays
            host_lsf = 4.0  # dummy
            truth_arr_host = np.array([host_z, host_ms, host_age, host_tau, host_av, host_lsf])
            truth_arr_sn = np.array([sn_z, sn_day, host_frac_init])

            # Labels for corner and trace plots
            label_list_host = [r'$z$', r'$log(Ms/M_\odot)$', r'$Age [Gyr]$', r'$\tau [Gyr]$', r'$A_V [mag]$', 'LSF']
            label_list_sn = [r'$z$', r'$Day$', r'$Host frac$']

            # Read previously run samples using pickle 
            #host_pickle = roman_slitless_dir + 'host_' + str(hostid) + '_emcee_sampler.pkl'
            host_pickle = roman_slitless_dir + 'host_' + str(hostid) + '_emcee_sampler.pkl'

            if os.path.isfile(host_pickle):
                read_pickle_make_plots('host', ndim_host, args_host, truth_arr_host, label_list_host, hostid, img_suffix)
                #read_pickle_make_plots('sn', ndim_sn, args_sn, truth_arr_sn, label_list_sn, segid, img_suffix)
            else:
                # Call emcee
                #run_emcee('sn', nwalkers, ndim_sn, logpost_sn, pos_sn, args_sn, segid)
                run_emcee('host', nwalkers, ndim_host, logpost_host, pos_host, args_host, hostid)

                read_pickle_make_plots('host', ndim_host, args_host, truth_arr_host, label_list_host, hostid, img_suffix)
                #read_pickle_make_plots('sn', ndim_sn, args_sn, truth_arr_sn, label_list_sn, segid, img_suffix)

            sys.exit(0)

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)



