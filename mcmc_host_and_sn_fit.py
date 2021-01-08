import numpy as np
from astropy.io import fits
import emcee
import corner
import scipy
from scipy.interpolate import griddata
from scipy.optimize import curve_fit

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
astropy_cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)

from multiprocessing import Pool
import pickle

import os
import sys
from functools import reduce
import time
import datetime as dt
import socket

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec

start = time.time()
print("Starting at:", dt.datetime.now())

# Assign directories and custom imports
home = os.getenv('HOME')
stacking_utils = home + '/Documents/GitHub/stacking-analysis-pears/util_codes/'
pears_figs_dir = home + '/Documents/pears_figs_data/'

roman_slitless_dir = home + "/Documents/GitHub/roman-slitless/"
ext_spectra_dir = home + "/Documents/roman_slitless_sims_results/"
template_dir = home + "/Documents/roman_slitless_sims_seds/"
roman_sims_seds = home + "/Documents/roman_slitless_sims_seds/"
emcee_diagnostics_dir = home + "/Documents/emcee_runs/emcee_diagnostics_roman/"

grism_sens_cat = np.genfromtxt(home + '/Documents/pylinear_ref_files/pylinear_config/Roman/roman_throughput_20190325.txt', \
    dtype=None, names=True, skip_header=3)

grism_sens_wav = grism_sens_cat['Wave'] * 1e4  # the text file has wavelengths in microns # needed in angstroms
grism_sens = grism_sens_cat['BAOGrism_1st']
grism_wav_idx = np.where(grism_sens > 0.25)

sys.path.append(stacking_utils)
import proper_and_lum_dist as cosmo
from dust_utils import get_dust_atten_model
from bc03_utils import get_age_spec

# Load in all models
# ------ THIS HAS TO BE GLOBAL!

# Read in SALT2 SN IA file from Lou
salt2_spec = np.genfromtxt(roman_sims_seds + "salt2_template_0.txt", \
    dtype=None, names=['day', 'lam', 'flam'], encoding='ascii')

if 'plffsn2' in socket.gethostname():
    extdir = '/astro/ffsn/Joshi/'
    modeldir = extdir + 'bc03_output_dir/'
else:
    extdir = '/Volumes/Joshi_external_HDD/Roman/'
    modeldir = extdir + 'bc03_output_dir/m62/'

assert os.path.isdir(modeldir)

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

"""
all_m22_models = []
tau_low = 0
tau_high = 20
for t in range(tau_low, tau_high, 1):
    tau_str = "{:.3f}".format(t).replace('.', 'p')
    a = np.load(modeldir + 'bc03_all_tau' + tau_str + '_m22_chab.npy', mmap_mode='r')
    all_m22_models.append(a)
    del a
# load models with large tau separately
#all_m22_models.append(np.load(modeldir + 'bc03_all_tau20p000_m22_chab.npy', mmap_mode='r'))
"""

print("Done loading all models. Time taken:", "{:.3f}".format(time.time()-start), "seconds.")

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

def logpost_host(theta, x, data, err, zprior, zprior_sigma):

    lp = logprior_host(theta, zprior, zprior_sigma)
    #print("Prior HOST:", lp)
    
    if not np.isfinite(lp):
        return -np.inf
    
    lnL = loglike_host(theta, x, data, err)

    #print("Likelihood HOST:", lnL)
    
    return lp + lnL

def logprior_host(theta, zprior, zprior_sigma):

    z, ms, age, logtau, av = theta
    #print("\nParameter vector given:", theta)

    if (0.0001 <= z <= 6.0):
    
        # Make sure model is not older than the Universe
        # Allowing at least 100 Myr for the first galaxies to form after Big Bang
        age_at_z = astropy_cosmo.age(z).value  # in Gyr
        age_lim = age_at_z - 0.1  # in Gyr

        if ((9.0 <= ms <= 12.5) and \
            (0.01 <= age <= age_lim) and \
            (-3.0 <= logtau <= 2.0) and \
            (0.0 <= av <= 5.0)):

            # Gaussian prior on redshift
            #ln_pz = np.log( 1.0 / (np.sqrt(2*np.pi)*zprior_sigma) ) - 0.5*(z - zprior)**2/zprior_sigma**2

            return 0.0
    
    return -np.inf

def loglike_host(theta, x, data, err):
    
    z, ms, age, logtau, av = theta

    y = model_host(x, z, ms, age, logtau, av)
    #print("Model func result:", y)

    # ------- Clip all arrays to where grism sensitivity is >= 25%
    # then get the log likelihood
    x0 = np.where( (x >= grism_sens_wav[grism_wav_idx][0]  ) &
                   (x <= grism_sens_wav[grism_wav_idx][-1] ) )[0]

    y = y[x0]
    data = data[x0]
    err = err[x0]
    x = x[x0]

    # ------- Vertical scaling factor
    #try:
    #alpha = np.nansum(data * y / err**2) / np.nansum(y**2 / err**2)
    #print("Alpha HOST:", "{:.2e}".format(alpha))
    #except RuntimeWarning:
    #    print("RuntimeWarning encountered.")
    #    print("Parameter vector given:", theta)
    #    sys.exit(0)

    #y = y * alpha

    # ------- log likelihood
    #chi2 = np.nansum( (y-data)**2/err**2 ) / len(y)
    lnLike = -0.5 * np.nansum( (y-data)**2/err**2 )# - 0.5 * np.nansum( np.log(2 * np.pi * err**2) )
    #stretch_fac = 10.0
    #lnLike = -0.5 * (1 + stretch_fac) * chi2

    #print("Pure chi2 term:", np.nansum( (y-data)**2/err**2 ))
    #print("Second error term:", np.nansum( np.log(2 * np.pi * err**2) ))
    #print("log likelihood HOST:", lnLike)

    """
    fig = plt.figure(figsize=(9,5))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'$\lambda\, [\mathrm{\AA}]$', fontsize=14)
    ax.set_ylabel(r'$f_\lambda\, [\mathrm{cgs}]$', fontsize=14)

    ax.plot(x, data, color='k')
    ax.fill_between(x, data - err, data + err, color='gray', alpha=0.5)

    ax.plot(x, y, color='firebrick')

    ax.set_xscale('log')
    plt.show()
    #sys.exit(0)
    """

    return lnLike

def model_host(x, z, ms, age, logtau, av):
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

    """
    metals = 0.02

    # Get the metallicity in the format that BC03 needs
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
    """

    tau = 10**logtau  # logtau is log of tau in gyr

    #print("log(tau [Gyr]):", logtau)
    #print("Tau [Gyr]:", tau)
    #print("Age [Gyr]:", age)

    if tau < 20.0:

        tau_int_idx = int((tau - int(np.floor(tau))) * 1e3)
        age_idx = np.argmin(abs(model_ages - age*1e9))
        model_idx = tau_int_idx * len(model_ages)  +  age_idx

        models_taurange_idx = np.argmin(abs(np.arange(tau_low, tau_high, 1) - int(np.floor(tau))))
        models_arr = all_m62_models[models_taurange_idx]

        #print("Tau int and age index:", tau_int_idx, age_idx)
        #print("Tau and age from index:", models_taurange_idx+tau_int_idx/1e3, model_ages[age_idx]/1e9)
        #print("Model tau range index:", models_taurange_idx)

    elif tau >= 20.0:
        
        logtau_arr = np.arange(1.30, 2.01, 0.01)
        logtau_idx = np.argmin(abs(logtau_arr - logtau))

        age_idx = np.argmin(abs(model_ages - age*1e9))
        model_idx = logtau_idx * len(model_ages) + age_idx

        models_arr = all_m62_models[-1]

        #print("logtau and age index:", logtau_idx, age_idx)
        #print("Tau and age from index:", 10**(logtau_arr[logtau_idx]), model_ages[age_idx]/1e9)

    #print("Model index:", model_idx)

    model_llam = models_arr[model_idx]

    """
    tauv = 0.0
    metallicity = 0.02
    model_llam = get_template(np.log10(age * 1e9), tau, tauv, metallicity, \
        log_age_arr, metal_arr, tau_gyr_arr, tauv_arr, \
        model_lam_grid, model_grid)
    model_lam = model_lam_grid
    model_lam, model_llam = remove_emission_lines(model_lam, model_llam)
    """

    # ------ Apply dust extinction
    model_dusty_llam = get_dust_atten_model(model_lam, model_llam, av)
    #print("Model dusty llam:", model_dusty_llam)

    # ------ Multiply luminosity by stellar mass
    model_dusty_llam = model_dusty_llam * 10**ms
    #print("Model dusty llam after ms:", model_dusty_llam)

    # ------ Apply redshift
    model_lam_z, model_flam_z = cosmo.apply_redshift(model_lam, model_dusty_llam, z)
    Lsol = 3.826e33
    model_flam_z = Lsol * model_flam_z
    #print("Model flam:", model_flam_z)

    # ------ Apply LSF
    model_lsfconv = scipy.ndimage.gaussian_filter1d(input=model_flam_z, sigma=1.0)
    #print("Model lsf conv:", model_lsfconv)

    # ------ Downgrade to grism resolution
    """
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
    """

    model_mod = griddata(points=model_lam_z, values=model_lsfconv, xi=x)
    #print("Model mod:", model_mod)

    #model_mod /= np.nanmedian(model_mod)
    #print("Model median:", np.nanmedian(model_mod))

    return model_mod

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

def loglike_sn(theta, x, data, err):
    
    z, day, av = theta

    y = model_sn(x, z, day, av)
    #print("Model SN func result:", y)

    # ------- Clip all arrays to where grism sensitivity is >= 25%
    # then get the log likelihood
    x0 = np.where( (x >= grism_sens_wav[grism_wav_idx][0]  ) &
                   (x <= grism_sens_wav[grism_wav_idx][-1] ) )[0]

    y = y[x0]
    data = data[x0]
    err = err[x0]
    x = x[x0]

    # ------- Vertical scaling factor
    alpha = np.nansum(data * y / err**2) / np.nansum(y**2 / err**2)
    #print("Alpha SN:", "{:.2e}".format(alpha))

    y = y * alpha

    lnLike = -0.5 * np.nansum( (y-data)**2/err**2 ) #  +  np.log(2 * np.pi * err**2))

    #print("Chi2 term:", np.sum((y-data)**2/err**2))
    #print("Second loglikelihood term:", np.log(2 * np.pi * err**2))
    #print("ln(likelihood) SN", lnLike)

    """
    fig = plt.figure(figsize=(9,5))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'$\lambda\, [\mathrm{\AA}]$', fontsize=14)
    ax.set_ylabel(r'$f_\lambda\, [\mathrm{cgs}]$', fontsize=14)

    ax.plot(x, data, color='k')
    ax.fill_between(x, data - err, data + err, color='gray', alpha=0.5)

    ax.plot(x, y, color='firebrick')

    ax.set_xscale('log')
    plt.show()
    #sys.exit(0)
    """
    
    return lnLike

def logprior_sn(theta):

    z, day, av = theta

    if ( 0.0001 <= z <= 6.0  and  -19 <= day <= 50  and  0.0 <= av <= 5.0):
        return 0.0
    
    return -np.inf

def logpost_sn(theta, x, data, err):

    lp = logprior_sn(theta)

    #print("SN prior:", lp)
    
    if not np.isfinite(lp):
        return -np.inf
    
    lnL = loglike_sn(theta, x, data, err)

    #print("SN log(likelihood):", lnL)
    
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

    #sn_mod = griddata(points=sn_lam_z, values=sn_flam_z, xi=x)

    # ------ combine host light
    # some fraction to account for host contamination
    # This fraction is a free parameter
    #sn_flam_hostcomb = sn_mod  +  host_frac * host_flam

    return sn_mod

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

        #err_arr[k] = np.sqrt(spec_noise[k])
        err_arr[k] = noise_level * spec_noise[k]

    return spec_noise, err_arr

def get_autocorr_time(sampler):

    # OLD code block when I didnt know I could 
    # just use tol=0 and avoid the autocorr error
    """
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
    """

    tau = sampler.get_autocorr_time(tol=0)
    print("Tau:", tau)

    return tau

def run_emcee(object_type, nwalkers, ndim, logpost, pos, args_obj, objid):

    print("Running on:", object_type, "with ID:", objid)

    # ----------- Set up the HDF5 file to incrementally save progress to
    emcee_savefile = emcee_diagnostics_dir + object_type + '_' + str(objid) + '_emcee_sampler.h5'
    backend = emcee.backends.HDFBackend(emcee_savefile)
    backend.reset(nwalkers, ndim)

    # ----------- Emcee 
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logpost, args=args_obj, pool=pool, backend=backend)
        #moves=emcee.moves.MHmove())
        sampler.run_mcmc(pos, 1000, progress=True)

    # ----------- Also save the final result as a pickle dump
    pickle.dump(sampler, open(emcee_savefile.replace('.h5','.pkl'), 'wb'))

    print("Done with fitting.")
    print("Mean acceptance Fraction:", np.mean(sampler.acceptance_fraction), "\n")

    return None

def read_pickle_make_plots(object_type, ndim, args_obj, truth_arr, label_list, objid, img_suffix):

    checkdir = '' #'generic_11112020/'
    pkl_path = emcee_diagnostics_dir + checkdir + object_type + '_' + str(objid) + '_emcee_sampler.pkl'
    #sampler = pickle.load(open(pkl_path, 'rb'))

    h5_path = pkl_path.replace('.pkl','.h5')
    sampler = emcee.backends.HDFBackend(h5_path)

    samples = sampler.get_chain()
    print(f"{bcolors.CYAN}\nRead in sampler:", h5_path, f"{bcolors.ENDC}")
    print("Samples shape:", samples.shape)

    #reader = emcee.backends.HDFBackend(pkl_path.replace('.pkl', '.h5'))
    #samples = reader.get_chain()
    #tau = reader.get_autocorr_time(tol=0)

    # Get autocorrelation time
    # Discard burn-in. You do not want to consider the burn in the corner plots/estimation.
    tau = get_autocorr_time(sampler)
    burn_in = int(2 * np.max(tau))
    thinning_steps = int(0.5 * np.min(tau))

    print(f"{bcolors.CYAN}")
    print("Average Tau:", np.mean(tau))
    print("Burn-in:", burn_in)
    print("Thinning steps:", thinning_steps)
    print(f"{bcolors.ENDC}")

    # plot trace
    fig1, axes1 = plt.subplots(ndim, figsize=(10, 6), sharex=True)

    for i in range(ndim):
        ax1 = axes1[i]
        ax1.plot(samples[:, :, i], "k", alpha=0.05)
        ax1.set_xlim(0, len(samples))
        ax1.set_ylabel(label_list[i])
        ax1.yaxis.set_label_coords(-0.1, 0.5)

    axes1[-1].set_xlabel("Step number")

    fig1.savefig(emcee_diagnostics_dir + 'emcee_trace_' + object_type + '_' + str(objid) + '_' + img_suffix + '.pdf', \
        dpi=200, bbox_inches='tight')

    # Create flat samples
    flat_samples = sampler.get_chain(discard=burn_in, thin=thinning_steps, flat=True)
    print("\nFlat samples shape:", flat_samples.shape)

    # plot corner plot

    # assign ranges to parameters
    # These need to be built from the corner quantile arrays.
    # For the test galaxies truth values can be used.
    # However, it seems like you cannot input the truth 
    # value (+- some padding) directly into the tuples in the range list.
    # For some unknown reason corner freezes when that is done.
    #range_list = [(1.945, 1.96), (14.0, 17.0), (0.0, 5.0), (-0.8, 2.0), (0.0, 1.6)]  # for 207
    #range_list = [(0.0, 1.5), (10.5, 15.5), (0.0, 12.0), (-2.5, 2.0), (1.0, 5.0)]  # for 475
    #range_list = [(1.585, 1.6), (12.5, 15.5), (0.0, 4.5), (-0.4, 2.0), (0.0, 2.2)]  # for 548
    #range_list = [(0.0, 2.0), (10.2, 15.5), (0.0, 10.0), (-2.2, 2.0), (0.0, 2.2)]  # for 755

    if object_type == 'host':
        cq_z = corner.quantile(x=flat_samples[:, 0], q=[0.16, 0.5, 0.84])
        cq_ms = corner.quantile(x=flat_samples[:, 1], q=[0.16, 0.5, 0.84])
        cq_age = corner.quantile(x=flat_samples[:, 2], q=[0.16, 0.5, 0.84])
        cq_tau = corner.quantile(x=flat_samples[:, 3], q=[0.16, 0.5, 0.84])
        cq_av = corner.quantile(x=flat_samples[:, 4], q=[0.16, 0.5, 0.84])

        # print parameter estimates
        print(f"{bcolors.CYAN}")
        print("Parameter estimates:")
        print("Redshift: ", cq_z)
        print("Stellar mass [log(M/M_sol)]:", cq_ms)
        print("Age [Gyr]: ", cq_age)
        print("log SFH Timescale [Gyr]: ", cq_tau)
        print("Visual extinction [mag]:", cq_av)
        print(f"{bcolors.ENDC}")

    #print(f"{bcolors.WARNING}\nUsing hardcoded ranges in corner plot.{bcolors.ENDC}")
    fig = corner.corner(flat_samples, quantiles=[0.16, 0.5, 0.84], labels=label_list, \
        label_kwargs={"fontsize": 14}, show_titles='True', title_kwargs={"fontsize": 14}, truths=truth_arr, \
        verbose=True, truth_color='tab:red', smooth=0.8, smooth1d=0.8)#, \
    #range=[(1.9525, 1.9535), (10.5, 11.5), (1.2, 2.4), (0.5, 1.3), (0.4, 0.7)])

    #corner_axes = np.array(fig.axes).reshape((ndim, ndim))

    # redshift is the first axis
    #corner_axes[0, 0].set_title()

    fig.savefig(emcee_diagnostics_dir + 'corner_' + object_type + '_' + str(objid) + '_' + img_suffix + '.pdf', \
        dpi=200, bbox_inches='tight')

    # ------------ Plot 100 random models from the parameter space
    # first pull out required stuff from args
    wav = args_obj[0]
    flam = args_obj[1]
    ferr = args_obj[2]

    fig3 = plt.figure(figsize=(9,4))
    ax3 = fig3.add_subplot(111)

    ax3.set_xlabel(r'$\mathrm{\lambda\ [\AA]}$', fontsize=15)
    ax3.set_ylabel(r'$\mathrm{f_\lambda\ [cgs]}$', fontsize=15)

    model_count = 0
    ind_list = []

    while model_count <= 100:

        ind = int(np.random.randint(len(flat_samples), size=1))
        ind_list.append(ind)
        # make sure sample has correct shape
        sample = flat_samples[ind]
        sample = sample.reshape(5)
        #print("\nAt random index:", ind)

        # Check that LSF is not negative
        #if sample[-1] < 0.0:
        #    print("Negative LSF ....")
        #    sample[-1] = 1.0

        # Check that the model is within +-1 sigma
        # of value inferred by corner contours
        model_z = sample[0]
        model_ms = sample[1]
        model_age = sample[2]
        model_tau = sample[3]
        model_av = sample[4]

        if (model_z >= cq_z[0]) and (model_z <= cq_z[2]) and \
           (model_ms >= cq_ms[0]) and (model_ms <= cq_ms[2]) and \
           (model_age >= cq_age[0]) and (model_age <= cq_age[2]) and \
           (model_tau >= cq_tau[0]) and (model_tau <= cq_tau[2]) and \
           (model_av >= cq_av[0]) and (model_av <= cq_av[2]):

            if object_type == 'host':
                m = model_host(wav, sample[0], sample[1], sample[2], sample[3], sample[4])
            elif object_type == 'sn':
                m = model_sn(wav, sample[0], sample[1], sample[2])

            # ------- Clip all arrays to where grism sensitivity is >= 25%
            x0 = np.where( (wav >= grism_sens_wav[grism_wav_idx][0]  ) &
                           (wav <= grism_sens_wav[grism_wav_idx][-1] ) )[0]

            m = m[x0]
            wav = wav[x0]
            flam = flam[x0]
            ferr = ferr[x0]

            ax3.plot(wav, m, color='royalblue', lw=0.8, alpha=0.05, zorder=2)

            # ------------------------ print info
            #lnL = logpost_host(sample, wav, flam, ferr)
            #if sample[0] > 1.97:
            #    print(f"{bcolors.FAIL}")
            #    print("With sample:", sample)
            #    print("Log likelihood for this sample:", lnL)
            #    print(f"{bcolors.ENDC}")

            #    ax3.plot(wav, flam, color='k', zorder=3)
            #    ax3.fill_between(wav, flam - ferr, flam + ferr, color='gray', alpha=0.5, zorder=3)
            #    plt.show()
            #    sys.exit(0)

            #else:
            #    print("With sample:", sample)
            #    print("Log likelihood for this sample:", lnL)

            model_count += 1

    print("\nList of randomly chosen indices:", ind_list)

    ax3.plot(wav, flam, color='k', lw=2.0, zorder=1)
    ax3.fill_between(wav, flam - ferr, flam + ferr, color='gray', alpha=0.5, zorder=1)

    fig3.savefig(emcee_diagnostics_dir + 'emcee_overplot_' + object_type + '_' + str(objid) + '_' + img_suffix + '.pdf', \
        dpi=200, bbox_inches='tight')

    return None

def get_optimal_fit(args_obj, object_type):

    print("Running scipy.optimize.curve_fit to determine initial position.")

    # first pull out required stuff from args
    wav = args_obj[0]
    flam = args_obj[1]
    ferr = args_obj[2]

    x0 = np.where( (wav >= grism_sens_wav[grism_wav_idx][0]  ) &
                   (wav <= grism_sens_wav[grism_wav_idx][-1] ) )[0]

    wav = wav[x0]
    flam = flam[x0]
    ferr = ferr[x0]

    flam_norm = flam / np.median(flam)
    ferr_norm = ferr / np.median(ferr)

    # Initial guess
    # Based on eyeballing the spectrum
    p0 = [1.95, 1.4, 12.0, 0.5]

    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(wav, flam_norm)
    ax.plot(wav, model_host_norm(wav, *p0))
    plt.show()
    sys.exit(0)
    """

    popt, pcov = curve_fit(f=model_host_norm, xdata=wav, ydata=flam_norm, p0=p0, sigma=ferr_norm, \
        bounds=[(1.8, 0, 0, 0), (2.0, 5, 20, 5)])

    print(popt)
    print(pcov)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(wav, flam_norm)
    ax.plot(wav, model_host_norm(wav, *popt))
    plt.show()
    sys.exit(0)

    return np.array([best_z, best_age, best_tau, best_av, 1.0])

def main():

    print(f"{bcolors.WARNING}")
    print("* * * *   [WARNING]: model has worse resolution than data in NIR. np.mean() will result in nan. Needs fixing.   * * * *")
    print("* * * *   [TODO]: When plotting \"best-fit\" models, only plot those that are within +- 1-sigma of the values from corner.  * * * *")
    print(f"{bcolors.ENDC}")

    ext_root = "romansim1"
    img_suffix = 'Y106_11_1'

    # Read in sed.lst
    sedlst_header = ['segid', 'sed_path']
    sedlst_path = roman_slitless_dir + 'pylinear_lst_files/' + 'sed_plffsn2_' + img_suffix + '.lst'
    sedlst = np.genfromtxt(sedlst_path, dtype=None, names=sedlst_header, encoding='ascii')
    print("Read in sed.lst from:", sedlst_path)

    # Read in the extracted spectra
    ext_spec_filename = ext_spectra_dir + 'plffsn2_run_jan5/' + ext_root + '_ext_x1d.fits'
    ext_hdu = fits.open(ext_spec_filename)
    print("Read in extracted spectra from:", ext_spec_filename)

    # Set pylinear f_lambda scaling factor
    pylinear_flam_scale_fac = 1e-17

    # This will come from detection on the direct image
    # For now this comes from the sedlst generation code
    # For Y106_11_1
    host_segids = np.array([475, 755, 548, 207])
    sn_segids = np.array([481, 753, 547, 241])

    for i in range(700, len(sedlst)):

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
            print("Host input tau [Gyr] and log(tau [Gyr]):", host_tau, "{:.3f}".format(np.log10(host_tau)))
            print("Host input metallicity:", host_met)
            print("Host input Av [mag]:", host_av)

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
            #host_flam, host_ferr = add_noise(host_flam, noise_level)
            #sn_flam, sn_ferr = add_noise(sn_flam, noise_level)

            host_ferr = noise_level * host_flam
            sn_ferr = noise_level * sn_flam

            # Manual mod to check if it'll get the correct 
            # stellar mass if the flux scaling is correct.
            # for galaxy 207 from dec7 run
            #host_flam /= 259.2
            #host_ferr /= 259.2

            # for galaxy 475 from nov30 run
            #host_flam /= 40.2
            #host_ferr /= 40.2

            # for galaxy 548 from nov30 run
            #host_flam /= 700
            #host_ferr /= 700

            # for galaxy 755 from nov30 run
            #host_flam /= 85.5
            #host_ferr /= 85.5

            #host_flam_norm = host_flam / np.median(host_flam)
            #host_ferr_norm = noise_level * host_flam_norm

            # -------- Test figure for HOST
            fig = plt.figure(figsize=(10,5))
            ax = fig.add_subplot()

            ax.set_xlabel(r'$\mathrm{\lambda\ [\AA]}$', fontsize=15)
            ax.set_ylabel(r'$\mathrm{f_\lambda\ [cgs]}$', fontsize=15)

            # plot extracted spectrum
            ax.plot(host_wav, host_flam, color='k', lw=2, \
                label='pyLINEAR extraction (div. const.) (sky noise added; no stat noise)')
            ax.fill_between(host_wav, host_flam - host_ferr, host_flam + host_ferr, \
                color='grey', alpha=0.5, zorder=1)

            m = model_host(host_wav, host_z, host_ms, host_age, np.log10(host_tau), host_av)

            # Only consider wavelengths where sensitivity is above 20%
            host_x0 = np.where( (host_wav >= grism_sens_wav[grism_wav_idx][0]  ) &
                                (host_wav <= grism_sens_wav[grism_wav_idx][-1] ) )[0]
            m = m[host_x0]

            a = np.nansum(host_flam[host_x0] * m / host_ferr[host_x0]**2) / np.nansum(m**2 / host_ferr[host_x0]**2)
            print("HOST a:", "{:.4e}".format(a))
            #m = a*m
            chi2_good = np.nansum( (m - host_flam[host_x0])**2 / host_ferr[host_x0]**2 )# / len(m)
            print("HOST base model chi2:", chi2_good)

            ax.plot(host_wav[host_x0], m, lw=1.0, color='tab:red', zorder=2, label='Downgraded model from mcmc code')

            # plot actual template passed into pylinear
            if 'plffsn2' not in socket.gethostname():
                h_path = h_path.replace('/home/bajoshi/', '/Users/baj/')
            host_template = np.genfromtxt(h_path, dtype=None, names=True, encoding='ascii')
            ax.plot(host_template['lam'], host_template['flux'], lw=1.0, \
                color='tab:green', zorder=1, label='model given to pyLINEAR')

            ax.set_xlim(9000, 20000)
            host_fig_ymin = np.min(host_flam)
            host_fig_ymax = np.max(host_flam)
            ax.set_ylim(host_fig_ymin * 0.4, host_fig_ymax * 1.2)

            ax.legend(loc=0, fontsize=12, frameon=False)

            plt.show()
            sys.exit(0)

            """
            # plot some other template that is NOT a good fit
            bad_model = model_host(host_wav, 1.95, 1.0, 13.0, 1.0, 3.0)
            bad_model = bad_model[host_x0]

            ab = np.nansum(host_flam[host_x0] * bad_model / host_ferr[host_x0]**2) / np.nansum(bad_model**2 / host_ferr[host_x0]**2)
            print(f"{bcolors.WARNING}\nHOST a for bad model:", "{:.3e}".format(ab))
            bad_model = ab*bad_model
            chi2_bad = np.nansum( (bad_model - host_flam[host_x0])**2 / host_ferr[host_x0]**2 )# / len(bad_model)
            print("HOST base model reduced chi2 for bad model:", chi2_bad)
            print(f"{bcolors.ENDC}")
            ax.plot(host_wav[host_x0], bad_model, lw=1.0, color='magenta', zorder=2, label='Representative bad model (downgraded)')

            # another model worse than the bad model
            worse_model = model_host(host_wav, 1.9, 3.0, 0.5, 0.1, 3.0)
            worse_model = worse_model[host_x0]

            aw = np.nansum(host_flam[host_x0] * worse_model / host_ferr[host_x0]**2) / np.nansum(worse_model**2 / host_ferr[host_x0]**2)
            print(f"{bcolors.FAIL}\nHOST a for bad model:", "{:.3e}".format(aw))
            worse_model = aw*worse_model
            chi2_worse = np.nansum( (worse_model - host_flam[host_x0])**2 / host_ferr[host_x0]**2 )# / len(worse_model)
            print("HOST base model reduced chi2 for worse model:", chi2_worse)
            print(f"{bcolors.ENDC}")
            ax.plot(host_wav[host_x0], worse_model, lw=1.0, color='slateblue', zorder=2, label='Representative worse model (downgraded)')

            # Get ln(likelihood) for all models
            lnl_good = logpost_host([host_z, host_age, host_tau, host_av, 3.0], host_wav, host_flam, host_ferr)
            lnl_bad = logpost_host([1.95, 1.0, 13.0, 1.0, 3.0], host_wav, host_flam, host_ferr)
            lnl_worse = logpost_host([1.9, 3.0, 0.5, 0.1, 3.0], host_wav, host_flam, host_ferr)

            # Info to plot
            ax.text(x=0.45, y=0.2, s=r"$\chi^2_\mathrm{good} \,=\, $" + "{:.2f}".format(chi2_good), \
                    verticalalignment='top', horizontalalignment='left', transform=ax.transAxes, color='tab:red', size=12)
            ax.text(x=0.45, y=0.14, s=r"$\chi^2_\mathrm{bad} \,=\, $" + "{:.2f}".format(chi2_bad), \
                    verticalalignment='top', horizontalalignment='left', transform=ax.transAxes, color='magenta', size=12)
            ax.text(x=0.45, y=0.08, s=r"$\chi^2_\mathrm{worse} \,=\, $" + "{:.2f}".format(chi2_worse), \
                    verticalalignment='top', horizontalalignment='left', transform=ax.transAxes, color='slateblue', size=12)

            ax.text(x=0.7, y=0.2, s=r"$\mathrm{ln}(\mathcal{L}_\mathrm{good}) \,=\, $" + "{:.2f}".format(lnl_good), \
                    verticalalignment='top', horizontalalignment='left', transform=ax.transAxes, color='tab:red', size=12)
            ax.text(x=0.7, y=0.14, s=r"$\mathrm{ln}(\mathcal{L}_\mathrm{bad}) \,=\, $" + "{:.2f}".format(lnl_bad), \
                    verticalalignment='top', horizontalalignment='left', transform=ax.transAxes, color='magenta', size=12)
            ax.text(x=0.7, y=0.08, s=r"$\mathrm{ln}(\mathcal{L}_\mathrm{worse}) \,=\, $" + "{:.2f}".format(lnl_worse), \
                    verticalalignment='top', horizontalalignment='left', transform=ax.transAxes, color='slateblue', size=12)

            ax.set_xlim(9000, 20000)
            host_fig_ymin = np.min(host_flam)
            host_fig_ymax = np.max(host_flam)
            ax.set_ylim(host_fig_ymin * 0.2, host_fig_ymax * 1.8)

            ax.legend(loc=0, frameon=False)

            fig.savefig(roman_slitless_dir + 'galaxy_model_lnL_examples.pdf', dpi=200, bbox_inches='tight')

            # Now save all these to an ascii file
            fh = open(roman_slitless_dir + 'galaxy_model_lnL_examples.txt', 'w')
            fh.write("# log likelihood values for the three models given below." + "\n")
            fh.write("# for the good model, ln(L) = " + "{:.2f}".format(lnl_good) + "\n")
            fh.write("# for the bad model, ln(L) = " + "{:.2f}".format(lnl_bad) + "\n")
            fh.write("# for the worse model, ln(L) = " + "{:.2f}".format(lnl_worse) + "\n")
            fh.write("#  lam  flam  ferr  flam_good  flam_bad  flam_worse" + "\n")

            for l in range(len(m)):
                fh.write( "{:.3f}".format(host_wav[host_x0][l]) + "  " + \
                          "{:.3e}".format(host_flam[host_x0][l]) + "  " + \
                          "{:.3e}".format(host_ferr[host_x0][l]) + "  " + \
                          "{:.3e}".format(m[l]) + "  " + \
                          "{:.3e}".format(bad_model[l]) + "  " + \
                          "{:.3e}".format(worse_model[l]) )
                fh.write("\n")

            fh.close()

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
            N = 10000   #number of "timesteps"

            #host_frac_init = 0.05
            #r = np.array([0.01, 20, host_frac_init])  # initial position

            r = np.array([1.9, 1.5, 12.4, 0.5, 2.0])
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
            #logl_list = []
            #chi2_list = []

            for i in range(200): #beginning the iteratitive loop

                print("\nMH Iteration", i, end='\n')
                #print("MH Iteration", i, end='\r')

                #rn0 = float(r[0] + jump_size_z * np.random.normal(size=1))
                #rn1 = int(r[1] + jump_size_day * np.random.choice([-1, 1]))
                #rn2 = float(r[2] + jump_size_host_frac * np.random.normal(size=1))
                #rn = np.array([rn0, rn1, rn2])

                rh0 = float(r[0] + jump_size_z * np.random.normal(size=1))
                rh1 = float(r[1] + jump_size_age * np.random.normal(size=1))
                rh2 = float(r[2] + jump_size_tau * np.random.normal(size=1))
                rh3 = float(r[3] + jump_size_av * np.random.normal(size=1))
                rh4 = float(r[4] + jump_size_lsf * np.random.normal(size=1))

                rn = np.array([rh0, rh1, rh2, rh3, rh4])

                #print("Proposal parameter vector", rn)
                
                #logpn = logpost_sn(rn, sn_wav, sn_flam, sn_ferr, host_flam)  #evaluating probability of proposal vector
                logpn = logpost_host(rn, host_wav, host_flam, host_ferr)  #evaluating probability of proposal vector
                #print("Proposed parameter vector log(probability):", logpn)
                dlogL = logpn - logp
                #print("dlogL:", dlogL)

                a = np.exp(dlogL)

                #print("Ratio of probabilities at proposed to current position:", a)

                if a >= 1:   #always keep it if probability got higher
                    #print("Will accept point since probability increased.")
                    logp = logpn
                    r = rn
                    accept+=1
        
                else:  #only keep it based on acceptance probability
                    #print("Probability decreased. Will decide whether to keep point or not.")
                    u = np.random.rand()  #random number between 0 and 1
                    if u < a:  #only if proposal prob / previous prob is greater than u, then keep new proposed step
                        logp = logpn
                        r = rn
                        accept+=1
                        #print("Point kept.")

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
            fig1, axes1 = plt.subplots(5, figsize=(10, 6), sharex=True)
            label_list_host = [r'$z$', r'$log(Ms/M_\odot)$', r'$Age [Gyr]$', r'$\tau [Gyr]$', r'$A_V [mag]$', 'LSF']

            for i in range(5):
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
            jump_size_z = 0.01
            jump_size_ms = 0.1  # log(ms)
            jump_size_age = 0.1  # in gyr
            jump_size_logtau = 0.01  # tau in gyr
            jump_size_av = 0.1  # magnitudes
            jump_size_lsf = 0.2

            jump_size_day = 2  # days

            # Define initial position and arguments required for emcee
            # The parameter vector is (redshift, ms, age, tau, av, lsf_sigma)
            # ms is actually log(ms/msol)
            # age in gyr and tau in gyr
            # dust parameter is av not tauv
            # lsf_sigma in angstroms

            #rhost_init = get_optimal_fit(args_host, object_type='host')
            #sys.exit(0)

            if hostid == 207:
                zprior = 1.95
                rhost_init = np.array([zprior, 11.0, 2.5, 1.0, 0.5])
            elif hostid == 475:
                zprior = 0.44
                rhost_init = np.array([zprior, 11.25, 2.0, 0.5, 3.5])
            elif hostid == 548:
                zprior = 1.59
                rhost_init = np.array([zprior, 10.7, 3.5, 1.0, 0.0])
            elif hostid == 755:
                zprior = 0.92
                rhost_init = np.array([zprior, 11.1, 2.0, 0.5, 0.0])

            zprior_sigma = 0.02  # standard deviation for the Gaussian prior on redshift

            args_sn = [sn_wav, sn_flam, sn_ferr]
            args_host = [host_wav, host_flam, host_ferr, zprior, zprior_sigma]

            print(f"{bcolors.GREEN}", "Starting position for HOST from where ball of walkers will be generated:\n", rhost_init, f"{bcolors.ENDC}")
            print("logpost at starting position for HOST galaxy:", logpost_host(rhost_init, host_wav, host_flam, host_ferr, zprior, zprior_sigma))

            rtrue = np.array([host_z, host_ms, host_age, np.log10(host_tau), host_av])
            print("logpost at true position for HOST galaxy:", logpost_host(rtrue, host_wav, host_flam, host_ferr, zprior, zprior_sigma))
            #print(f"{bcolors.WARNING}", "Lower log(posterior) probability at true position likely due to metallicity difference.", f"{bcolors.ENDC}")

            rsn_init = np.array([1.8, 1, 0.2])  # redshift, day relative to peak, and dust extinction
            print(f"{bcolors.GREEN}Starting position for SN from where ball of walkers will be generated:\n", rsn_init, f"{bcolors.ENDC}")
            print("logpost at starting position for SN:", logpost_sn(rsn_init, sn_wav, sn_flam, sn_ferr))

            # Setup dims and walkers
            nwalkers = 300
            ndim_host = 5
            ndim_sn  = 3

            # generating ball of walkers about initial position defined above
            pos_host = np.zeros(shape=(nwalkers, ndim_host))
            pos_sn = np.zeros(shape=(nwalkers, ndim_sn))

            #z_init = np.linspace(0.001, 3.0, nwalkers)

            for i in range(nwalkers):

                # ---------- For HOST
                rh0 = float(rhost_init[0] + jump_size_z * np.random.normal(size=1))
                rh1 = float(rhost_init[1] + jump_size_ms * np.random.normal(size=1))
                rh2 = float(rhost_init[2] + jump_size_age * np.random.normal(size=1))
                rh3 = float(rhost_init[3] + jump_size_logtau * np.random.normal(size=1))
                rh4 = float(rhost_init[4] + jump_size_av * np.random.normal(size=1))

                rh = np.array([rh0, rh1, rh2, rh3, rh4])

                pos_host[i] = rh

                # ---------- For SN
                rsn0 = float(rsn_init[0] + jump_size_z * np.random.normal(size=1))
                rsn1 = int(rsn_init[1] + jump_size_day * np.random.normal(size=1))
                rsn2 = float(rsn_init[2] + jump_size_av * np.random.normal(size=1))

                rsn = np.array([rsn0, rsn1, rsn2])

                pos_sn[i] = rsn
            
            # Set up truth arrays
            truth_arr_host = np.array([host_z, host_ms, host_age, np.log10(host_tau), host_av])
            truth_arr_sn = np.array([sn_z, sn_day, sn_av])

            # Labels for corner and trace plots
            label_list_host = [r'$z$', r'$\mathrm{log(M_s/M_\odot)}$', r'$\mathrm{Age\, [Gyr]}$', \
            r'$\mathrm{\log(\tau\, [Gyr])}$', r'$A_V [mag]$'] 
            label_list_sn = [r'$z$', r'$Day$', r'$A_V [mag]$']

            # Read previously run samples using pickle 
            checkdir = '' #'generic_11112020/'
            host_h5 = emcee_diagnostics_dir + checkdir + 'host_' + str(hostid) + '_emcee_sampler.h5'
            sn_h5 = emcee_diagnostics_dir + 'sn_' + str(segid) + '_emcee_sampler.h5'

            if os.path.isfile(host_h5):
                #read_pickle_make_plots('sn', ndim_sn, args_sn, truth_arr_sn, label_list_sn, segid, img_suffix)
                read_pickle_make_plots('host', ndim_host, args_host, truth_arr_host, label_list_host, hostid, img_suffix)
            else:
                # Call emcee
                #run_emcee('sn', nwalkers, ndim_sn, logpost_sn, pos_sn, args_sn, segid)
                #read_pickle_make_plots('sn', ndim_sn, args_sn, truth_arr_sn, label_list_sn, segid, img_suffix)

                run_emcee('host', nwalkers, ndim_host, logpost_host, pos_host, args_host, hostid)
                read_pickle_make_plots('host', ndim_host, args_host, truth_arr_host, label_list_host, hostid, img_suffix)

            sys.exit(0)

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)



