from astropy.io import fits
import scipy
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import griddata

import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
astropy_cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)
from specutils.analysis import snr_derived
from specutils import Spectrum1D

import os
import sys
import socket
import time
import datetime as dt
import glob
from functools import reduce

import numpy as np
import emcee
import corner
from multiprocessing import Pool
#from lmfit import Parameters, fit_report, Minimizer

from numba import njit

import matplotlib.pyplot as plt

start = time.time()
print("Starting at:", dt.datetime.now())

print("Emcee version:", emcee.__version__)
print("Corner version:", corner.__version__)

# Assign directories and custom imports
cwd = os.path.dirname(os.path.abspath(__file__))
fitting_utils = cwd + '/utils/'

home = os.getenv('HOME')
pears_figs_dir = home + '/Documents/pears_figs_data/'

sys.path.append(fitting_utils)
import dust_utils as du

# Define constants
Lsol = 3.826e33
sn_day_arr = np.arange(-19,50,1)

# Load in all models
# ------ THIS HAS TO BE GLOBAL!

# Read in SALT2 SN IA file from Lou
salt2_spec = np.genfromtxt(fitting_utils + "salt2_template_0.txt", \
    dtype=None, names=['day', 'lam', 'flam'], encoding='ascii')

# Get the dirs correct
if 'plffsn2' in socket.gethostname():
    home = os.getenv('HOME')
    extdir = '/astro/ffsn/Joshi/'
    modeldir = extdir + 'bc03_output_dir/'
    roman_direct_dir = home + '/Documents/roman_direct_sims/sims2021/'
else:
    extdir = '/Volumes/Joshi_external_HDD/Roman/'
    modeldir = extdir + 'bc03_output_dir/m62/'
    roman_direct_dir = extdir + 'roman_direct_sims/sims2021/'

assert os.path.isdir(modeldir)
assert os.path.isdir(roman_direct_dir)

dir_img_part = 'part1'
img_sim_dir = roman_direct_dir + 'K_5degimages_' + dir_img_part + '/'

# Now do the actual reading in
model_lam = np.load(extdir + "bc03_output_dir/bc03_models_wavelengths.npy")
model_ages = np.load(extdir + "bc03_output_dir/bc03_models_ages.npy")

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

# Read in all models and parameters
model_lam_grid = np.load(pears_figs_dir + 'model_lam_grid_withlines_chabrier.npy')
model_grid = np.load(pears_figs_dir + 'model_comp_spec_llam_withlines_chabrier.npy',
    mmap_mode='r')

ml = np.asarray(model_lam_grid, dtype=np.float64)

log_age_arr = np.load(pears_figs_dir + 'log_age_arr_chab.npy')
metal_arr = np.load(pears_figs_dir + 'metal_arr_chab.npy')
tau_gyr_arr = np.load(pears_figs_dir + 'tau_gyr_arr_chab.npy')
tauv_arr = np.load(pears_figs_dir + 'tauv_arr_chab.npy')

# Also load in lookup table for luminosity distance
dl_cat = np.genfromtxt(fitting_utils + 'dl_lookup_table.txt', dtype=None, names=True)
# Get arrays 
dl_z_arr = np.asarray(dl_cat['z'], dtype=np.float64)
dl_cm_arr = np.asarray(dl_cat['dl_cm'], dtype=np.float64)
age_gyr_arr = np.asarray(dl_cat['age_gyr'], dtype=np.float64)

del dl_cat

# --------------
# Get prism sensitivity curve
prism_sens_cat = np.genfromtxt(fitting_utils + 'roman_throughput_20190325.txt', \
    dtype=None, names=True, skip_header=3)

prism_sens_wav = prism_sens_cat['Wave'] * 1e4
# the text file has wavelengths in microns # needed in angstroms
prism_sens = prism_sens_cat['SNPrism']
prism_wav_idx = np.where(prism_sens > 0.3)

# Not using the sens limit above but
# Hardcoded for now
prism_wmin, prism_wmax = 7600, 18000
print("Done loading all models. Time taken:", "{:.3f}".format(time.time()-start), "seconds.")

# Done with global stuff.
# ----------------------------------
def get_snr(wav, flux):

    spectrum1d_wav = wav * u.AA
    spectrum1d_flux = flux * u.erg / (u.cm * u.cm * u.s * u.AA)
    spec1d = Spectrum1D(spectral_axis=spectrum1d_wav, flux=spectrum1d_flux)

    return snr_derived(spec1d)

@njit
def get_dl_at_z(z):

    adiff = np.abs(dl_z_arr - z)
    z_idx = np.argmin(adiff)
    dl = dl_cm_arr[z_idx]

    return dl

@njit
def apply_redshift(restframe_wav, restframe_lum, redshift):

    dl = get_dl_at_z(redshift)

    redshifted_wav = restframe_wav * (1 + redshift)
    redshifted_flux = restframe_lum / (4 * np.pi * dl * dl * (1 + redshift))

    return redshifted_wav, redshifted_flux

def get_template(age, tau, tauv, metallicity, \
    log_age_arr, metal_arr, tau_gyr_arr, tauv_arr, \
    model_comp_spec_withlines_mmap):

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

    return model_llam

@njit
def add_stellar_vdisp(spec_wav, spec_flux, vdisp):

    # Now compute the broadened spectrum by numerically
    # integrating a Gaussian stellar velocity function
    # with the stellar vdisp.
    # Integration done numerically as a Riemann sum.

    speed_of_light = 299792.458  # km per second
    delta_v = 10.0

    vdisp_spec = np.zeros(len(spec_wav))

    #print(len(spec_wav))

    for w in range(len(spec_wav)):
        #print(w)
        lam = spec_wav[w]

        I = 0

        # Now compute the integrand numerically
        # between velocities that are within 
        # +- 3-sigma using the specified velocity dispersion.
        # Mean of all velocities should be 0,
        # of course since the avg vel of all stars within
        # a rotating disk or in an elliptical galaxy should be zero.
        for v in np.arange(-3*vdisp, 3*vdisp, delta_v):

            beta = 1 + (v/speed_of_light)
            new_lam = lam / beta
            new_lam_idx = np.argmin(np.abs(spec_wav - new_lam))

            flux_at_new_lam = spec_flux[new_lam_idx]

            gauss_exp_func = np.exp(-1*v*v/(2*vdisp*vdisp))

            prod = flux_at_new_lam * gauss_exp_func * delta_v
            I += prod

        vdisp_spec[w] = I / (vdisp*np.sqrt(2*np.pi))

    return vdisp_spec

def model_galaxy(x, z, ms, age, logtau, av, stellar_vdisp=False):
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

    # If using hte larger model set with no emission lines
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

    elif tau >= 20.0:

        logtau_arr = np.arange(1.30, 2.01, 0.01)
        logtau_idx = np.argmin(abs(logtau_arr - logtau))

        age_idx = np.argmin(abs(model_ages - age*1e9))
        model_idx = logtau_idx * len(model_ages) + age_idx

        models_arr = all_m62_models[-1]

    # Force to numpy array for numba
    model_llam = np.asarray(models_arr[model_idx], dtype=np.float64)
    """

    # Smaller model set with emission lines
    tau = 10**logtau  # logtau is log of tau in gyr
    tauv = av / 1.086
    model_llam = get_template(np.log10(age * 1e9), tau, tauv, 0.02, \
        log_age_arr, metal_arr, tau_gyr_arr, tauv_arr, model_grid)

    model_llam = np.asarray(model_llam, dtype=np.float64)

    # ------ Apply stellar velocity dispersion
    # assumed for now as a constant 220 km/s
    # TODO: optimize
    # -- This does not have to be done each time the model function
    #    is called because we're assuming a constant vel disp
    if stellar_vdisp:
        sigmav = 220
        model_vdisp = add_stellar_vdisp(ml, model_llam, sigmav)

        # ------ Apply dust extinction
        model_dusty_llam = du.get_dust_atten_model(ml, model_vdisp, av)

    else:
        # ------ Apply dust extinction
        model_dusty_llam = du.get_dust_atten_model(ml, model_llam, av)

    model_dusty_llam = np.asarray(model_dusty_llam, dtype=np.float64)

    # ------ Multiply luminosity by stellar mass
    model_dusty_llam = model_dusty_llam * 10**ms

    # ------ Apply redshift
    model_lam_z, model_flam_z = apply_redshift(ml, model_dusty_llam, z)
    #model_flam_z = Lsol * model_flam_z

    # ------ Apply LSF
    #model_lsfconv = gaussian_filter1d(input=model_flam_z, sigma=30.0)

    # ------ Downgrade and regrid to grism resolution
    model_mod = griddata(points=model_lam_z, values=model_flam_z, xi=x)

    return model_mod

def loglike_galaxy(theta, x, data, err, x0):

    z, ms, age, logtau, av = theta

    y = model_galaxy(x, z, ms, age, logtau, av)

    #alpha = np.sum(data * y / err**2) / np.sum(y**2 / err**2)
    #y = y * alpha

    lnLike = get_lnLike_clip(y, data, err, x0)

    """
    print("Pure chi2 term:", np.nansum( (y-data)**2/err**2 ))

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'$\lambda\, [\mathrm{\AA}]$', fontsize=14)
    ax.set_ylabel(r'$f_\lambda\, [\mathrm{cgs}]$', fontsize=14)

    ax.plot(x, data, color='k')
    ax.fill_between(x, data - err, data + err, color='gray', alpha=0.5)

    ax.plot(x, y, color='firebrick')

    plt.show()
    """

    return lnLike

@njit
def get_lnLike_clip(y, data, err, x0):

    # Clip arrays
    y = y[x0]
    data = data[x0]
    err = err[x0]

    lnLike = -0.5 * np.nansum( (y-data)**2/err**2 )

    return lnLike

@njit
def get_age_at_z(z):

    adiff = np.abs(dl_z_arr - z)
    z_idx = np.argmin(adiff)
    age_at_z = age_gyr_arr[z_idx]  # in Gyr

    return age_at_z

@njit
def logprior_galaxy(theta, zprior, zprior_sigma):

    z, ms, age, logtau, av = theta
    #print("\nParameter vector given:", theta)

    if (0.0001 <= z <= 6.0):

        # Make sure model is not older than the Universe
        # Allowing at least 100 Myr for the first galaxies to form after Big Bang
        age_at_z = get_age_at_z(z)
        age_lim = age_at_z - 0.1  # in Gyr

        if ((9.0 <= ms <= 12.5) and \
            (0.01 <= age <= age_lim) and \
            (-3.0 <= logtau <= 2.0) and \
            (0.0 <= av <= 5.0)):

            return 0.0

    return -np.inf

def logpost_galaxy(theta, x, data, err, zprior, zprior_sigma, x0):

    lp = logprior_galaxy(theta, zprior, zprior_sigma)

    if not np.isfinite(lp):
        return -np.inf

    lnL = loglike_galaxy(theta, x, data, err, x0)

    return lp + lnL

def read_galaxy_data(galaxy_filename):

    truth_dict = {}

    with open(galaxy_filename, 'r') as dat:

        all_lines = dat.readlines()

        # First initialize arrays to save spectra
        # Also get truths
        for line in all_lines:
            if 'SIM_REDSHIFT_HOST:' in line:
                true_z = float(line.split()[1])

            if 'NSPECTRA:' in line:
                nspectra = int(line.split()[1])

            if 'SPECTRUM_NLAM:' in line:
                spec_nlam = int(line.split()[1])

        # Set up truth dict
        truth_dict['z'] = true_z

        try:
            assert nspectra == 1
        except NameError:
            return None, None, None, None, None, None, 1

        # Set up empty arrays
        l0 = np.empty(spec_nlam)
        l1 = np.empty(spec_nlam)
        gal_flam = np.empty(spec_nlam)
        gal_simflam = np.empty(spec_nlam)
        gal_ferr = np.empty(spec_nlam)

        # Now get spectra
        i = 0

        for line in all_lines:

            if line[:5] == 'SPEC:':
                lsp = line.split()

                l0[i] = float(lsp[1])
                l1[i] = float(lsp[2])

                gal_flam[i] = float(lsp[3])
                gal_ferr[i] = float(lsp[4])
                gal_simflam[i] = float(lsp[5])

                i += 1

            if 'SPECTRUM_END:' in line:
                break

    # get the midpoints of the wav ranges
    gal_wav = (l0 + l1) / 2

    return nspectra, gal_wav, gal_flam, gal_ferr, gal_simflam, truth_dict, 0

def read_pickle_make_plots(savedir, object_type, ndim, args_obj, label_list):

    h5_path = savedir + 'emcee_sampler_' + object_type + '.h5'
    sampler = emcee.backends.HDFBackend(h5_path)

    samples = sampler.get_chain()
    print("\nRead in sampler:", h5_path)
    print("Samples shape:", samples.shape)

    #reader = emcee.backends.HDFBackend(pkl_path.replace('.pkl', '.h5'))
    #samples = reader.get_chain()
    #tau = reader.get_autocorr_time(tol=0)

    # Get autocorrelation time
    # Discard burn-in. You do not want to consider the burn in the corner plots/estimation.
    tau = sampler.get_autocorr_time(tol=0)
    if not np.any(np.isnan(tau)):
        burn_in = int(2 * np.max(tau))
        thinning_steps = int(0.5 * np.min(tau))
    else:
        burn_in = 200
        thinning_steps = 30

    print("Average Tau:", np.mean(tau))
    print("Burn-in:", burn_in)
    print("Thinning steps:", thinning_steps)

    # plot trace
    fig1, axes1 = plt.subplots(ndim, figsize=(10, 6), sharex=True)

    for i in range(ndim):
        ax1 = axes1[i]
        ax1.plot(samples[:, :, i], "k", alpha=0.05)
        #ax1.axhline(y=truth_arr[i], color='tab:red', lw=2.0)
        ax1.set_xlim(0, len(samples))
        ax1.set_ylabel(label_list[i], fontsize=15)
        ax1.yaxis.set_label_coords(-0.1, 0.5)

    axes1[-1].set_xlabel("Step number")

    fig1.savefig(savedir + 'emcee_trace_' + object_type + '.pdf', dpi=200, bbox_inches='tight')

    # Create flat samples
    flat_samples = sampler.get_chain(discard=burn_in, thin=thinning_steps, flat=True)
    print("\nFlat samples shape:", flat_samples.shape)

    cq_z = corner.quantile(x=flat_samples[:, 0], q=[0.16, 0.5, 0.84])
    cq_ms = corner.quantile(x=flat_samples[:, 1], q=[0.16, 0.5, 0.84])
    cq_age = corner.quantile(x=flat_samples[:, 2], q=[0.16, 0.5, 0.84])
    cq_tau = corner.quantile(x=flat_samples[:, 3], q=[0.16, 0.5, 0.84])
    cq_av = corner.quantile(x=flat_samples[:, 4], q=[0.16, 0.5, 0.84])

    # print parameter estimates
    print("Parameter estimates:")
    print("Redshift: ", cq_z)
    print("Stellar mass [log(M/M_sol)]:", cq_ms)
    print("Age [Gyr]: ", cq_age)
    print("log SFH Timescale [Gyr]: ", cq_tau)
    print("Visual extinction [mag]:", cq_av)

    fig = corner.corner(flat_samples, quantiles=[0.16, 0.5, 0.84], labels=label_list, \
        label_kwargs={"fontsize": 14}, show_titles='True', title_kwargs={"fontsize": 14}, \
        verbose=True, smooth=1.0, smooth1d=1.0)



    fig.savefig(savedir + 'corner_' + object_type + '.pdf', dpi=200, bbox_inches='tight')

    # ------------ Plot 100 random models from the parameter space 
    # within +-1sigma of corner estimates
    # first pull out required stuff from args
    wav = args_obj[0]
    flam = args_obj[1]
    ferr = args_obj[2]

    fig3 = plt.figure(figsize=(9,4))
    ax3 = fig3.add_subplot(111)

    ax3.set_xlabel(r'$\mathrm{\lambda\ [\AA]}$', fontsize=15)
    ax3.set_ylabel(r'$\mathrm{f_\lambda\ [erg\, s^{-1}\, cm^{-2}\, \AA^{-1}]}$', fontsize=15)

    model_count = 0
    ind_list = []

    while model_count <= 100:

        ind = int(np.random.randint(len(flat_samples), size=1))
        ind_list.append(ind)

        # make sure sample has correct shape
        sample = flat_samples[ind]

        model_okay = 0
        sample = sample.reshape(5)

        # Get the parameters of the sample
        model_z = sample[0]
        model_ms = sample[1]
        model_age = sample[2]
        model_tau = sample[3]
        model_av = sample[4]

        # Check that the model is within +-1 sigma
        # of value inferred by corner contours
        if (model_z >= cq_z[0]) and (model_z <= cq_z[2]) and \
           (model_ms >= cq_ms[0]) and (model_ms <= cq_ms[2]) and \
           (model_age >= cq_age[0]) and (model_age <= cq_age[2]) and \
           (model_tau >= cq_tau[0]) and (model_tau <= cq_tau[2]) and \
           (model_av >= cq_av[0]) and (model_av <= cq_av[2]):

           model_okay = 1

        # Now plot if the model is okay
        if model_okay:

            m = model_galaxy(wav, sample[0], sample[1], sample[2], sample[3], sample[4])

            ax3.plot(wav, m, color='royalblue', lw=0.5, alpha=0.05, zorder=2)

            model_count += 1

    ax3.plot(wav, flam, color='k', lw=1.0, zorder=1)
    ax3.fill_between(wav, flam - ferr, flam + ferr, color='gray', alpha=0.5, zorder=1)

    # ADD LEGEND
    ax3.text(x=0.65, y=0.92, s='--- Simulated data', 
        verticalalignment='top', horizontalalignment='left', 
        transform=ax3.transAxes, color='k', size=12)
    ax3.text(x=0.65, y=0.85, s='--- 100 randomly chosen samples', 
        verticalalignment='top', horizontalalignment='left', 
        transform=ax3.transAxes, color='royalblue', size=12)

    # Get plotting limits
    yidx = np.where((wav >= prism_wmin) & (wav <= prism_wmax))[0]

    ymin = np.min(flam[yidx]) * 0.8
    ymax = np.max(flam[yidx]) * 1.2

    ax3.set_xlim(prism_wmin, prism_wmax)
    ax3.set_ylim(ymin, ymax)

    fig3.savefig(savedir + 'emcee_overplot_' + object_type + '.pdf', 
        dpi=200, bbox_inches='tight')


    # Close all figures
    fig1.clear()
    fig.clear()
    fig3.clear()

    #plt.clf()
    #plt.cla()
    plt.close(fig1)
    plt.close(fig)
    plt.close(fig3)

    return None

def main():

    # data dir
    datadir = home + '/Documents/sn_sit_hackday/testv3/Prism_shallow_hostIav3/'
    savedir = datadir + 'results/'

    checkplot = False

    # these files don't have galaxy spectra
    #toskip_shallow = []
    #toskip_deep = ['10043', '10037', '10038', '10015']
    #toskip = toskip_shallow
    skipped_list = []

    # Other preliminary stuff
    nwalkers = 1200
    niter = 500

    ncount = 0
    for fl in glob.glob(datadir + '*.DAT'):

        # Check if it needs to be skipped
        continue_flag = 0
        #for u in range(len(toskip)):
        #    skip_g = toskip[u]
        #    if skip_g in os.path.basename(fl):
        #        print("\nSkipping:", fl)
        #        continue_flag = 1
        #        break

        if not continue_flag:
            ncount += 1

            print("\n----------------")
            print("Filename:", os.path.basename(fl))

            nspectra, gal_wav, gal_flam, gal_ferr,\
            gal_simflam, truth_dict, return_code = read_galaxy_data(fl)

            if return_code == 1:
                print("File contains no spectra. Skipping.")
                skipped_list.append(os.path.basename(fl))
                continue

            # Basic plot to check data quality
            snr = get_snr(gal_wav, gal_flam)
            print("SNR:", snr)
            print("True values", truth_dict)

            fl_name_base = os.path.basename(fl).split('.DAT')[0]
            galid = int(fl_name_base.split('_')[-1].lstrip('SN'))
            print("Galaxy ID:", galid)

            if checkplot:
                # Code block to check figure
                fig = plt.figure(figsize=(10,5))
                ax = fig.add_subplot(111)

                ax.set_xlabel(r'$\mathrm{\lambda\ [\AA]}$', fontsize=15)
                ax.set_ylabel(r'$\mathrm{f_\lambda\ [erg\, s^{-1}\, cm^{-2}\, \AA^{-1}]}$',
                fontsize=15)

                ax.plot(gal_wav, gal_flam, lw=2.0, color='k')
                ax.fill_between(gal_wav, gal_flam - gal_ferr, gal_flam + gal_ferr,
                color='gray', alpha=0.5)

                ax.plot(gal_wav, gal_simflam, lw=2.0, color='firebrick')

                # Get plotting limits
                xmin, xmax = 7600, 18000
                yidx = np.where((gal_wav >= xmin) & (gal_wav <= xmax))[0]

                ymin = np.min(gal_flam[yidx]) * 0.7
                ymax = np.max(gal_flam[yidx]) * 1.4

                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)

                plt.show()
                if ncount > 10:
                    sys.exit(0)
                else:
                    continue

            # Only consider wavelengths where sensitivity is above 25%
            #x0 = np.where( (gal_wav >= prism_sens_wav[prism_wav_idx][0]  ) &
            #               (gal_wav <= prism_sens_wav[prism_wav_idx][-1] ) )[0]

            x0 = np.where( (gal_wav >= 7700  ) & (gal_wav <= 18000 ) )[0]

            # Setup for emcee
            # Labels for corner and trace plots
            label_list_galaxy = [r'$z$', r'$\mathrm{log(M_s/M_\odot)}$',
            r'$\mathrm{Age\, [Gyr]}$', \
            r'$\mathrm{\log(\tau\, [Gyr])}$', r'$A_V [mag]$']

            # Set jump sizes # ONLY FOR INITIAL POSITION SETUP
            jump_size_z = 0.5
            jump_size_ms = 1.0  # log(ms)
            jump_size_age = 1.0  # in gyr
            jump_size_logtau = 0.2  # tau in gyr
            jump_size_av = 0.5  # magnitudes

            zprior = 0.5
            zprior_sigma = 0.02

            args_galaxy = [gal_wav, gal_flam, gal_ferr, zprior, zprior_sigma, x0]

            # Initial guess
            rgal_init = np.array([zprior, 10.0, 6.0, 1.0, 0.2])

            # Setup dims and walkers
            ndim_gal = 5

            # generating ball of walkers about initial position defined above
            pos_gal = np.zeros(shape=(nwalkers, ndim_gal))

            for i in range(nwalkers):

                # ---------- For galaxies
                rg0 = float(rgal_init[0] + jump_size_z * np.random.normal(size=1))
                rg1 = float(rgal_init[1] + jump_size_ms * np.random.normal(size=1))
                rg2 = float(rgal_init[2] + jump_size_age * np.random.normal(size=1))
                rg3 = float(rgal_init[3] + jump_size_logtau * np.random.normal(size=1))
                rg4 = float(rgal_init[4] + jump_size_av * np.random.normal(size=1))

                rg = np.array([rg0, rg1, rg2, rg3, rg4])

                pos_gal[i] = rg

            print("Starting position for galaxies from where")
            print("ball of walkers will be generated:")
            print(rgal_init)

            print("logpost at starting position for galaxy:")
            print(logpost_galaxy(rgal_init, gal_wav, gal_flam, gal_ferr, \
                zprior, zprior_sigma, x0))

            # Running emcee
            print("\nRunning emcee...")

            ## ----------- Set up the HDF5 file to incrementally save progress to
            emcee_savefile = savedir + 'emcee_sampler_' + str(galid) + '.h5'
            if not os.path.isfile(emcee_savefile):

                backend = emcee.backends.HDFBackend(emcee_savefile)
                backend.reset(nwalkers, ndim_gal)

                with Pool() as pool:

                    sampler = emcee.EnsembleSampler(nwalkers, ndim_gal, logpost_galaxy,
                        args=args_galaxy, backend=backend, pool=pool,
                        moves=emcee.moves.KDEMove())
                    sampler.run_mcmc(pos_gal, niter, progress=True)

                print("Finished running emcee.")
                print("Mean acceptance Fraction:", np.mean(sampler.acceptance_fraction), "\n")

                read_pickle_make_plots(savedir, str(galid), ndim_gal,
                    args_galaxy, label_list_galaxy)

    # Print list of skipped files
    print('Skipped files:')
    print(skipped_list)

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)
