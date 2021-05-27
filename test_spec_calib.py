import numpy as np
from astropy.io import fits
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter1d
from numba import njit

import emcee
import corner

import os
import sys
import time
import datetime as dt
import socket

import matplotlib.pyplot as plt

home = os.getenv('HOME')
ext_spectra_dir = home + "/Documents/roman_slitless_sims_results/"
roman_slitless_dir = home + "/Documents/GitHub/roman-slitless/"
roman_sims_seds = home + "/Documents/roman_slitless_sims_seds/"

fitting_pipeline_dir = roman_slitless_dir + "fitting_pipeline/"
fitting_utils = fitting_pipeline_dir + "/utils/"

sys.path.append(fitting_utils)
import dust_utils as du
from get_snr import get_snr

start = time.time()
print("Starting at:", dt.datetime.now())

# ----------------------
# Define constants
Lsol = 3.826e33
# Set pylinear f_lambda scaling factor
pylinear_flam_scale_fac = 1e-17

# ----------------------
# Load in all models
# ------ THIS HAS TO BE GLOBAL!
sn_day_arr = np.arange(-19,50,1)

# Read in SALT2 SN IA file from Lou
salt2_spec = np.genfromtxt(roman_sims_seds + "salt2_template_0.txt", \
    dtype=None, names=['day', 'lam', 'flam'], encoding='ascii')

# ----------------------
# Get the dirs correct
if 'plffsn2' in socket.gethostname():
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

# ----------------------
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

# ----------------------
# Also load in lookup table for luminosity distance
dl_cat = np.genfromtxt(fitting_utils + 'dl_lookup_table.txt', dtype=None, names=True)
# Get arrays 
dl_z_arr = np.asarray(dl_cat['z'], dtype=np.float64)
dl_cm_arr = np.asarray(dl_cat['dl_cm'], dtype=np.float64)
age_gyr_arr = np.asarray(dl_cat['age_gyr'], dtype=np.float64)

del dl_cat

# ----------------------
# Load in the sensitivity file
prism_sens_cat = np.genfromtxt(home + '/Documents/pylinear_ref_files/' + \
    'pylinear_config/Roman/roman_throughput_20190325.txt', \
    dtype=None, names=True, skip_header=3)

prism_sens_wav = prism_sens_cat['Wave'] * 1e4
# the text file has wavelengths in microns # needed in angstroms
prism_sens = prism_sens_cat['SNPrism']
prism_wav_idx = np.where(prism_sens > 0.3)

# ----------------------
# Also load in lookup table for luminosity distance
dl_cat = np.genfromtxt(fitting_utils + 'dl_lookup_table.txt', dtype=None, names=True)
# Get arrays 
dl_z_arr = np.asarray(dl_cat['z'], dtype=np.float64)
dl_cm_arr = np.asarray(dl_cat['dl_cm'], dtype=np.float64)
age_gyr_arr = np.asarray(dl_cat['age_gyr'], dtype=np.float64)

del dl_cat

print("Done loading all models and supplementary data.")
print("Time taken:", "{:.3f}".format(time.time()-start), "seconds.")

# ----------------------
# FUNCTION DEFS

def get_chi2(model, flam, ferr, apply_a=True, indices=None):

    # Compute a and chi2
    if isinstance(indices, np.ndarray):

        a = np.nansum(flam[indices] * model / ferr[indices]**2) / \
            np.nansum(model**2 / ferr[indices]**2)
        if apply_a:
            model = a*model

        chi2 = np.nansum( (model - flam[indices])**2 / ferr[indices]**2 )

    else:

        a = np.nansum(flam * model / ferr**2) / np.nansum(model**2 / ferr**2)
        if apply_a:
            model = a*model

        chi2 = np.nansum( (model - flam)**2 / ferr**2 )

    if apply_a:
        return a, chi2
    else:
        return chi2

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

    # ------ Apply some LSF. 
    # This is a NUISANCE FACTOR ONLY FOR NOW
    # When we use actual SNe they will be point sources.
    #lsf_sigma = 0.5
    #sn_flam_z = scipy.ndimage.gaussian_filter1d(input=sn_flam_z, sigma=lsf_sigma)

    sn_mod = griddata(points=sn_lam_z, values=sn_flam_z, xi=x)

    # ------ combine host light
    # some fraction to account for host contamination
    # This fraction is a free parameter
    #sn_flam_hostcomb = sn_mod  +  host_frac * host_flam

    return sn_mod

def model_galaxy(x, z, ms, age, logtau, av):
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

        models_taurange_idx = np.argmin(abs(np.arange(tau_low, tau_high, 1) - \
            int(np.floor(tau))))
        models_arr = all_m62_models[models_taurange_idx]

    elif tau >= 20.0:
        
        logtau_arr = np.arange(1.30, 2.01, 0.01)
        logtau_idx = np.argmin(abs(logtau_arr - logtau))

        age_idx = np.argmin(abs(model_ages - age*1e9))
        model_idx = logtau_idx * len(model_ages) + age_idx

        models_arr = all_m62_models[-1]

    model_llam = np.asarray(models_arr[model_idx], dtype=np.float64)
    """
      This np.asarray stuff (here and for model_lam below) is very
      important for numba to be able to do its magic. It does not 
      like args passed into a numba @jit(nopython=True) decorated 
      function to come from np.load(..., mmap_mode='r').
      So I made the arrays passed into the function explicitly be
      numpy arrays of dtype=np.float64. 
      For now only the two functions in dust_utils are numba decorated
      because applying the dust extinction was the most significant
      bottleneck in this code. I suspect if more functions were numba
      decorated then the code will go even faster.
      E.g., after using numba an SN run of 2000 steps finishes in 
      <~2 min whereas it used to take ~25 min (on my laptop). On 
      PLFFSN2 the same run used to take ~9 min, it now finishes in
        seconds!
      For a galaxy a run of 2000 steps 
      
    """

    # ------ Apply dust extinction
    ml = np.asarray(model_lam, dtype=np.float64)
    model_dusty_llam = du.get_dust_atten_model(ml, model_llam, av)

    # ------ Multiply luminosity by stellar mass
    model_dusty_llam = model_dusty_llam * 10**ms

    # ------ Apply redshift
    model_lam_z, model_flam_z = apply_redshift(ml, model_dusty_llam, z)
    model_flam_z = Lsol * model_flam_z

    # ------ Apply LSF
    model_lsfconv = gaussian_filter1d(input=model_flam_z, sigma=1.0)

    # ------ Downgrade to grism resolution
    model_mod = griddata(points=model_lam_z, values=model_lsfconv, xi=x)

    return model_mod

def loglike(theta, x, y, yerr, m):
    a0, a1 = theta
    
    xp = (x - x[0]) / (x[-1] - x[0])
    calib_line = a0 + a1 * xp
    calib_model = y / calib_line

    lnLike = -0.5 * np.sum((m - calib_model) ** 2 / yerr**2)

    return lnLike

def logprior(theta):

    a0, a1 = theta

    if (0.0 <= a0 <= 3.0) and (-2.0 <= a1 <= 2.0):
        return 0.0

    return -np.inf

def logpost(theta, x, data, err, m):

    lp = logprior(theta)
    
    if not np.isfinite(lp):
        return -np.inf
    
    lnL = loglike(theta, x, data, err, m)
    
    return lp + lnL

def main():

    # --------------- Preliminary stuff
    ext_root = "romansim_prism_"

    img_basename = '5deg_'
    img_suffix = 'Y106_0_6'

    exptime = '_3600s'

    # --------------- Read in sed.lst
    sedlst_header = ['segid', 'sed_path']
    sedlst_path = roman_slitless_dir + 'pylinear_lst_files/' + 'sed_' + img_suffix + '.lst'
    sedlst = np.genfromtxt(sedlst_path, dtype=None, names=sedlst_header, encoding='ascii')
    print("Read in sed.lst from:", sedlst_path)

    print("Number of spectra in file:", len(sedlst))

    # --------------- Read in source catalog
    cat_filename = img_sim_dir + img_basename + img_suffix + '.cat'
    cat_header = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'ALPHA_J2000', 'DELTA_J2000', 
                  'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_AUTO', 'MAGERR_AUTO', 
                  'FLUX_RADIUS', 'FWHM_IMAGE']
    cat = np.genfromtxt(cat_filename, dtype=None, names=cat_header, encoding='ascii')

    # --------------- Read in the extracted spectra
    ext_spec_filename = ext_spectra_dir + ext_root + img_suffix + exptime + '_x1d.fits'
    ext_hdu = fits.open(ext_spec_filename)
    print("Read in extracted spectra from:", ext_spec_filename)

    # --------------- Get stuff needed for fitting
    segid_to_test = 483
    source_type = 'sn'
    print("\nTesting calibration fit for SegID:", segid_to_test)

    # Get spectrum
    wav = ext_hdu[('SOURCE', segid_to_test)].data['wavelength']
    flam = ext_hdu[('SOURCE', segid_to_test)].data['flam'] * pylinear_flam_scale_fac
    
    # Get snr
    snr = get_snr(wav, flam)

    # Set noise level based on snr
    noise_lvl = 1/snr

    # Create ferr array
    ferr = noise_lvl * flam

    # Only consider wavelengths where sensitivity is above the cutoff above
    x0 = np.where( (wav >= prism_sens_wav[prism_wav_idx][0]  ) &
                   (wav <= prism_sens_wav[prism_wav_idx][-1] ) )[0]

    wav = wav[x0]
    flam = flam[x0]
    ferr = ferr[x0]

    # Now get the model that was used as input
    # Not used for the fitting 
    # Only to show in the plot
    # I got the truths by eye from the sed.lst file.
    if source_type == 'sn':
        sn_z = 0.256
        sn_day = 24
        sn_av = 2.942

        m = model_sn(wav, sn_z, sn_day, sn_av)

    elif source_type == 'galaxy':

        galaxy_av = 4.26
        galaxy_met = 0.02
        galaxy_tau = 14.585
        galaxy_age = 6.96
        galaxy_ms = 10.98
        galaxy_z = 0.626

        galaxy_logtau = np.log10(galaxy_tau)

        m = model_galaxy(wav, galaxy_z, galaxy_ms, galaxy_age, galaxy_logtau, galaxy_av)

    a, chi2 = get_chi2(m, flam, noise_lvl*flam)
    m = m * a

    a0, a1 = 1.6, -0.4

    # DO NOT DELETE CODE BLOCK
    # Useful for figuring out initial guess
    x = (wav - wav[0]) / (wav[-1] - wav[0])
    calib_line = a0 + a1 * x
    calib_spec = flam / calib_line
    fig = plt.figure(figsize=(9,5))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'$\mathrm{Wavelength\, [\AA]}$', fontsize=14)
    ax.set_ylabel(r'$\mathrm{f_\lambda \, [cgs]}$', fontsize=14)

    axt = ax.twinx()
    axt.plot(wav, calib_line, color='turquoise', ls='--', 
        lw=2.0, label='Calib curve', zorder=1)
    
    ax.plot(wav, flam, color='k', lw=1.5, label='x1d spectrum', zorder=2)
    ax.fill_between(wav, flam-ferr, flam+ferr, color='gray', alpha=0.4)

    ax.plot(wav, m, color='tab:red', lw=2.0, label='Input model', zorder=2)

    ax.plot(wav, calib_spec, color='purple', lw=1.5, 
        label='Calibrated x1d spectrum', zorder=2)

    ax.legend(loc=0, frameon=False, fontsize=13)

    plt.show()
    sys.exit(0)

    # ----------------------
    # Bayesian fit for calibration
    label_list = ['a0', 'a1']
    init_guess = np.array([a0, a1])

    jump_size_a0 = 0.001
    jump_size_a1 = 0.001

    # Setup dims and walkers
    nwalkers = 1000
    ndim = 2

    # generating ball of walkers about initial position defined above
    pos = np.zeros(shape=(nwalkers, ndim))

    for i in range(nwalkers):

        # ---------- For galaxies
        rn0 = float(init_guess[0] + jump_size_a0 * np.random.normal(size=1))
        rn1 = float(init_guess[1] + jump_size_a1 * np.random.normal(size=1))

        rn = np.array([rn0, rn1])

        pos[i] = rn

    print("Starting position from where")
    print("ball of walkers will be generated:", init_guess)

    print("logpost at starting position:", logpost(init_guess, wav, flam, ferr, m))

    # Running emcee
    print("\nRunning emcee...")

    ## ----------- Set up the HDF5 file to incrementally save progress to
    emcee_savefile = 'emcee_sampler_calibtest' + str(segid_to_test) + '.h5'
    if not os.path.isfile(emcee_savefile):

        backend = emcee.backends.HDFBackend(emcee_savefile)
        backend.reset(nwalkers, ndim)

        args = [wav, flam, ferr, m]

        sampler = emcee.EnsembleSampler(nwalkers, ndim, logpost, args=args, backend=backend)
        sampler.run_mcmc(pos, 500, progress=True)

        print("Finished running emcee.")
        print("Mean acceptance Fraction:", np.mean(sampler.acceptance_fraction), "\n")

    sampler = emcee.backends.HDFBackend(emcee_savefile)

    tau = sampler.get_autocorr_time(tol=0)
    print("Autocorrelation time vector:", tau)

    burn_in = int(2 * np.max(tau))
    thinning_steps = int(0.5 * np.min(tau))

    print("Average autocorr:", np.mean(tau))
    print("Burn-in:", burn_in)
    print("Thinning steps:", thinning_steps)

    flat_samples = sampler.get_chain(discard=burn_in, thin=thinning_steps, flat=True)

    figcorner = corner.corner(flat_samples, quantiles=[0.16, 0.5, 0.84], labels=label_list, \
        label_kwargs={"fontsize": 14}, show_titles='True', title_kwargs={"fontsize": 14}, \
        verbose=True, smooth=0.8, smooth1d=0.8)

    # Overplot
    inds = np.random.randint(len(flat_samples), size=200)

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'$\mathrm{Wavelength}\, [\AA]$', fontsize=14)
    ax.set_ylabel(r'$\mathrm{f_\lambda \, [cgs]}$', fontsize=14)

    xp = (wav - wav[0]) / (wav[-1] - wav[0])

    ax.plot(wav, flam, color='k', lw=1.5, label='x1d spectrum', zorder=2)
    ax.fill_between(wav, flam-ferr, flam+ferr, color='gray', alpha=0.4)

    ax.plot(wav, m, color='tab:red', lw=2.0, label='Input model', zorder=2)

    for ind in inds:
        sample = flat_samples[ind]

        a0, a1 = sample[0], sample[1]
    
        calib_line = a0 + a1 * xp
        calib_model = flam / calib_line
        
        ax.plot(wav, calib_model, color='teal', alpha=0.05)

    ax.legend(loc=0, frameon=False, fontsize=13)

    plt.show()

    return None

if __name__ == '__main__':

    main()
    
    sys.exit(0)