import numpy as np
import pymc3 as pm
import theano.tensor as tt

from astropy.io import fits
import scipy
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import griddata

import astropy.units as u
from specutils.analysis import snr_derived
from specutils import Spectrum1D

import arviz as az
import matplotlib.pyplot as plt
import corner

import os
import sys
import time
import datetime as dt
import socket

home = os.getenv('HOME')
ext_spectra_dir = home + "/Documents/roman_slitless_sims_results/"
roman_slitless_dir = home + "/Documents/GitHub/roman-slitless/"
roman_sims_seds = home + "/Documents/roman_slitless_sims_seds/"

stacking_utils = home + '/Documents/GitHub/stacking-analysis-pears/util_codes/'

sys.path.append(stacking_utils)
import dust_utils as du

start = time.time()
print("Starting at:", dt.datetime.now())

# Define constants
Lsol = 3.826e33

# Load in all models
# ------ THIS HAS TO BE GLOBAL!

# Read in SALT2 SN IA file from Lou
salt2_spec = np.genfromtxt(roman_sims_seds + "salt2_template_0.txt", \
    dtype=None, names=['day', 'lam', 'flam'], encoding='ascii')

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

# Now do the actual reading in
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
dl_cat = np.genfromtxt(stacking_utils + 'dl_lookup_table.txt', dtype=None, names=True)
# Get arrays 
dl_z_arr = np.asarray(dl_cat['z'], dtype=np.float64)
dl_cm_arr = np.asarray(dl_cat['dl_cm'], dtype=np.float64)
age_gyr_arr = np.asarray(dl_cat['age_gyr'], dtype=np.float64)

del dl_cat

print("Done loading all models. Time taken:", "{:.3f}".format(time.time()-start), "seconds.")

# ------------------
grism_sens_cat = np.genfromtxt(home + '/Documents/pylinear_ref_files/pylinear_config/Roman/roman_throughput_20190325.txt', \
    dtype=None, names=True, skip_header=3)

grism_sens_wav = grism_sens_cat['Wave'] * 1e4  # the text file has wavelengths in microns # needed in angstroms
grism_sens = grism_sens_cat['BAOGrism_1st']
grism_wav_idx = np.where(grism_sens > 0.25)
# ------------------

def get_snr(wav, flux):

    spectrum1d_wav = wav * u.AA
    spectrum1d_flux = flux * u.erg / (u.cm * u.cm * u.s * u.AA)
    spec1d = Spectrum1D(spectral_axis=spectrum1d_wav, flux=spectrum1d_flux)

    return snr_derived(spec1d)

def model_galaxy(x, z, ms, age, logtau, av):

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

    model_llam = np.asarray(models_arr[model_idx], dtype=np.float64)

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

def loglike_galaxy(theta, x, data, err):
    
    z, ms, age, logtau, av = theta

    y = model_galaxy(x, z, ms, age, logtau, av)

    # Only consider wavelengths where sensitivity is above 25%
    x0 = np.where( (x >= grism_sens_wav[grism_wav_idx][0]  ) &
                   (x <= grism_sens_wav[grism_wav_idx][-1] ) )[0]

    lnLike = get_lnLike_clip(y, data, err, x0)

    return lnLike

def get_lnLike_clip(y, data, err, x0):

    # Clip arrays
    y = y[x0]
    data = data[x0]
    err = err[x0]

    lnLike = -0.5 * np.nansum( (y-data)**2/err**2 )

    return lnLike

def get_dl_at_z(z):

    adiff = np.abs(dl_z_arr - z)
    z_idx = np.argmin(adiff)
    dl = dl_cm_arr[z_idx]

    return dl

def get_age_at_z(z):

    adiff = np.abs(dl_z_arr - z)
    z_idx = np.argmin(adiff)
    age_at_z = age_gyr_arr[z_idx]  # in Gyr

    return age_at_z

def apply_redshift(restframe_wav, restframe_lum, redshift):

    dl = get_dl_at_z(redshift)

    redshifted_wav = restframe_wav * (1 + redshift)
    redshifted_flux = restframe_lum / (4 * np.pi * dl * dl * (1 + redshift))

    return redshifted_wav, redshifted_flux

class LogLike(tt.Op):
    itypes = [tt.dvector]
    otypes = [tt.dscalar]

    def __init__(self, loglike_galaxy, x, data, err):

        # add inputs as class attributes
        self.likelihood = loglike_galaxy

        self.x = x
        self.data = data
        self.err = err

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        (theta,) = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(theta, self.x, self.data, self.err)

        outputs[0][0] = np.array(logl)  # output the log-likelihood


if __name__ == '__main__':

    print("\n##################")
    print("WARNING: only use pymc3 in the base conda env. NOT in astroconda.")
    print("##################\n")

    print(f"Running on PyMC3 v{pm.__version__}")
    print(f"Running on ArviZ v{az.__version__}")
    print("\n")
    # ---------------------------

    # --------------- Preliminary stuff
    ext_root = "romansim_"

    img_basename = '5deg_'
    img_suffix = 'Y106_0_2'

    exptime1 = '_900s'
    exptime2 = '_1800s'
    exptime3 = '_3600s'

    ext_spec_filename3 = ext_spectra_dir + ext_root + img_suffix + exptime3 + '_x1d.fits'
    ext_hdu3 = fits.open(ext_spec_filename3)
    print("Read in extracted spectra from:", ext_spec_filename3)

    segid_to_test = 1
    print("\nTesting fit for SegID:", segid_to_test)

    # Set pylinear f_lambda scaling factor
    pylinear_flam_scale_fac = 1e-17

    # Get spectrum
    wav = ext_hdu3[('SOURCE', segid_to_test)].data['wavelength']
    flam = ext_hdu3[('SOURCE', segid_to_test)].data['flam'] * pylinear_flam_scale_fac
    
    # Get snr
    snr = get_snr(wav, flam)

    # Set noise level based on snr
    noise_lvl = 1/snr

    # Create ferr array
    ferr = noise_lvl * flam

    # Set up for run
    ndraws = 1000  # number of draws from the distribution
    nburn = 200  # number of "burn-in points" (which we'll discard)

    nchains = 4
    ncores = 4

    ndim = 5

    # Labels for corner and trace plots
    label_list_galaxy = [r'$z$', r'$\mathrm{log(M_s/M_\odot)}$', r'$\mathrm{Age\, [Gyr]}$', \
    r'$\mathrm{\log(\tau\, [Gyr])}$', r'$A_V [mag]$'] 

    # create our Op
    logl = LogLike(loglike_galaxy, wav, flam, ferr)

    # use PyMC3 to sampler from log-likelihood
    with pm.Model() as model:
        # --------------- Priors
        # uniform priors 
        z = pm.Uniform("z", lower=0.0, upper=3.0)
        
        # Make sure model is not older than the Universe
        # Allowing at least 100 Myr for the first galaxies to form after Big Bang
        #age_at_z = get_age_at_z(z)
        #age_lim = age_at_z - 0.1  # in Gyr

        ms = pm.Uniform("ms", lower=9.0, upper=12.5)
        age = pm.Uniform("age", lower=0.01, upper=10.0)
        logtau = pm.Uniform("logtau", lower=-3.0, upper=2.0)
        av = pm.Uniform("av", lower=0.0, upper=5.0)

        # ----------------
    
        # convert inputs to a tensor vector
        theta = tt.as_tensor_variable([z, ms, age, logtau, av])
    
        # use a DensityDist (use a lamdba function to "call" the Op)
        #pm.DensityDist("likelihood", my_logl, observed={"v": theta})
        like = pm.Potential("like", logl(theta))

    with model:
        trace = pm.sample(ndraws, cores=ncores, chains=nchains, tune=nburn, discard_tuned_samples=True)
        print(pm.summary(trace).to_string())

    sys.exit(0)
