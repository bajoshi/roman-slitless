import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import griddata

import os
import sys
import socket
import time
import datetime as dt

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
astropy_cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)

import matplotlib.pyplot as plt

start = time.time()
print("Starting at:", dt.datetime.now())

# Assign directories and custom imports
home = os.getenv('HOME')
stacking_utils = home + '/Documents/GitHub/stacking-analysis-pears/util_codes/'

roman_slitless_dir = home + "/Documents/GitHub/roman-slitless/"
ext_spectra_dir = home + "/Documents/roman_slitless_sims_results/"
roman_sims_seds = home + "/Documents/roman_slitless_sims_seds/"

sys.path.append(stacking_utils)
import proper_and_lum_dist as cosmo
from dust_utils import get_dust_atten_model
from bc03_utils import get_age_spec

grism_sens_cat = np.genfromtxt(home + '/Documents/pylinear_ref_files/pylinear_config/Roman/roman_throughput_20190325.txt', \
    dtype=None, names=True, skip_header=3)

grism_sens_wav = grism_sens_cat['Wave'] * 1e4  # the text file has wavelengths in microns # needed in angstroms
grism_sens = grism_sens_cat['BAOGrism_1st']
grism_wav_idx = np.where(grism_sens > 0.25)

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

print("Done loading all models. Time taken:", "{:.3f}".format(time.time()-start), "seconds.")


def logpost_host(theta, x, data, err, zprior, zprior_sigma):

    lp = logprior_host(theta, zprior, zprior_sigma)
    print("Prior HOST:", lp)
    
    if not np.isfinite(lp):
        return -np.inf
    
    lnL = loglike_host(theta, x, data, err)

    #print("Likelihood HOST:", lnL)
    
    return lp + lnL


def logprior_host(theta, zprior, zprior_sigma):

    z, ms, age, logtau, av = theta
    print("\nParameter vector given:", theta)

    if (0.0001 <= z <= 6.0):
    
        # Make sure model is not older than the Universe
        # Allowing at least 100 Myr for the first galaxies to form after Big Bang
        age_at_z = astropy_cosmo.age(z).value  # in Gyr
        age_lim = age_at_z - 0.1  # in Gyr

        if ((0.0 <= ms <= 14.0) and \
            (0.01 <= age <= age_lim) and \
            (-3.0 <= logtau <= 2.0) and \
            (0.0 <= av <= 5.0)):

            # Gaussian prior on redshift
            ln_pz = np.log( 1.0 / (np.sqrt(2*np.pi)*zprior_sigma) ) - 0.5*(z - zprior)**2/zprior_sigma**2

            return ln_pz
    
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
    lnLike = -0.5 * np.nansum( (y-data)**2/err**2 ) - 0.5 * np.nansum( np.log(2 * np.pi * err**2) )
    #stretch_fac = 10.0
    #lnLike = -0.5 * (1 + stretch_fac) * chi2

    print("Pure chi2 term:", np.nansum( (y-data)**2/err**2 ))
    print("Second error term:", np.nansum( np.log(2 * np.pi * err**2) ))
    print("log likelihood HOST:", lnLike)

    
    fig = plt.figure(figsize=(9,5))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'$\lambda\, [\mathrm{\AA}]$', fontsize=14)
    ax.set_ylabel(r'$f_\lambda\, [\mathrm{cgs}]$', fontsize=14)

    ax.plot(x, data, color='k')
    ax.fill_between(x, data - err, data + err, color='gray', alpha=0.5)

    ax.plot(x, y, color='firebrick')

    ax.set_xscale('log')
    plt.show()
    sys.exit(0)
    

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

    metals = 0.02

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

    # ------ Apply dust extinction
    model_dusty_llam = get_dust_atten_model(model_lam, model_llam, av)

    # ------ Multiply luminosity by stellar mass
    model_dusty_llam = model_dusty_llam * 10**ms

    # ------ Apply redshift
    model_lam_z, model_flam_z = cosmo.apply_redshift(model_lam, model_dusty_llam, z)
    Lsol = 3.826e33
    model_flam_z = Lsol * model_flam_z

    # ------ Apply LSF
    model_lsfconv = gaussian_filter1d(input=model_flam_z, sigma=1.0)

    # ------ Downgrade to grism resolution
    model_mod = griddata(points=model_lam_z, values=model_lsfconv, xi=x)

    return model_mod


def main():

    ext_root = "romansim1"
    img_suffix = 'Y106_11_1'

    # Read in sed.lst
    sedlst_header = ['segid', 'sed_path']
    sedlst_path = roman_slitless_dir + 'pylinear_lst_files/' + 'sed_plffsn2_' + img_suffix + '.lst'
    sedlst = np.genfromtxt(sedlst_path, dtype=None, names=sedlst_header, encoding='ascii')
    print("Read in sed.lst from:", sedlst_path)

    # Read in the extracted spectra
    ext_spec_filename = ext_spectra_dir + 'plffsn2_run_dec7/' + ext_root + '_ext_x1d.fits'
    ext_hdu = fits.open(ext_spec_filename)
    print("Read in extracted spectra from:", ext_spec_filename)

    # Set pylinear f_lambda scaling factor
    pylinear_flam_scale_fac = 1e-17

    # ---- Get data and test fitting
    hostid = 207

    host_wav = ext_hdu[('SOURCE', hostid)].data['wavelength']
    host_flam = ext_hdu[('SOURCE', hostid)].data['flam'] * pylinear_flam_scale_fac

    # ---- Apply noise and get dummy noisy spectra
    noise_level = 0.03  # relative to signal

    host_ferr = noise_level * host_flam

    # ---- fitting
    zprior = 1.96
    zprior_sigma = 0.05
    rhost_init = np.array([zprior, 13.3,  1.0, 1.1, 0.0])

    # Test figure showing 
    fig = plt.figure()
    ax = fig.add_subplot(111)

    z_arr = np.arange(0.001, 6.001, 0.001)
    for z in z_arr:
        pdf_z = ( 1.0 / (np.sqrt(2*np.pi)*zprior_sigma) ) * np.exp(-0.5*(z - zprior)**2/zprior_sigma**2)
        print("{:.3f}".format(z), "      ", "{:.3e}".format(pdf_z))

    #sys.exit(0)
    #ln_pz = np.log( 1.0 / (np.sqrt(2*np.pi)*zprior_sigma) ) - 0.5*(z_arr - zprior)**2/zprior_sigma**2
    pdf_z = ( 1.0 / (np.sqrt(2*np.pi)*zprior_sigma) ) * np.exp(-0.5*(z_arr - zprior)**2/zprior_sigma**2)

    ax.plot(z_arr, pdf_z)
    #ax.set_yscale('log')

    plt.show()
    sys.exit(0)

    # now call posterior func to test
    logpost_host(rhost_init, host_wav, host_flam, host_ferr, zprior, zprior_sigma)

    return None



if __name__ == '__main__':
    main()
    sys.exit(0)










