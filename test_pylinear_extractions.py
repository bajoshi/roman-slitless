from astropy.io import fits
import scipy
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import griddata

import os
import sys
import socket
import time
import datetime as dt

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import emcee
import corner
from multiprocessing import Pool
from lmfit import Parameters, fit_report, Minimizer

from numba import njit

import matplotlib.pyplot as plt

home = os.getenv('HOME')
ext_spectra_dir = home + "/Documents/roman_slitless_sims_results/"
roman_slitless_dir = home + "/Documents/GitHub/roman-slitless/"
roman_sims_seds = home + "/Documents/roman_slitless_sims_seds/"

fitting_pipeline_dir = roman_slitless_dir + "fitting_pipeline/"
fitting_utils = fitting_pipeline_dir + "/utils/"

sys.path.append(fitting_utils)
import proper_and_lum_dist as cosmo
import dust_utils as du
from get_snr import get_snr

start = time.time()
print("Starting at:", dt.datetime.now())

# Define constants
Lsol = 3.826e33
sn_day_arr = np.arange(-19,50,1)

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

# Also load in lookup table for luminosity distance
dl_cat = np.genfromtxt(fitting_utils + 'dl_lookup_table.txt', dtype=None, names=True)
# Get arrays 
dl_z_arr = np.asarray(dl_cat['z'], dtype=np.float64)
dl_cm_arr = np.asarray(dl_cat['dl_cm'], dtype=np.float64)
age_gyr_arr = np.asarray(dl_cat['age_gyr'], dtype=np.float64)

del dl_cat

print("Done loading all models. Time taken:")
print("{:.3f}".format(time.time()-start), "seconds.")

# Set pylinear f_lambda scaling factor
pylinear_flam_scale_fac = 1e-17

# ------------------
grism_sens_cat = np.genfromtxt(home + '/Documents/pylinear_ref_files/' + \
    'pylinear_config/Roman/roman_throughput_20190325.txt', \
    dtype=None, names=True, skip_header=3)

grism_sens_wav = grism_sens_cat['Wave'] * 1e4
# the text file has wavelengths in microns # needed in angstroms
grism_sens = grism_sens_cat['BAOGrism_1st']
grism_wav_idx = np.where(grism_sens > 0.25)
# ------------------

prism_sens_cat = np.genfromtxt(home + '/Documents/pylinear_ref_files/' + \
    'pylinear_config/Roman/roman_throughput_20190325.txt', \
    dtype=None, names=True, skip_header=3)

prism_sens_wav = prism_sens_cat['Wave'] * 1e4
# the text file has wavelengths in microns # needed in angstroms
prism_sens = prism_sens_cat['SNPrism']
prism_wav_idx = np.where(prism_sens > 0.25)

def residual(pars, x, data, err):

    vals = pars.valuesdict()
    zval = vals['z']
    msval = vals['ms']
    ageval = vals['age']
    logtauval = vals['logtau']
    avval = vals['av']

    m0 = model_galaxy(x, z=zval, 
        ms=msval, 
        age=ageval, 
        logtau=logtauval, 
        av=avval)
    r = (m0 - data) / err

    return r

def get_optimal_fit(args_obj, init_guess):

    #print("Running optimizer (LMFIT) to determine initial position.")

    # first pull out required stuff from args
    wav = args_obj[0]
    flam = args_obj[1]
    ferr = args_obj[2]

    x0 = np.where( (wav >= grism_sens_wav[grism_wav_idx][0]  ) &
                   (wav <= grism_sens_wav[grism_wav_idx][-1] ) )[0]

    wav = wav[x0]
    flam = flam[x0]
    ferr = ferr[x0]

    # Initial guess and bounds
    fit_params = Parameters()
    fit_params.add('z', min=init_guess[0] - 0.02, max=init_guess[0] + 0.02)
    fit_params.add('ms', min=init_guess[1] - 0.1, max=init_guess[1] + 0.1)
    fit_params.add('age', min=init_guess[2] - 3.0, max=init_guess[2] + 3.0)
    fit_params.add('logtau', min=init_guess[3] - 0.5, max=init_guess[3] + 0.5)
    fit_params.add('av', min=init_guess[4] - 1.0, max=init_guess[4] + 1.0)

    fitter = Minimizer(residual, fit_params, fcn_args=(wav, flam, ferr))
    out = fitter.minimize(method='brute', Ns=10, workers=-1)

    print("Result from optimal fit:")
    print(fit_report(out))

    """

    m0 = model_galaxy(wav, z=init_guess[0], 
        ms=init_guess[1], 
        age=init_guess[2], 
        logtau=init_guess[3], 
        av=init_guess[4])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(wav, flam)
    ax.plot(wav, m0, label='init guess')
    ax.plot(wav, model_galaxy(wav, *out.brute_x0), label='best fit')
    ax.legend()
    plt.show()
    """

    return out.brute_x0


@njit
def get_dl_at_z(z):

    adiff = np.abs(dl_z_arr - z)
    z_idx = np.argmin(adiff)
    dl = dl_cm_arr[z_idx]

    return dl

@njit
def apply_redshift(restframe_wav, restframe_lum, redshift):

    dl = get_dl_at_z(redshift)
    
    #dl = luminosity_distance(redshift)  # returns dl in Mpc
    #dl = dl * 3.086e24  # convert to cm

    redshifted_wav = restframe_wav * (1 + redshift)
    redshifted_flux = restframe_lum / (4 * np.pi * dl * dl * (1 + redshift))

    return redshifted_wav, redshifted_flux

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

def plot_single_exptime_extraction(sedlst, ext_hdu, disperser='prism'):

    # --------------- plot each spectrum in a for loop
    count = 0
    for i in range(200, len(sedlst)):

        # Get spectra
        segid = sedlst['segid'][i]

        print("\nPlotting SegID:", segid)

        wav = ext_hdu[('SOURCE', segid)].data['wavelength']
        flam = ext_hdu[('SOURCE', segid)].data['flam'] * pylinear_flam_scale_fac

        # Check SNR
        snr = get_snr(wav, flam)

        print("SNR for this spectrum:", "{:.2f}".format(snr))

        # Also get magnitude
        segid_idx = np.where(cat['NUMBER'] == int(segid))[0]
        obj_mag = "{:.3f}".format(float(cat['MAG_AUTO'][segid_idx]))
        print("Object magnitude from SExtractor:", obj_mag)

        if snr < 3.0:
            print("Skipping due to low SNR.")
            continue

        # Set noise level based on snr
        noise_lvl = 1/snr

        # Read in the dummy template passed to pyLINEAR
        template_name = os.path.basename(sedlst['sed_path'][i])
        template_name_list = template_name.split('.txt')[0].split('_')

        # Get template properties
        if 'salt' in template_name:
            
            sn_av = float(template_name_list[-1].replace('p', '.').replace('av',''))
            sn_z = float(template_name_list[-2].replace('p', '.').replace('z',''))
            sn_day = int(template_name_list[-3].replace('day',''))

            print('SN info:')
            print('Redshift:', sn_z)
            print('Phase:', sn_day)
            print('Av:', sn_av)

        else:

            galaxy_av = float(template_name_list[-1].replace('p', '.').replace('av',''))
            galaxy_met = float(template_name_list[-2].replace('p', '.').replace('met',''))
            galaxy_tau = float(template_name_list[-3].replace('p', '.').replace('tau',''))
            galaxy_age = float(template_name_list[-4].replace('p', '.').replace('age',''))
            galaxy_ms = float(template_name_list[-5].replace('p', '.').replace('ms',''))
            galaxy_z = float(template_name_list[-6].replace('p', '.').replace('z',''))

            galaxy_logtau = np.log10(galaxy_tau)

            print('Galaxy info:')
            print('Redshift:', galaxy_z)
            print('Stellar mass:', galaxy_ms)
            print('Age:', galaxy_age)
            print('Tau:', galaxy_tau)
            print('Metallicity:', galaxy_met)
            print('Av:', galaxy_av)

        # Now plot
        fig = plt.figure(figsize=(13,6))
        ax = fig.add_subplot(111)

        ax.set_xlabel(r'$\mathrm{\lambda\ [\AA]}$', fontsize=15)
        ax.set_ylabel(r'$\mathrm{f_\lambda\ [erg\, s^{-1}\, cm^{-2}\, \AA^{-1}]}$', 
            fontsize=15)

        # extracted spectra
        ax.plot(wav, flam, label='Extracted spectrum', lw=1.5)

        # models
        if 'salt' in template_name:
            m = model_sn(wav, sn_z, sn_day, sn_av)
        else:
            m = model_galaxy(wav, galaxy_z, galaxy_ms, galaxy_age, galaxy_logtau, galaxy_av)

        # Only consider wavelengths where sensitivity is above 25%
        if disperser == 'grism':
            x0 = np.where( (wav >= grism_sens_wav[grism_wav_idx][0]  ) &
                           (wav <= grism_sens_wav[grism_wav_idx][-1] ) )[0]
        elif disperser == 'prism':
            x0 = np.where( (wav >= prism_sens_wav[prism_wav_idx][0]  ) &
                           (wav <= prism_sens_wav[prism_wav_idx][-1] ) )[0]

        m = m[x0]
        w = wav[x0]
        flam = flam[x0]

        a, chi2 = get_chi2(m, flam, noise_lvl*flam)

        print("Object a:", "{:.4e}".format(a))
        print("Object base model chi2:", chi2)

        # scale the model
        # using the longer exptime alpha for now
        m = m * a

        ax.plot(w, m, label='model', lw=2.5)

        # Add some text to the plot
        ax.text(x=0.85, y=0.45, s=r'$\mathrm{SegID:\ }$' + str(segid), color='k', 
            verticalalignment='top', horizontalalignment='left', 
            transform=ax.transAxes, size=14)
        ax.text(x=0.85, y=0.4, s=r'$m_{Y106}\, = \, $' + obj_mag, color='k', 
            verticalalignment='top', horizontalalignment='left', 
            transform=ax.transAxes, size=14)
        ax.text(x=0.85, y=0.35, s=r'$\mathrm{SNR}\, = \, $' + "{:.2f}".format(snr), 
            color='k', 
            verticalalignment='top', horizontalalignment='left', 
            transform=ax.transAxes, size=14)

        ax.legend(loc=4, fontsize=14)

        ax.set_xlim(7800, 17800)

        plt.show()

        plt.clf()
        plt.cla()
        plt.close()

        count += 1

        if count > 15: break

    return None

def plot_extractions(sedlst, ext_hdu1, ext_hdu2, ext_hdu3, disperser='prism'):

    # --------------- plot each spectrum in a for loop
    for i in range(len(sedlst)):

        # Get spectra
        segid = sedlst['segid'][i]

        print("\nPlotting SegID:", segid)

        wav1 = ext_hdu1[('SOURCE', segid)].data['wavelength']
        flam1 = ext_hdu1[('SOURCE', segid)].data['flam'] * pylinear_flam_scale_fac

        wav2 = ext_hdu2[('SOURCE', segid)].data['wavelength']
        flam2 = ext_hdu2[('SOURCE', segid)].data['flam'] * pylinear_flam_scale_fac

        wav3 = ext_hdu3[('SOURCE', segid)].data['wavelength']
        flam3 = ext_hdu3[('SOURCE', segid)].data['flam'] * pylinear_flam_scale_fac

        # First check the SNR on the longest exptime
        # Skip if below 3.0
        snr = get_snr(wav3, flam3)

        print("SNR for the 900 s exptime spectrum:", "{:.2f}".format(get_snr(wav1, flam1)))
        print("SNR for the 3600 s exptime spectrum:", "{:.2f}".format(snr))

        # Also get magnitude
        segid_idx = np.where(cat['NUMBER'] == int(segid))[0]
        obj_mag = "{:.3f}".format(float(cat['MAG_AUTO'][segid_idx]))
        print("Object magnitude from SExtractor:", obj_mag)

        if snr < 3.0:
            print("Skipping due to low SNR.")
            continue

        # Set noise level based on snr
        noise_lvl = 1/snr

        # Read in the dummy template passed to pyLINEAR
        template_name = os.path.basename(sedlst['sed_path'][i])
        template_name_list = template_name.split('.txt')[0].split('_')

        # Get template properties
        if 'salt' in template_name:
            
            sn_av = float(template_name_list[-1].replace('p', '.').replace('av',''))
            sn_z = float(template_name_list[-2].replace('p', '.').replace('z',''))
            sn_day = int(template_name_list[-3].replace('day',''))

        else:

            galaxy_av = float(template_name_list[-1].replace('p', '.').replace('av',''))
            galaxy_met = float(template_name_list[-2].replace('p', '.').replace('met',''))
            galaxy_tau = float(template_name_list[-3].replace('p', '.').replace('tau',''))
            galaxy_age = float(template_name_list[-4].replace('p', '.').replace('age',''))
            galaxy_ms = float(template_name_list[-5].replace('p', '.').replace('ms',''))
            galaxy_z = float(template_name_list[-6].replace('p', '.').replace('z',''))

            galaxy_logtau = np.log10(galaxy_tau)

        # Now plot
        fig = plt.figure(figsize=(13,6))
        ax = fig.add_subplot(111)

        ax.set_xlabel(r'$\mathrm{\lambda\ [\AA]}$', fontsize=15)
        ax.set_ylabel(r'$\mathrm{f_\lambda\ [erg\, s^{-1}\, cm^{-2}\, \AA^{-1}]}$', 
            fontsize=15)

        # extracted spectra
        #ax.plot(wav1, flam1, label='900 s')
        #ax.plot(wav2, flam2, label='1800 s')
        ax.plot(wav3, flam3, label='3600 s')

        # models
        if 'salt' in template_name:
            m = model_sn(wav1, sn_z, sn_day, sn_av)
        else:
            m = model_galaxy(wav1, galaxy_z, galaxy_ms, galaxy_age, galaxy_logtau, galaxy_av)

        # Only consider wavelengths where sensitivity is above 25%
        if disperser == 'grism':
            x0 = np.where( (wav1 >= grism_sens_wav[grism_wav_idx][0]  ) &
                           (wav1 <= grism_sens_wav[grism_wav_idx][-1] ) )[0]
        elif disperser == 'prism':
            x0 = np.where( (wav1 >= prism_sens_wav[prism_wav_idx][0]  ) &
                           (wav1 <= prism_sens_wav[prism_wav_idx][-1] ) )[0]

        m = m[x0]
        w = wav1[x0]
        flam1 = flam1[x0]
        flam2 = flam2[x0]
        flam3 = flam3[x0]

        a1, chi2_1 = get_chi2(m, flam1, noise_lvl*flam1)
        a2, chi2_2 = get_chi2(m, flam2, noise_lvl*flam2)
        a3, chi2_3 = get_chi2(m, flam3, noise_lvl*flam3)

        print("Object a for 900 s exptime:", "{:.4e}".format(a1))
        print("Object base model chi2 for 900 s exptime:", chi2_1)

        print("Object a for 1800 s exptime:", "{:.4e}".format(a2))
        print("Object base model chi2 for 1800 s exptime:", chi2_2)

        print("Object a for 3600 s exptime:", "{:.4e}".format(a3))
        print("Object base model chi2 for 3600 s exptime:", chi2_3)

        # scale the model
        # using the longer exptime alpha for now
        m = m * a3

        ax.plot(w, m, label='model')

        # Add some text to the plot
        ax.text(x=0.85, y=0.45, s=r'$\mathrm{SegID:\ }$' + str(segid), color='k', 
            verticalalignment='top', horizontalalignment='left', 
            transform=ax.transAxes, size=14)
        ax.text(x=0.85, y=0.4, s=r'$m_{Y106}\, = \, $' + obj_mag, color='k', 
            verticalalignment='top', horizontalalignment='left', 
            transform=ax.transAxes, size=14)
        ax.text(x=0.85, y=0.35, s=r'$\mathrm{SNR}_{3600}\, = \, $' + "{:.2f}".format(snr), 
            color='k', 
            verticalalignment='top', horizontalalignment='left', 
            transform=ax.transAxes, size=14)

        ax.legend(loc=4, fontsize=14)

        plt.show()

        plt.clf()
        plt.cla()
        plt.close()

    return None

def loglike_galaxy(theta, x, data, err, x0):
    
    z, ms, age, logtau, av = theta

    y = model_galaxy(x, z, ms, age, logtau, av)

    lnLike = get_lnLike_clip(y, data, err)

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
def get_lnLike_clip(y, data, err):

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

def read_pickle_make_plots(object_type, ndim, args_obj, label_list, truth_arr):

    h5_path = 'emcee_sampler_' + object_type + '.h5'
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
        ax1.axhline(y=truth_arr[i], color='tab:red', lw=2.0)
        ax1.set_xlim(0, len(samples))
        ax1.set_ylabel(label_list[i], fontsize=15)
        ax1.yaxis.set_label_coords(-0.1, 0.5)

    axes1[-1].set_xlabel("Step number")

    fig1.savefig('emcee_trace_' + object_type + '.pdf', dpi=200, bbox_inches='tight')

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
        verbose=True, smooth=1.0, smooth1d=1.0, truth_color='tab:red', truths=truth_arr)

    #corner_axes = np.array(fig.axes).reshape((ndim, ndim))

    # redshift is the first axis
    #corner_axes[0, 0].set_title()

    fig.savefig('corner_' + object_type + '.pdf', dpi=200, bbox_inches='tight')

    # ------------ Plot 100 random models from the parameter 
    # space within +-1sigma of corner estimates
    # first pull out required stuff from args
    wav = args_obj[0]
    flam = args_obj[1]
    ferr = args_obj[2]

    fig3 = plt.figure(figsize=(9,4))
    ax3 = fig3.add_subplot(111)

    ax3.set_xlabel(r'$\mathrm{\lambda\ [\AA]}$', fontsize=15)
    ax3.set_ylabel(r'$\mathrm{f_\lambda\ [erg\, s^{-1}\, cm^{-2}\, \AA^{-1}]}$', 
        fontsize=15)

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

    fig3.savefig('emcee_overplot_' + object_type + '.pdf', dpi=200, bbox_inches='tight')

    return None

if __name__ == '__main__':
    
    # --------------- Preliminary stuff
    ext_root = "romansim_prism_"

    img_basename = '5deg_'
    img_suffix = 'Y106_0_6'

    exptime1 = '_900s'
    exptime2 = '_1800s'
    exptime3 = '_3600s'

    # --------------- Read in sed.lst
    sedlst_header = ['segid', 'sed_path']
    sedlst_path = roman_slitless_dir + 'pylinear_lst_files/' + 'sed_' + img_suffix + '.lst'
    sedlst = np.genfromtxt(sedlst_path, dtype=None, names=sedlst_header, encoding='ascii')
    print("Read in sed.lst from:", sedlst_path)

    print("Number of spectra in file:", len(sedlst))

    # --------------- Read in source catalog
    cat_filename = img_sim_dir + img_basename + img_suffix + '.cat'
    cat_header = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'ALPHA_J2000', 'DELTA_J2000', \
    'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_AUTO', 'MAGERR_AUTO', 'FLUX_RADIUS', 'FWHM_IMAGE']
    cat = np.genfromtxt(cat_filename, dtype=None, names=cat_header, encoding='ascii')

    # --------------- Read in the extracted spectra
    ext_spec_filename = ext_spectra_dir + ext_root + img_suffix + exptime3 + '_x1d.fits'
    ext_hdu = fits.open(ext_spec_filename)
    print("Read in extracted spectra from:", ext_spec_filename)

    plot_single_exptime_extraction(sedlst, ext_hdu)

    sys.exit(0)

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


    # ------------------------
    plot_extractions(sedlst, ext_hdu1, ext_hdu2, ext_hdu3)

    sys.exit(0)

    # ------------------------
    # Test the fitting. Basic goals:
    # 1. Obviously, recover the input. 
    # 2. Run on multiple cores (at 100% or close for each core)
    # 3. Non-parametric SFHs

    # Other goals:
    # Test with pyMC3 and Multinest/NUTS/ other packages for multimodal posteriors.
    # Test with Prospector and Dynesty
    # 
    # ------------------------

    segid_to_test = 1
    print("\nTesting fit for SegID:", segid_to_test)

    # Get spectrum
    wav = ext_hdu3[('SOURCE', segid_to_test)].data['wavelength']
    flam = ext_hdu3[('SOURCE', segid_to_test)].data['flam'] * pylinear_flam_scale_fac
    
    # Get snr
    snr = get_snr(wav, flam)

    # Set noise level based on snr
    noise_lvl = 1/snr

    # Create ferr array
    ferr = noise_lvl * flam

    # Only consider wavelengths where sensitivity is above 25%
    x0 = np.where( (wav >= grism_sens_wav[grism_wav_idx][0]  ) &
                   (wav <= grism_sens_wav[grism_wav_idx][-1] ) )[0]

    # Setup for emcee
    # Labels for corner and trace plots
    label_list_galaxy = [r'$z$', r'$\mathrm{log(M_s/M_\odot)}$', r'$\mathrm{Age\, [Gyr]}$', \
    r'$\mathrm{\log(\tau\, [Gyr])}$', r'$A_V [mag]$'] 

    # Set jump sizes # ONLY FOR INITIAL POSITION SETUP
    jump_size_z = 0.5
    jump_size_ms = 1.0  # log(ms)
    jump_size_age = 1.0  # in gyr
    jump_size_logtau = 0.2  # tau in gyr
    jump_size_av = 0.5  # magnitudes

    zprior = 0.5
    zprior_sigma = 0.02

    args_galaxy = [wav, flam, ferr, zprior, zprior_sigma, x0]

    # Initial guess
    rgal_init = np.array([zprior, 11.6, 4.0, 0.5, 2.4])

    # Get optimal position
    #brute_x0 = get_optimal_fit(args_galaxy, rgal_init)

    # Re-initialize to optimal position
    #rgal_init = np.array([brute_x0[0], brute_x0[1], brute_x0[2], brute_x0[3], brute_x0[4]])

    # Setup dims and walkers
    nwalkers = 300
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
    print("ball of walkers will be generated:\n", rgal_init)

    print("logpost at starting position for galaxy:")
    print(logpost_galaxy(rgal_init, wav, flam, ferr, zprior, zprior_sigma, x0))

    # Running emcee
    print("\nRunning emcee...")

    ## ----------- Set up the HDF5 file to incrementally save progress to
    emcee_savefile = 'emcee_sampler_testgalaxy' + str(segid_to_test) + '.h5'
    backend = emcee.backends.HDFBackend(emcee_savefile)
    backend.reset(nwalkers, ndim_gal)

    sampler = emcee.EnsembleSampler(nwalkers, ndim_gal, logpost_galaxy, 
        args=args_galaxy, pool=pool, backend=backend)
    #moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2),],)
    sampler.run_mcmc(pos_gal, 1000, progress=True)

    print("Finished running emcee.")
    print("Mean acceptance Fraction:", np.mean(sampler.acceptance_fraction), "\n")

    # Read in the dummy template passed to pyLINEAR
    sedlst_segid_idx = int(np.where(sedlst['segid'] == segid_to_test)[0])

    template_name = os.path.basename(sedlst['sed_path'][sedlst_segid_idx])
    template_name_list = template_name.split('.txt')[0].split('_')

    # Get template properties
    galaxy_av = float(template_name_list[-1].replace('p', '.').replace('av',''))
    galaxy_met = float(template_name_list[-2].replace('p', '.').replace('met',''))
    galaxy_tau = float(template_name_list[-3].replace('p', '.').replace('tau',''))
    galaxy_age = float(template_name_list[-4].replace('p', '.').replace('age',''))
    galaxy_ms = float(template_name_list[-5].replace('p', '.').replace('ms',''))
    galaxy_z = float(template_name_list[-6].replace('p', '.').replace('z',''))

    galaxy_logtau = np.log10(galaxy_tau)

    # Create truths array and plot
    truth_arr = np.array([galaxy_z, galaxy_ms, galaxy_age, galaxy_logtau, galaxy_av])

    read_pickle_make_plots('testgalaxy' + str(segid_to_test), ndim_gal, 
        args_galaxy, label_list_galaxy, truth_arr)


    sys.exit(0)




