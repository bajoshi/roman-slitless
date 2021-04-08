import numpy as np
from astropy.io import fits
import scipy
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import griddata

import astropy.units as u
from specutils.analysis import snr_derived
from specutils import Spectrum1D

import os
import sys
import socket
import time
import datetime as dt

import matplotlib.pyplot as plt

home = os.getenv('HOME')
ext_spectra_dir = home + "/Documents/roman_slitless_sims_results/"
roman_slitless_dir = home + "/Documents/GitHub/roman-slitless/"
roman_sims_seds = home + "/Documents/roman_slitless_sims_seds/"

stacking_utils = home + '/Documents/GitHub/stacking-analysis-pears/util_codes/'

sys.path.append(stacking_utils)
import proper_and_lum_dist as cosmo
import dust_utils as du

start = time.time()
print("Starting at:", dt.datetime.now())

# Load in all models
# ------ THIS HAS TO BE GLOBAL!

# Read in SALT2 SN IA file from Lou
salt2_spec = np.genfromtxt(roman_sims_seds + "salt2_template_0.txt", \
    dtype=None, names=['day', 'lam', 'flam'], encoding='ascii')

# Get the dirs correct
if 'plffsn2' in socket.gethostname():
    extdir = '/astro/ffsn/Joshi/'
    modeldir = extdir + 'bc03_output_dir/'
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

print("Done loading all models. Time taken:", "{:.3f}".format(time.time()-start), "seconds.")

# ------------------
grism_sens_cat = np.genfromtxt(home + '/Documents/pylinear_ref_files/pylinear_config/Roman/roman_throughput_20190325.txt', \
    dtype=None, names=True, skip_header=3)

grism_sens_wav = grism_sens_cat['Wave'] * 1e4  # the text file has wavelengths in microns # needed in angstroms
grism_sens = grism_sens_cat['BAOGrism_1st']
grism_wav_idx = np.where(grism_sens > 0.25)
# ------------------

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

        models_taurange_idx = np.argmin(abs(np.arange(tau_low, tau_high, 1) - int(np.floor(tau))))
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
    model_lam_z, model_flam_z = cosmo.apply_redshift(model_lam, model_dusty_llam, z)
    Lsol = 3.826e33
    model_flam_z = Lsol * model_flam_z

    # ------ Apply LSF
    model_lsfconv = gaussian_filter1d(input=model_flam_z, sigma=1.0)

    # ------ Downgrade to grism resolution
    model_mod = griddata(points=model_lam_z, values=model_lsfconv, xi=x)

    return model_mod

def get_chi2(model, flam, ferr, indices=None):

    # Compute a and chi2
    if indices.size:

        a = np.nansum(flam[indices] * model / ferr[indices]**2) / np.nansum(model**2 / ferr[indices]**2)
        model = a*model
        chi2 = np.nansum( (model - flam[indices])**2 / ferr[indices]**2 )

    else:

        a = np.nansum(flam * model / ferr**2) / np.nansum(model**2 / ferr**2)
        model = a*model
        chi2 = np.nansum( (model - flam)**2 / ferr**2 )

    return a, chi2

def get_snr(wav, flux):

    spectrum1d_wav = wav * u.AA
    spectrum1d_flux = flux * u.erg / (u.cm * u.cm * u.s * u.AA)
    spec1d = Spectrum1D(spectral_axis=spectrum1d_wav, flux=spectrum1d_flux)

    return snr_derived(spec1d)

if __name__ == '__main__':
    
    # --------------- Preliminary stuff
    ext_root = "romansim_"

    img_basename = '5deg_'
    img_suffix = 'Y106_0_2'

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

    # Set pylinear f_lambda scaling factor
    pylinear_flam_scale_fac = 1e-17

    # --------------- Read in the extracted spectra
    ext_spec_filename1 = ext_spectra_dir + ext_root + img_suffix + exptime1 + '_x1d.fits'
    ext_hdu1 = fits.open(ext_spec_filename1)
    print("Read in extracted spectra from:", ext_spec_filename1)

    ext_spec_filename2 = ext_spectra_dir + ext_root + img_suffix + exptime2 + '_x1d.fits'
    ext_hdu2 = fits.open(ext_spec_filename2)
    print("Read in extracted spectra from:", ext_spec_filename2)

    ext_spec_filename3 = ext_spectra_dir + ext_root + img_suffix + exptime3 + '_x1d.fits'
    ext_hdu3 = fits.open(ext_spec_filename3)
    print("Read in extracted spectra from:", ext_spec_filename3)

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
            
            sn_av = float(t[-1].replace('p', '.').replace('av',''))
            sn_z = float(t[-2].replace('p', '.').replace('z',''))
            sn_day = int(t[-3].replace('day',''))

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
        ax.set_ylabel(r'$\mathrm{f_\lambda\ [erg\, s^{-1}\, cm^{-2}\, \AA^{-1}]}$', fontsize=15)

        # extracted spectra
        ax.plot(wav1, flam1, label='900 s')
        ax.plot(wav2, flam2, label='1800 s')
        ax.plot(wav3, flam3, label='3600 s')

        # models
        m = model_galaxy(wav1, galaxy_z, galaxy_ms, galaxy_age, galaxy_logtau, galaxy_av)

        # Only consider wavelengths where sensitivity is above 25%
        x0 = np.where( (wav1 >= grism_sens_wav[grism_wav_idx][0]  ) &
                       (wav1 <= grism_sens_wav[grism_wav_idx][-1] ) )[0]
        m = m[x0]
        w = wav1[x0]

        a1, chi2_1 = get_chi2(m, flam1, noise_lvl*flam1, x0)
        a2, chi2_2 = get_chi2(m, flam2, noise_lvl*flam2, x0)
        a3, chi2_3 = get_chi2(m, flam3, noise_lvl*flam3, x0)

        print("Galaxy a for 900 s exptime:", "{:.4e}".format(a1))
        print("Galaxy base model chi2 for 900 s exptime:", chi2_1)

        print("Galaxy a for 1800 s exptime:", "{:.4e}".format(a2))
        print("Galaxy base model chi2 for 1800 s exptime:", chi2_2)

        print("Galaxy a for 3600 s exptime:", "{:.4e}".format(a3))
        print("Galaxy base model chi2 for 3600 s exptime:", chi2_3)

        # scale the model
        # using the longer exptime alpha for now
        m = m * a3

        ax.plot(w, m, label='model')

        # Add some text to the plot
        ax.text(x=0.85, y=0.45, s=r'$\mathrm{SegID:\ }$' + str(segid), color='k', \
            verticalalignment='top', horizontalalignment='left', transform=ax.transAxes, size=14)
        ax.text(x=0.85, y=0.4, s=r'$m_{Y106}\, = \, $' + obj_mag, color='k', \
            verticalalignment='top', horizontalalignment='left', transform=ax.transAxes, size=14)

        ax.legend(loc=4, fontsize=14)

        plt.show()

        

    sys.exit(0)




