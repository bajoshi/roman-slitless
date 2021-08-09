import numpy as np
from astropy.io import fits

import matplotlib.pyplot as plt

import os
import sys
import socket

# -----------------
if 'plffsn2' in socket.gethostname():
    extdir = '/astro/ffsn/Joshi/'
    modeldir = extdir + 'bc03_output_dir/'
    
    pylinear_lst_dir = extdir + 'pylinear_lst_files/'
    roman_direct_dir = extdir + 'roman_direct_sims/sims2021/'

    roman_slitless_dir = extdir + "GitHub/roman-slitless/"
    fitting_utils = roman_slitless_dir + "fitting_pipeline/utils/"
    ext_spectra_dir = extdir + "/roman_slitless_sims_results/"
    results_dir = ext_spectra_dir + 'fitting_results/'

else:
    extdir = '/Volumes/Joshi_external_HDD/Roman/'
    modeldir = extdir + 'bc03_output_dir/m62/'
    
    pylinear_lst_dir = "/Volumes/Joshi_external_HDD/Roman/pylinear_lst_files/"
    roman_direct_dir = extdir + 'roman_direct_sims/sims2021/'

    home = os.getenv("HOME")
    roman_slitless_dir = home + "/Documents/GitHub/roman-slitless/"
    fitting_utils = roman_slitless_dir + "fitting_pipeline/utils/"
    ext_spectra_dir = "/Volumes/Joshi_external_HDD/Roman/roman_slitless_sims_results/"
    results_dir = ext_spectra_dir + 'fitting_results/'

assert os.path.isdir(modeldir)
assert os.path.isdir(pylinear_lst_dir)
assert os.path.isdir(roman_direct_dir)
assert os.path.isdir(ext_spectra_dir)

# Custom imports
sys.path.append(fitting_utils)
from get_snr import get_snr

# Set pylinear f_lambda scaling factor
pylinear_flam_scale_fac = 1e-17

# Header for SExtractor catalog
cat_header = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'ALPHA_J2000', 'DELTA_J2000', 
'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_AUTO', 'MAGERR_AUTO', 'FLUX_RADIUS', 'FWHM_IMAGE']

if __name__ == '__main__':

    # ---- preliminaries
    ext_root = 'romansim_prism_'
    exptime = '_6000s'

    img_suffix = 'Y106_0_1'

    # ---- Read x1d file
    ext_spec_filename = ext_spectra_dir + ext_root + img_suffix + exptime + '_x1d.fits'
    ext_hdu = fits.open(ext_spec_filename)
    print("Read in extracted spectra from:", ext_spec_filename)

    # ---- read Sextractor catalog
    catfile = roman_direct_dir + 'K_5degimages_part1/' + '5deg_' + img_suffix + '_SNadded.cat'
    cat = np.genfromtxt(catfile, dtype=None, names=cat_header, encoding='ascii')

    # ---- read sed lst
    sedlst_header = ['segid', 'sed_path']
    sedlst_path = pylinear_lst_dir + 'sed_' + img_suffix + '.lst'
    sedlst = np.genfromtxt(sedlst_path, dtype=None, names=sedlst_header, encoding='ascii')

    # --------------- loop and find all SN segids
    all_sn_segids = []
    for i in range(len(sedlst)):
        if 'salt' in sedlst['sed_path'][i]:
            all_sn_segids.append(sedlst['segid'][i])

    print('ALL SN segids in this file:', all_sn_segids)
    print('Total SNe:', len(all_sn_segids))

    # -------- Get SNR and mag for all SNe
    all_sn_mags = []
    all_sn_snr = []

    for segid in all_sn_segids:

        # ---- Get magnitude from catalog
        cat_segid_idx = np.where(cat['NUMBER'] == segid)[0]
        mag = cat['MAG_AUTO'][cat_segid_idx]

        all_sn_mags.append(mag)

        # ---- Get SNR from extracted spectrum
        wav = ext_hdu[('SOURCE', segid)].data['wavelength']
        flam = ext_hdu[('SOURCE', segid)].data['flam'] * pylinear_flam_scale_fac

        snr = get_snr(wav, flam)

        all_sn_snr.append(snr)

    # ---------------------------- SNR vs mag plot
    # Manual entries from running HST/WFC3 spectroscopic ETC
    # For G102 and G141
    etc_mags = np.arange(18.0, 25.5, 0.5)
    etc_g102_snr = np.array([558.0, 414.0, 300.1, 211.89, 145.79, 
                             98.03, 64.68, 42.07, 27.09, 17.32, 
                             11.02, 6.99, 4.43, 2.80, 1.77])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_ylabel('SNR of extracted 1d spec', fontsize=14)
    ax.set_xlabel('F106 mag', fontsize=14)

    ax.scatter(all_sn_mags, all_sn_snr, s=8, color='k', label='pyLINEAR sim result')
    ax.scatter(etc_mags, etc_g102_snr, s=8, color='royalblue', label='WFC3 G102 ETC prediction')

    ax.legend(loc=0, fontsize=14)

    ax.set_yscale('log')

    fig.savefig(results_dir + 'pylinear_sim_snr_vs_mag.pdf', 
        dpi=200, bbox_inches='tight')
    fig.clear()
    plt.close(fig)