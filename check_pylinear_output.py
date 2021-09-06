import numpy as np
from astropy.io import fits
from astropy.convolution import convolve, Box1DKernel

import matplotlib.pyplot as plt
from tqdm import tqdm

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
sys.path.append(roman_slitless_dir + 'fitting_pipeline/')
sys.path.append(fitting_utils)
from get_snr import get_snr
from get_template_inputs import get_template_inputs
from fit_sn_romansim import model_sn

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

    showplot = False

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
    
    # -------- Get SNR and mag for all objects
    all_sn_mags = []
    all_sn_snr  = []
    
    all_galaxy_mags = []
    all_galaxy_snr  = []

    snr_per_pix_z1 = []
    noise_per_pix_z1 = []

    smooth_snr_list = []
    
    for i in tqdm(range(len(sedlst)), desc='Processing object'):
    
        # First match with catalog
        current_segid = sedlst['segid'][i]
        cat_idx = np.where(cat['NUMBER'] == current_segid)[0]
    
        # now get magnitude
        mag = cat['MAG_AUTO'][cat_idx]
    
        # Get spectrum from extracted file adn SNR
        wav = ext_hdu[('SOURCE', current_segid)].data['wavelength']
        flam = ext_hdu[('SOURCE', current_segid)].data['flam'] * pylinear_flam_scale_fac
        snr = get_snr(wav, flam)
    
        # Append to appropriate lists depending on object type
        if 'salt' in sedlst['sed_path'][i]:
            all_sn_mags.append(mag)
            all_sn_snr.append(snr)
        else:
            all_galaxy_mags.append(mag)
            all_galaxy_snr.append(snr)

        # Do the z~1 comparison
        # ------- Get template inputs
        template_name = os.path.basename(sedlst['sed_path'][i])
        if 'salt' in template_name:
            template_inputs = get_template_inputs(template_name)  # needed for plotting
            ztrue = template_inputs[0]

            if 0.9 <= ztrue <= 1.1:
                # Get noise array
                ferr_lo = ext_hdu[('SOURCE', current_segid)].data['flounc'] * pylinear_flam_scale_fac
                ferr_hi = ext_hdu[('SOURCE', current_segid)].data['fhiunc'] * pylinear_flam_scale_fac

                ferr = (ferr_lo + ferr_hi)/2.0
                snr_per_pix = flam / ferr

                snr_per_pix_z1.append(snr_per_pix)
                noise_per_pix_z1.append(ferr)

                wav_idx = np.where((wav > 8000.0) & (wav < 17800.0))[0]

                wav = wav[wav_idx]
                ferr = ferr[wav_idx]
                snr_per_pix = snr_per_pix[wav_idx]
                flam = flam[wav_idx]

                # Plot smoothed spectrum to see how 
                # close it is to truth
                sf = convolve(flam, Box1DKernel(4))

                # Now noise the smooth spectrum using the correct noise
                # level and overplot
                noise_correct = ferr * 5 #np.sqrt(5)
                sf_noised = np.zeros(len(sf))
                for w in range(len(wav)):
                    sf_noised[w] = np.random.normal(loc=sf[w], scale=noise_correct[w], size=1)

                smooth_snr = sf_noised/noise_correct
                smooth_snr_list.append(np.mean(smooth_snr))

                if showplot:

                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.set_xlabel('Wavelength', fontsize=15)
                    ax.set_ylabel('Noise [cgs]', fontsize=15)

                    # Plot SNR as a func of wav
                    axt = ax.twinx()
                    axt.set_ylabel('SNR')
                    #axt.plot(wav, snr_per_pix, color='green', zorder=1)
                    
                    # Plot noise as a func of wav
                    #ax.plot(wav, ferr, color='crimson', zorder=1)

                    # Plot extracted spectrum
                    ax.plot(wav, flam, color='k', lw=0.5, zorder=2)
                    ax.fill_between(wav, flam-ferr, flam+ferr, color='gray', alpha=0.5)

                    ax.plot(wav, sf, lw=4.0, color='cyan', zorder=3)  # smoothed spec
                    ax.plot(wav, sf_noised, color='b')  # Noised smooth spec

                    # Plot SNR for smoothed spec
                    axt.plot(wav, smooth_snr, color='seagreen', lw=1.0)
                    print('Avg SNR per pix in 5 hour exptime:', np.mean(smooth_snr))
                    
                    # Also plot noise

                    # Plot true spectrum
                    ax.plot(wav, model_sn(wav, ztrue, template_inputs[1], template_inputs[2]), 
                        color='deeppink', lw=2.5, zorder=4)

                    # Set limits and show
                    ax.set_xlim(8000, 17800)

                    #plt.show()

                    fig.clear()
                    plt.close(fig)

    print('Average smoothed SNR list:', smooth_snr_list, len(smooth_snr_list))
    print('Mean smooth SNR:', np.mean(np.array(smooth_snr_list)))

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

    ax.scatter(all_galaxy_mags, all_galaxy_snr, marker='o', s=6, 
        color='k', label='pyLINEAR sim result, galaxies', zorder=1)
    ax.scatter(all_sn_mags, all_sn_snr,         marker='o', s=7, 
        color='crimson', facecolors='None', label='pyLINEAR sim result, SNe', zorder=2)
    ax.scatter(etc_mags, etc_g102_snr, s=8, color='royalblue', zorder=3,
        label='WFC3 G102 ETC prediction' + '\n' + 'Exptime: 18000s')

    ax.legend(loc=0, fontsize=11)

    ax.set_yscale('log')

    ax.set_xlim(17.03, 25.38)

    fig.savefig(results_dir + 'pylinear_sim_snr_vs_mag.pdf', 
        dpi=200, bbox_inches='tight')
    fig.clear()
    plt.close(fig)


    