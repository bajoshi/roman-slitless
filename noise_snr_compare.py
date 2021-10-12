import numpy as np
from astropy.io import fits
from scipy.interpolate import griddata

import os
import sys

import matplotlib.pyplot as plt

home = os.getenv('HOME')
roman_slitless_dir = home + '/Documents/GitHub/roman-slitless/'

def get_avg_pylinear_z1spec():

    # Read in results file
    ext_spectra_dir = "/Volumes/Joshi_external_HDD/Roman/roman_slitless_sims_results/"
    results_dir = ext_spectra_dir + 'fitting_results/'
    
    resfile = results_dir + 'zrecovery_pylinear_sims_pt0.txt'
    cat = np.genfromtxt(resfile, dtype=None, names=True, encoding='ascii')

    # Also read in x1d file
    one_hr_x1d = ext_spectra_dir + 'romansim_prism_Y106_0_1_1200s_x1d.fits'
    ext_hdu = fits.open(one_hr_x1d)

    # Now get all spectra that are 1 hour exptime and close to z~1
    # ------------ Gather all spectra
    # First need the wavelengths 
    # Since the wav array is always the same just get the first one
    wav = ext_hdu[1].data['wavelength']

    z1_bin_idx = np.where((cat['z_true'] >= 0.97) & (cat['z_true'] <= 1.03))[0]
    print('\nAveraging', len(z1_bin_idx), 'spectra...')

    all_spec  = np.zeros((len(z1_bin_idx), len(wav)))
    all_noise = np.zeros((len(z1_bin_idx), len(wav)))
    avg_snr_1hr = []

    for s in range(len(z1_bin_idx)):
        snr_1hr = cat['SNR1200'][z1_bin_idx][s]
        avg_snr_1hr.append(snr_1hr)

        mag = cat['Y106mag'][z1_bin_idx][s]
        z = cat['z_true'][z1_bin_idx][s]

        segid = cat['SNSegID'][z1_bin_idx][s]

        all_spec[s]  = ext_hdu[('SOURCE', segid)].data['flam'] * 1e-17

        ferr_lo = ext_hdu[('SOURCE', segid)].data['flounc'] * 1e-17
        ferr_hi = ext_hdu[('SOURCE', segid)].data['fhiunc'] * 1e-17
        all_noise[s] = (ferr_lo + ferr_hi)/2

        # print(s, 
        #       '{:.2f}'.format(mag),
        #       '{:.3f}'.format(z),
        #       '{:.3f}'.format(snr_1hr))

    avg_snr = np.mean(np.array(avg_snr_1hr))

    print('Average SNR for 1-hour exposures of z~1 SNe:', 
          '{:.3f}'.format(avg_snr), '\n')

    # ----------- Average the spectra and return
    mean_spec = np.mean(all_spec, axis=0)
    mean_noise = np.mean(all_noise, axis=0)

    # Close open HDUs
    ext_hdu.close()

    return wav, mean_spec, mean_noise

if __name__ == '__main__':

    # pyLINEAR spectra
    wav, mean_spec, mean_noise = get_avg_pylinear_z1spec()

    # Rubin's scaled spectrum from Jeff Kruk
    comp = np.genfromtxt("/Volumes/Joshi_external_HDD/Roman/sensitivity_files/Rubin_SNIa_z1_1hr.txt",
                         dtype=None, names=True, encoding='ascii')

    # Scaling factor 
    # This is needed because the Kruk et al spectrum is 1000
    # seconds and the comparison pylinear spectrum here is
    # 3600 seconds
    scalefac = 3.6

    # -------------------------
    # Comparison figures
    fig = plt.figure(figsize=(9,6))
    fig.suptitle('z=1 SN Ia, 1000 seconds prism exposure', fontsize=15, y=0.94)
    gs = fig.add_gridspec(nrows=5, ncols=1, left=0.05, right=0.95, wspace=0.1)
    ax1 = fig.add_subplot(gs[:3])
    ax2 = fig.add_subplot(gs[3:])

    # SNR
    ax1.plot(wav, mean_spec / scalefac, lw=2.0, color='firebrick', label='pyLINEAR flux')
    ax1.plot(comp['waveA'], comp['signal'], lw=2.0, color='blue', label='Jeff Kruk/Rubin et al. flux')

    axt = ax1.twinx()

    # Noise
    axt.plot(wav, mean_noise, ls='--', lw=1.5, color='firebrick', label='pyLINEAR noise')
    axt.plot(comp['waveA'], comp['noise'], ls='--', lw=1.5, color='blue', label='Jeff Kruk/Rubin et al. noise')

    ax1.legend(loc='upper right', frameon=False, fontsize=12)
    axt.legend(loc='upper center', frameon=False, fontsize=12)

    #ax1.set_ylim(0, 45.0)
    ax1.set_xlim(7700, 18000)
    axt.set_ylim(1e-23, 2e-19)

    # -------- plot differences
    mean_noise_regrid = griddata(points=wav, values=mean_noise, xi=comp['waveA'])
    mean_spec_regrid  = griddata(points=wav, values=mean_spec/scalefac,  xi=comp['waveA'])

    #ax2.plot(comp['waveA'], comp['noise']/mean_noise_regrid, color='deeppink', label='Noise1/Noise2')
    ax2.plot(comp['waveA'], mean_spec_regrid/comp['signal'], color='deeppink', label='Flux1/Flux2')

    ax2.legend(loc='upper center', frameon=False, fontsize=12)

    # ------- Labels and ticks
    ax1.set_xticks([])

    ax2.set_xlabel('Wavelength', fontsize=15)
    ax2.set_ylabel('sim1/sim2', fontsize=15)

    ax1.set_ylabel('F-lambda', fontsize=15)
    axt.set_ylabel('Noise', fontsize=15)

    fig.savefig(roman_slitless_dir + 'figures/snr_noise_comparison.pdf', 
                dpi=300, bbox_inches='tight')

    sys.exit(0)



