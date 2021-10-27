import numpy as np
from astropy.io import fits
from scipy.interpolate import griddata

import os
import sys

import matplotlib.pyplot as plt

home = os.getenv('HOME')
roman_slitless_dir = home + '/Documents/GitHub/roman-slitless/'

sys.path.append(roman_slitless_dir + 'fitting_pipeline/utils/')
from get_template_inputs import get_template_inputs
from get_snr import get_snr
from get_all_sn_segids import get_all_sn_segids

def get_avg_pylinear_z1spec():

    # Set directories
    ext_spectra_dir = "/Volumes/Joshi_external_HDD/Roman/roman_slitless_sims_results/"
    img_sim_dir = "/Volumes/Joshi_external_HDD/Roman/roman_direct_sims/sims2021/K_5degimages_part1/"
    pylinear_lst_dir = "/Volumes/Joshi_external_HDD/Roman/pylinear_lst_files/"

    # Now get all spectra that are 1 hour exptime and close to z~1
    # ------------ Gather all spectra
    all_spec  = []
    all_noise = []
    avg_snr_1hr = []

    total_spectra = 0

    for i in range(3):

        print('----------')

        sedlst_fl = pylinear_lst_dir + 'sed_Y106_0_' + str(i+1) + '.lst'
        sedlst = np.genfromtxt(sedlst_fl, dtype=None, names=['segid', 'sed_path'], encoding='ascii')

        all_sn_segids = get_all_sn_segids(sedlst_fl)

        one_hr_x1d = ext_spectra_dir + 'romansim_prism_Y106_0_' + str(i+1) + '_1200s_x1d.fits'
        xhdu = fits.open(one_hr_x1d)

        for j in range(len(all_sn_segids)):

            segid = all_sn_segids[j]

            sed_idx = int(np.where(sedlst['segid'] == segid)[0])
            template_name = sedlst['sed_path'][sed_idx]
            inp = get_template_inputs(template_name)
            ztrue = inp[0]

            # Check if the redshift is okay
            if (ztrue >= 0.97) and (ztrue <= 1.03):

                # Get spectrum
                wav = xhdu[('SOURCE', segid)].data['wavelength']
                flam = xhdu[('SOURCE', segid)].data['flam'] * 1e-17

                ferr_lo = xhdu[('SOURCE', segid)].data['flounc'] * 1e-17
                ferr_hi = xhdu[('SOURCE', segid)].data['fhiunc'] * 1e-17
                noise = (ferr_lo + ferr_hi)/2

                snr = get_snr(wav, flam)

                # Append
                all_spec.append(flam)
                all_noise.append(noise)
                avg_snr_1hr.append(snr)

                total_spectra += 1

                print(segid, ztrue, '{:.2f}'.format(snr))

        xhdu.close()

    print('\nAveraging', total_spectra, 'spectra...')

    all_spec = np.array(all_spec)
    all_noise = np.array(all_noise)

    avg_snr = np.mean(np.array(avg_snr_1hr))
    print('Average SNR for 1-hour exposures of z~1 SNe:', 
          '{:.3f}'.format(avg_snr), '\n')

    # ----------- Average the spectra and return
    mean_spec = np.mean(all_spec, axis=0)
    mean_noise = np.mean(all_noise, axis=0)

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



