import numpy as np

from astropy.io import fits
from astropy import units as u
from specutils import Spectrum1D
from specutils.analysis import snr, snr_derived

import os
import sys

import matplotlib.pyplot as plt

home = os.getenv('HOME')
ext_spectra_dir = home + "/Documents/roman_slitless_sims_results/"

def main():

    ext_root = "romansim_"
    img_suffix = 'Y106_11_1'

    # Read in the sims for each exptime
    #sim1 = fits.open(ext_spectra_dir + 'exptime_sims/300s/' + ext_root + img_suffix + '_x1d.fits')
    #sim2 = fits.open(ext_spectra_dir + 'exptime_sims/900s/' + ext_root + img_suffix + '_x1d.fits')
    #sim3 = fits.open(ext_spectra_dir + 'exptime_sims/2500s/' + ext_root + img_suffix + '_x1d.fits')

    # ----------------------------- For SNe
    ids_sn = np.array([241, 481, 547, 753])

    z_arr_sn = np.array([1.94044061, 0.41406845, 1.59551759, 1.03262136])
    ztruths_sn = np.array([1.953, 0.443, 1.592, 0.918])

    # Get the snr and redshfit accuracy
    snr_arr_sn = [6.5, 2.6, 7.1, 2.6]
    z_acc_sn = (ztruths_sn - z_arr_sn) / (1 + ztruths_sn)

    # Now get errors on redshifts
    zuppers_sn = np.array([1.94060596, 0.41458185, 1.59578856, 1.0326508])
    zlowers_sn = np.array([1.94028437, 0.41379181, 1.59534436, 1.03255838])

    z_err_sn = np.array([[zuppers_sn - z_arr_sn], [z_arr_sn - zlowers_sn]])
    z_err_sn = z_err_sn.reshape((2,len(ztruths_sn)))

    # ----------------------------- For galaxies
    # snr 162 for id 207
    ids_gal = np.array([207])

    z_arr_gal = np.array([1.9524])
    ztruths_gal = np.array([1.953])

    # Get the snr and redshfit accuracy
    snr_arr_gal = [10.05]
    z_acc_gal = (ztruths_gal - z_arr_gal) / (1 + ztruths_gal)

    # Now get errors on redshifts
    zuppers_gal = np.array([1.9526])
    zlowers_gal = np.array([1.9522])

    z_err_gal = np.array([[zuppers_gal - z_arr_gal], [z_arr_gal - zlowers_gal]])
    z_err_gal = z_err_gal.reshape((2,len(ztruths_gal)))

    # print redshift accuracies to screen
    print(z_acc_sn)
    print(z_acc_gal)

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel(r'$\mathrm{SNR}$', fontsize=16)
    ax.set_ylabel(r'$\frac{\Delta z}{1 + z_\mathrm{truth}}$', fontsize=16)

    ax.errorbar(snr_arr_gal, z_acc_gal, yerr=z_err_gal, fmt='o', \
        markersize=5.0, markerfacecolor='k', markeredgecolor='k', ecolor='gray', elinewidth=1.2, label='Galaxies')

    ax.errorbar(snr_arr_sn, z_acc_sn, yerr=z_err_sn, fmt='o', \
        markersize=5.0, markerfacecolor='None', markeredgecolor='k', ecolor='gray', elinewidth=1.2, label='SNe')
    ax.axhline(y=0.0, ls='--', color='gray')

    ax.set_ylim(-0.08, 0.08)

    ax.legend()

    fig.savefig('snr_vs_z_accuracy.pdf', dpi=200, bbox_inches='tight')

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)
