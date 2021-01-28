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

    # For SNe
    ids_sn = np.array([241, 481, 547, 753])
    z_arr_sn = np.array([1.9404, 0.4141, 1.5955, 1.0326])
    ztruths_sn = np.array([1.953, 0.443, 1.592, 0.918])

    snr_arr_sn = [6.5, 2.6, 7.1, 2.6]

    z_acc_sn = (ztruths_sn - z_arr_sn) / (1 + ztruths_sn)

    # For galaxies
    # snr 162 for id 207
    ids_gal = np.array([207])
    z_arr_gal = np.array([1.9528])
    ztruths_gal = np.array([1.953])

    snr_arr_gal = [162]

    z_acc_gal = (ztruths_gal - z_arr_gal) / (1 + ztruths_gal)

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel(r'$\mathrm{SNR}$', fontsize=16)
    ax.set_ylabel(r'$\frac{\Delta z}{1 + z_\mathrm{truth}}$', fontsize=16)

    #ax.scatter(snr_arr_gal, z_acc_gal, s=40, color='k')

    ax.scatter(snr_arr_sn, z_acc_sn, s=40, color='k', lw=2.2, facecolors='None')
    ax.axhline(y=0.0, ls='--', color='k')

    ax.set_ylim(-0.08, 0.08)

    plt.show()

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)
