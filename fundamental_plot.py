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

    #
    ids = np.array([241, 481, 547, 753])
    z_arr = np.array([1.9404, 0.4141, 1.5955, 1.0326])
    ztruths = np.array([1.953, 0.443, 1.592, 0.918])

    z_acc = (ztruths - z_arr) / (1 + ztruths)

    # snr 162 for id 207

    snr_arr = [6.5, 2.6, 7.1, 2.6]

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(snr_arr, z_acc, s=40, color='royalblue', lw=2.0, facecolors='None')
    ax.axhline(y=0.0, ls='--', color='k')

    ax.set_ylim(-0.08, 0.08)

    plt.show()

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)
