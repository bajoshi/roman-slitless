import numpy as np

from astropy.io import fits
from astropy import units as u
from specutils import Spectrum1D
from specutils.analysis import snr, snr_derived

import matplotlib.pyplot as plt

ext_spectra_dir = home + "/Documents/roman_slitless_sims_results/"

def main():

    ext_root = "romansim_"
    img_suffix = 'Y106_11_1'

    # Read in the sims for each exptime
    sim1 = fits.open(ext_spectra_dir + 'exptime_sims/300s/' + ext_root + img_suffix + '_x1d.fits')
    sim2 = fits.open(ext_spectra_dir + 'exptime_sims/900s/' + ext_root + img_suffix + '_x1d.fits')
    sim3 = fits.open(ext_spectra_dir + 'exptime_sims/2500s/' + ext_root + img_suffix + '_x1d.fits')

    # 

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(snr_arr, z_acc)

    plt.show()

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)
