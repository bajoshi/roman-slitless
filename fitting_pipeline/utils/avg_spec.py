import matplotlib.pyplot as plt

from astropy.io import fits
import numpy as np

def get_avg_spec(redshift, ehdu):



    return mean_spec

def plot_avg_spec(cat, all_exptimes):

    ext_spectra_dir = '/Volumes/Joshi_external_HDD/Roman/roman_slitless_sims_results/'
    img_suffix = 'Y106_0_1'

    # ----- Read in x1d file to get spectra
    ext_spec_filename1 = ext_spectra_dir + 'romansim_prism_' + img_suffix + all_exptimes[0] + '_x1d.fits'
    ext_hdu1 = fits.open(ext_spec_filename1)

    ext_spec_filename2 = ext_spectra_dir + 'romansim_prism_' + img_suffix + all_exptimes[1] + '_x1d.fits'
    ext_hdu2 = fits.open(ext_spec_filename2)

    ext_spec_filename3 = ext_spectra_dir + 'romansim_prism_' + img_suffix + all_exptimes[2] + '_x1d.fits'
    ext_hdu3 = fits.open(ext_spec_filename3)

    ext_spec_filename4 = ext_spectra_dir + 'romansim_prism_' + img_suffix + all_exptimes[3] + '_x1d.fits'
    ext_hdu4 = fits.open(ext_spec_filename4)

    # --------------------- PLOT
    fig, axes = plt.subplots(figsize=(13,6), nrows=1, ncols=4)
    plt.show()

    # Close hdu
    ext_hdu1.close()
    ext_hdu2.close()
    ext_hdu3.close()
    ext_hdu4.close()

    return None