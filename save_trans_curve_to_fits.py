from astropy.io import fits
import numpy as np
from scipy.interpolate import griddata

import os
import sys

home = os.getenv('HOME')
pylinear_config_roman_dir = home + \
    '/Documents/pylinear_ref_files/pylinear_config/Roman/'


def save_thru_curve_to_fits(wav, sens, err, disperser, savedir, order=1):

    col1 = fits.Column(name='Wavelength', format='E', array=wav)
    col2 = fits.Column(name='Sensitivity', format='E', array=sens)
    col3 = fits.Column(name='Error', format='E', array=err)
    cols = fits.ColDefs([col1, col2, col3])
   
    thdu = fits.BinTableHDU.from_columns(cols)
   
    p = fits.PrimaryHDU()
    hdul = fits.HDUList(p)
    hdul.append(thdu)
        
    hdul.writeto(savedir + 'Roman_' + disperser + '_' + str(order) + 
                 '_throughput_20190325.fits', overwrite=True)

    return None


def main():

    rt = np.genfromtxt(pylinear_config_roman_dir + 
                       'roman_throughput_20190325.txt',
                       dtype=None, names=True, skip_header=3)

    # Read in sensitivity file
    # See the code sens_file_test.py
    sens_dir = "/Volumes/Joshi_external_HDD/Roman/sensitivity_files/"
    sens = np.genfromtxt(sens_dir + 'Roman_prism_sensitivity.txt', 
                         dtype=None, names=True, encoding='ascii')
    
    # ---------------- Prism ---------------- #
    wp = rt['Wave'] * 1e4  # convert to angstroms from microns

    sens_grid = griddata(points=sens['Wav'], values=sens['Sensitivity'], 
                         xi=wp, fill_value=0.0)

    tp = rt['SNPrism'] * sens_grid

    print("Prism sensitivity:")
    print(tp)

    save_thru_curve_to_fits(wp, tp, np.zeros(len(tp)), 'p127', 
                            pylinear_config_roman_dir)

    sys.exit()

    # ---------------- Grism 1st ---------------- #
    w1 = rt['Wave'] * 1e4  # convert to angstroms from microns
    t1 = rt['BAOGrism_1st'] * senslimit

    save_thru_curve_to_fits(w1, t1, np.zeros(len(t1)), 'g150', order=1)

    # ---------------- Grism 0th ---------------- #
    w0 = rt['Wave'] * 1e4  # convert to angstroms from microns
    t0 = rt['BAOGrism_0th'] * senslimit

    save_thru_curve_to_fits(w0, t0, np.zeros(len(t0)), 'g150', order=0)
    
    return None


if __name__ == '__main__':
    main()
    sys.exit(0)
