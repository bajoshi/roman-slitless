"""
Generates plot of noise level in pyLINEAR sims as a func
of magnitude.

This code measures the average noise level in a spectrum
by measuring the standard deviation of flux in a set of
resolution elements (say using a moving window). This
measurement is repeated for all spectra for objects within
a chosen magnitude bin.
"""

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

import sys

extdir = '/Volumes/Joshi_external_HDD/Roman/'
pylinear_lst_dir = extdir + 'pylinear_lst_files/'
ext_spectra_dir = extdir + 'roman_slitless_sims_results/'

if __name__ == '__main__':

    img_suffix = 'Y106_0_1'
    exptime = '_400s'

    # --------------- Read in sed.lst
    sedlst_header = ['segid', 'sed_path']
    sedlst_path = pylinear_lst_dir + 'sed_' + img_suffix + '.lst'
    sedlst = np.genfromtxt(sedlst_path, dtype=None,
                           names=sedlst_header, encoding='ascii')
    print("Read in sed.lst from:", sedlst_path)

    # --------------- loop and find all SN segids
    all_sn_segids = []
    for i in range(len(sedlst)):
        if ('salt' in sedlst['sed_path'][i]):
            all_sn_segids.append(sedlst['segid'][i])

    print('ALL SN segids in this file:', all_sn_segids)

    # --------------- Read in the extracted spectra
    ext_spec_filename = (ext_spectra_dir + 'romansim_prism_' + img_suffix
                         + exptime + '_x1d.fits')

    ext_hdu = fits.open(ext_spec_filename)
    print("Read in extracted spectra from:", ext_spec_filename)

    # --------------- Set up mag bins
    mag_bins = np.arange(22.0, 29.5, 0.5)

    for i in range(len(all_sn_segids)):
        segid = all_sn_segids[i]

        wav = ext_hdu[('SOURCE', segid)].data['wavelength']
        flam = ext_hdu[('SOURCE', segid)].data['flam'] * 1e-17

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                                       figsize=(13, 4))
        ax1.plot(wav, flam, color='k')

        # Get avg noise
        window = 8
        slist = []
        for j in range(len(wav) - window):
            mu = np.median(flam[j:j + window])
            a = (flam[j:j + window] - mu)**2 / mu**2
            window_noise = np.sqrt(np.sum(a) / window)
            slist.append(window_noise)

        # print(len(slist), len(wav))
        print(np.median(np.array(slist)))

        ax2.scatter(np.arange(len(slist)), slist, s=5, color='k')
        # ax2.set_yscale('log')

        plt.show()
        fig.clear()
        plt.close(fig)

    sys.exit(0)
