import numpy as np
from astropy.io import fits

import os
import sys
import glob

import matplotlib.pyplot as plt

def add_noise_new(sci, exptime):
    sky  = 1.1     # e/s/pix  # zodi + thermal + sky
    npix = 4088 * 4088
    dark = 0.015   # e/s/pix
    read = 10.0    # electrons per read
    read /= npix
    
    size = sci.shape
    print('img of shape:', size)
    
    signal = sci + sky + dark
    variance = signal + read**2
    sigma = np.sqrt(variance)
    
    sci_scaled = sci * exptime
    new_sig = np.random.normal(loc=sci_scaled, scale=sigma, size=size)
    
    final_sig = new_sig / exptime
    
    err = sigma
    
    return final_sig, err

def add_noise_old(sci, exptime):
    sky  = 1.1     # e/s/pix  # zodi + thermal + sky
    npix = 4088 * 4088
    dark = 0.015   # e/s/pix
    read = 10.0    # electrons per read
    read /= npix
    
    size = sci.shape
    print('img of shape:', size)
    
    signal = sci + sky + dark
    variance = signal + read**2
    sigma = np.sqrt(variance)
    
    new_sig = np.random.normal(loc=signal, scale=sigma, size=size)
    
    final_sig = (new_sig / exptime) - sky
    
    err = sigma
    
    return final_sig, err


if __name__ == '__main__':
    
    fltdir = '/Volumes/Joshi_external_HDD/Roman/roman_slitless_sims_results/'

    for fltimg in glob.glob(fltdir + '*_flt.fits'):
        print('Working on image:', fltimg)
        
        fh = fits.open(fltimg)
        sci = fh[1].data
        exptime = 3600

        fsn, en = add_noise_new(sci, exptime)
        fso, eo = add_noise_old(sci, exptime)
        
        fh.close()
        
        h1 = fits.PrimaryHDU(data=fsn, header=fh[1].header)
        h1.writeto(fltimg.replace('.fits', '_new_noised.fits'), overwrite=True)

        h2 = fits.PrimaryHDU(data=fso, header=fh[1].header)
        h2.writeto(fltimg.replace('.fits', '_old_noised.fits'), overwrite=True)

    sys.exit(0)

