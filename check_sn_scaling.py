import numpy as np
from astropy.io import fits

import os, sys
import subprocess

import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    # ---------------
    img_sim_dir = '/Volumes/Joshi_external_HDD/Roman/roman_direct_sims/sims2021/K_5degimages_part1/'
    roman_slitless_dir = '/Users/baj/Documents/GitHub/roman-slitless/'

    # Read in the reference image of the star from 
    # Y106_0_6 at 72.1175280, -53.5739388 (~16th mag)
    ref_cutout_path = img_sim_dir + 'ref_cutout_psf.fits'

    # Explicitly run SExtractor again to get the 
    # magnitude  and counts 
    refdat = fits.getdata(ref_cutout_path)

    os.chdir(img_sim_dir)

    checkimage   = ref_cutout_path.replace('.fits', '_segmap.fits')
    sextractor   = subprocess.run(['sex', os.path.basename(ref_cutout_path), 
        '-PARAMETERS_NAME', 'ref.param',
        '-DETECT_MINAREA', '100',
        '-MAG_ZEROPOINT', '26.264',
        '-CATALOG_NAME', 'ref.cat',
        '-CHECKIMAGE_TYPE', 'SEGMENTATION',
        '-CHECKIMAGE_NAME', checkimage], check=True)

    os.chdir(roman_slitless_dir)

    # Now read in the catalog just generated and get the counts
    cat_header = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 
        'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_AUTO', 'MAGERR_AUTO',]
    cat = np.genfromtxt(img_sim_dir + 'ref.cat', 
        dtype=None, names=cat_header, encoding='ascii')

    ref_counts = cat['FLUX_AUTO']

    # ---------------
    print('\nSum over full reference cutout:', np.sum(refdat))
    print('Counts from SExtractor:', ref_counts)
    print('Mag from SExtractor:', cat['MAG_AUTO'])

    # ---------------
    # Find the source in the segmap
    segmap = fits.getdata(checkimage)
    segpix = np.where(segmap == 1)

    ref_summed_counts = np.sum(refdat[segpix[0], segpix[1]])

    npix = len(segpix[0])
    back = 4.70201e-6  # Read this from terminal SExtractor output
    total_back = back * npix

    print('Explicity summed counts over seg pix:', ref_summed_counts, '\n')
    print(ref_summed_counts - total_back)

    print('Explicity summed counts will not have background subtracted')
    print('so they are expected to be higher than the counts quoted by SExtractor????\n')

    # Now scale the reference and see how the derived counts 
    # deviate from the required counts.



    sys.exit(0)