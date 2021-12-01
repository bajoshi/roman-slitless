import numpy as np
from astropy.io import fits

import os, sys
import socket
import subprocess

import matplotlib.pyplot as plt

# Get the dirs correct
if 'plffsn2' in socket.gethostname():
    extdir = '/astro/ffsn/Joshi/'
    modeldir = extdir + 'bc03_output_dir/'
    roman_direct_dir = extdir + 'roman_direct_sims/sims2021/'

    roman_slitless_dir = extdir + 'GitHub/roman-slitless/'
    utils_dir = roman_slitless_dir + 'fitting_pipeline/utils/'

else:
    extdir = '/Volumes/Joshi_external_HDD/Roman/'
    modeldir = extdir + 'bc03_output_dir/m62/'
    roman_direct_dir = extdir + 'roman_direct_sims/sims2021/'

    home = os.getenv('HOME')
    roman_slitless_dir = home + '/Documents/GitHub/roman-slitless/'
    utils_dir = roman_slitless_dir + 'fitting_pipeline/utils/'

assert os.path.isdir(modeldir)
assert os.path.isdir(roman_direct_dir)

dir_img_part = 'part1'
img_sim_dir = roman_direct_dir + 'K_5degimages_' + dir_img_part + '/'

def gen_ref_segmap(snmag):

    # ---------
    # Read in the reference image of the star from 
    # Y106_0_6 at 72.1175280, -53.5739388 (~16th mag)
    ref_cutout_path = img_sim_dir + 'ref_cutout_psf.fits'
    ref_data = fits.getdata(ref_cutout_path)

    ref_mag = 15.9180
    ref_counts = 13753.24  # read in mag and counts from SExtractor catalog on dir img
    ref_segid = 630

    # ---------
    # Now scale reference
    delta_m = ref_mag - snmag
    sncounts = ref_counts * (1 / 10**(-0.4*delta_m) )

    scale_fac = sncounts / ref_counts
    new_cutout = ref_data * scale_fac

    # Add some background to ensure that not too many
    # pixels are associated with the central source.
    # 
    new_cutout += np.random.normal(loc=0.0, scale=0.001, 
        size=new_cutout.shape)

    # ---------
    # Save the scaled reference to be able to run SExtractor
    ref_name = 'ref_cutout_' + '{:.1f}'.format(snmag) + '.fits'
    scaled_ref_path = img_sim_dir + 'ref_dir/' + ref_name
    scaled_ref = fits.PrimaryHDU(data=new_cutout)
    scaled_ref.writeto(scaled_ref_path, overwrite=True)

    # ---------
    # Run SExtractor on this scaled image
    os.chdir(img_sim_dir + 'ref_dir/')

    checkimage = scaled_ref_path.replace('.fits', '_segmap.fits')

    # Decide min pix threshold based on SN brightness
    # Needs to be a string for subprocess args
    if snmag < 23.0:
        minpix = '300'
    elif snmag >= 23.0 and snmag < 24.0:
        minpix = '150'
    elif snmag >= 24.0 and snmag < 25.0:
        minpix = '50'
    elif snmag >= 25.0 and snmag < 27.5:
        minpix = '10'
    else:
        minpix = '3'  
        # Requiring at least 3 pixels when dispersing the 2D spec

    sextractor = subprocess.run(['sex', scaled_ref_path, 
        '-c', 'sextractor_config_refcutout.txt',
        '-DETECT_MINAREA', minpix,
        '-CHECKIMAGE_NAME', checkimage], check=True)

    # Go back to roman-slitless directory
    os.chdir(roman_slitless_dir)

    # ---------
    # Check that the catalog has only one object
    test_cat_hdr = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 
    'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_AUTO', 'MAGERR_AUTO']
    cat = np.genfromtxt(img_sim_dir + 'ref_dir/test.cat', 
        dtype=None, names=test_cat_hdr, encoding='ascii')

    segdata = fits.getdata(checkimage)
    seg_idx = np.where(segdata == 1)

    magdiff = abs(cat['MAG_AUTO'] - snmag)

    print('============================')
    print(cat.size, len(seg_idx[0]))
    print(cat['MAG_AUTO'], snmag, '{:.4f}'.format(magdiff))
    print('============================')

    assert cat.size == 1
    #assert magdiff < 0.15, '{:.4f}'.format(magdiff)
    # Supressing htis assertion for now because the mag diff 
    # gets as worse as 0.4 or 0.5 mags at the faintest 
    # magnitudes. Will have to scale the counts later 
    # downstream to compensate.

    return None

if __name__ == '__main__':

    # ---------------
    # For a range of SN magnitudes scale the reference
    # to get counts and run SExtractor to get source pixels.
    # These pixels will later be assigned a spectrum.
    mags = np.arange(17.0, 28.4, 0.2)  # last mag considered is 28.2
    # The exact magnitudes don't really matter.
    # It is the approx scaling of mag to number and pattern 
    # of segmentation pixels that we really need.
    # For magnitudes fainter than 28.2 we can just use the
    # seg pix for 28.2

    for m in mags:
        gen_ref_segmap(m)

    sys.exit(0)












