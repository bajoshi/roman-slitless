# Notes for this test
# Kevin's images headers state BUNIT = ELECTRONS/S
# Therefore, if multiplied by the exptime the image should be in counts
# and given the correct ZP I should get back the truth mags.
# but clearly from test 0 is the signal is increased by multiplying
# by exptime then the diff between true and sextractor mags will be
# worse.

# Therefore, I think the BUNIT actually is in counts
# but must be divided by the exptime to get to counts
# per second as required.

# ---- TEST 0
# Check magnitudes with direct image as is

# ---- TEST 1
# Check mags with image divided by exptime.
# This is to be used in conjunction with the test_for_zp.py
# code. Run this test here and then run test_for_zp.py.
# It will generate a plot that shows the differences for 
# the mags for the chosen images.

import numpy as np
from astropy.io import fits

import os
import sys
import subprocess
import socket

import matplotlib.pyplot as plt

# ------------------------------------
if 'plffsn2' in socket.gethostname():
    extdir = '/astro/ffsn/Joshi/'
    modeldir = extdir + 'bc03_output_dir/'
    
    roman_sims_seds = extdir + 'roman_slitless_sims_seds/'
    pylinear_lst_dir = extdir + 'pylinear_lst_files/'
    roman_direct_dir = extdir + 'roman_direct_sims/sims2021/'
    roman_slitless_dir = extdir + "GitHub/roman-slitless/"

else:
    extdir = '/Volumes/Joshi_external_HDD/Roman/'
    modeldir = extdir + 'bc03_output_dir/m62/'
    
    roman_sims_seds = "/Volumes/Joshi_external_HDD/Roman/roman_slitless_sims_seds/"
    pylinear_lst_dir = "/Volumes/Joshi_external_HDD/Roman/pylinear_lst_files/"
    roman_direct_dir = extdir + 'roman_direct_sims/sims2021/'
    home = os.getenv("HOME")
    roman_slitless_dir = home + "/Documents/GitHub/roman-slitless/"

assert os.path.isdir(modeldir)
assert os.path.isdir(roman_sims_seds)
assert os.path.isdir(pylinear_lst_dir)
assert os.path.isdir(roman_direct_dir)
# ------------------------------------

def phot_check_f105w(mag, counts, zp):
    """
    Func accepts mag and counts from SExtractor catalog as args
    """

    # Working in AB mags
    m = mag + 48.6
    m /= -2.5
    
    fnu = 10**m
    speed_of_light_Ang = 3e18  # in angstroms per second
    lam_pivot = 10552.0 # angstroms for F105W
    flam = speed_of_light_Ang * fnu / lam_pivot**2  # convert to flam
    
    PHOTFLAM = 10**(-0.4 * zp) * 0.1088 / lam_pivot**2
    flam_pylinear = PHOTFLAM * counts

    # Return # Both flam should be consistent!
    return fnu, flam, flam_pylinear, '{:.3f}'.format(flam/flam_pylinear), PHOTFLAM

def main():

    dir_img_part = 'part1'
    img_sim_dir = roman_direct_dir + 'sextractor_mag_zp_test/'

    """
    # ---------- open test images
    img1 = img_sim_dir + '5deg_Y106_0_2.fits'
    img2 = img_sim_dir + '5deg_Y106_0_15.fits'

    h1 = fits.open(img1)
    h2 = fits.open(img2)
    
    print('Brightness unit for test img 1: ', h1[1].header['BUNIT'])
    print('Brightness unit for test img 2: ', h2[1].header['BUNIT'], '\n')

    # ---------- Preliminary setup
    cat_filename1 = img1.replace('.fits', '.cat')
    checkimage1   = img1.replace('.fits', '_segmap.fits')

    cat_filename2 = img2.replace('.fits', '.cat')
    checkimage2   = img2.replace('.fits', '_segmap.fits')

    # check that single extension file is saved
    img1_sci = img1.replace('.fits', '_sci.fits')
    img2_sci = img2.replace('.fits', '_sci.fits')

    hn1 = fits.PrimaryHDU(data=h1[1].data, header=h1[1].header)
    hn1.writeto(img1_sci, overwrite=True)

    #hn2 = fits.PrimaryHDU(data=h2[1].data, header=h2[1].header)
    #hn2.writeto(img2_sci, overwrite=True)

    # Read in truth file for comparing mags
    truth_match = fits.open(roman_direct_dir + '5deg_truth_gal.fits')

    # ---------- Start the tests
    # ---------------------- TEST 0
    os.chdir(img_sim_dir)

    # Use subprocess to call sextractor.
    # The args passed MUST be passed in this way.
    # i.e., args that would be separated by a space on the 
    # command line must be passed here separated by commas.
    # It will not work if you join all of these args in a 
    # string with spaces where they are supposed to be; 
    # even if the command looks right when printed out.
    if (not os.path.isfile(checkimage1)) or (not os.path.isfile(cat_filename1)):
        sextractor = subprocess.run(['sex', img1_sci, 
            '-c', 'roman_sims_sextractor_config.txt', 
            '-CATALOG_NAME', cat_filename1, 
            '-CHECKIMAGE_NAME', checkimage1], check=True)

    # loop over all catalog objects
    # Only doing this for one img
    # The diff is pretty evident
    cat_header = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'ALPHA_J2000', 'DELTA_J2000', 
     'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_AUTO', 'MAGERR_AUTO', 'FLUX_RADIUS', 'FWHM_IMAGE']
    cat1 = np.genfromtxt(cat_filename1, dtype=None, names=cat_header, encoding='ascii')

    ra_tol = 0.5 / 3600
    dec_tol = 0.5 / 3600

    allra  = truth_match[1].data['ra']  * 180.0/np.pi  # since the truth's are in radians
    alldec = truth_match[1].data['dec'] * 180.0/np.pi  # since the truth's are in radians

    magdiff_list = []

    for i in range(len(cat1)):
        current_ra = cat1['ALPHA_J2000'][i]
        current_dec = cat1['DELTA_J2000'][i]

        truth_idx = np.where((allra >= current_ra - ra_tol) & (allra <= current_ra + ra_tol) \
                           & (alldec >= current_dec - dec_tol) & (alldec <= current_dec + dec_tol))[0]

        if (truth_idx.size) and (len(truth_idx) == 1):  # not all objects are present in truth file
            truth_idx = int(truth_idx)

            sexmag = cat1['MAG_AUTO'][i]
            truthmag = truth_match[1].data['Y106'][truth_idx]

            print('\nObject: ', i+1)
            print(truth_idx)
            print(truth_match[1].data['bflux'][truth_idx], truth_match[1].data['dflux'][truth_idx])  # background and dark flux?
            magdiff = truthmag - sexmag
            print(sexmag, truthmag, magdiff)

            magdiff_list.append(magdiff)

    magdiff_list = np.asarray(magdiff_list)
    print('Average mag diff [True - SExtractor]:', np.mean(magdiff_list))
    print('Done with test 0.-----------------\n')
    """
    # ----------------------------------------------
    # # ---------------------- TEST 1

    # Scaling factor for direct images
    # The difference here that appears in the power of 10
    # is the difference between the ZP of the current img
    # and what I think the correct ZP is i.e., the WFC3/F105W ZP.
    dirimg_scaling = 10**(-0.4 * (31.7956 - 26.264))

    # List of images to test
    img_suffix_list = ['Y106_0_1', 'Y106_0_5', 'Y106_0_9', 'Y106_0_12', 'Y106_0_13']

    for img_suffix in img_suffix_list:

        img = img_sim_dir + '5deg_' + img_suffix + '.fits'
        img_cps = img.replace('.fits', '_cps.fits')

        # First divide and save all cps imgs
        if not os.path.isfile(img_cps):
            dir_hdu = fits.open(img)
            cps_sci_arr = dir_hdu[1].data * dirimg_scaling
            cps_hdr = dir_hdu[1].header
            dir_hdu.close()

            cps_hdr['BUNIT'] = 'CPS'
            
            chdu = fits.PrimaryHDU(data=cps_sci_arr, header=cps_hdr)
            chdu.writeto(img_cps, overwrite=True)
            print('Written:', img_cps)

        # Set up and run sextractor
        checkimage = img_cps.replace('.fits', '_segmap.fits')
        cat_filename = img_cps.replace('.fits', '.cat')

        if (not os.path.isfile(checkimage)) or (not os.path.isfile(cat_filename)):
            sextractor = subprocess.run(['sex', img_cps, 
                '-c', 'roman_sims_sextractor_config.txt', 
                '-CATALOG_NAME', cat_filename, 
                '-CHECKIMAGE_NAME', checkimage], check=True)

    print('\n *** Prepped next test. Run test_for_zp.py now.\n')

    # Close hdus
    #h1.close()
    #h2.close()

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)