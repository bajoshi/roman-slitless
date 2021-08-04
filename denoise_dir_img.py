import numpy as np
from astropy.io import fits

from tqdm import tqdm
import socket
import subprocess
import os
import sys

import matplotlib.pyplot as plt

# Get the dirs correct
if 'plffsn2' in socket.gethostname():
    extdir = '/astro/ffsn/Joshi/'
    modeldir = extdir + 'bc03_output_dir/'
    roman_direct_dir = extdir + 'roman_direct_sims/sims2021/'
    roman_slitless_dir = extdir + "GitHub/roman-slitless/"
else:
    extdir = '/Volumes/Joshi_external_HDD/Roman/'
    modeldir = extdir + 'bc03_output_dir/m62/'
    roman_direct_dir = extdir + 'roman_direct_sims/sims2021/'
    home = os.getenv("HOME")
    roman_slitless_dir = home + "/Documents/GitHub/roman-slitless/"

assert os.path.isdir(modeldir)
assert os.path.isdir(roman_direct_dir)

dir_img_part = 'part1'
img_sim_dir = roman_direct_dir + 'K_5degimages_' + dir_img_part + '/'

# This class came from stackoverflow
# SEE: https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-python
class bcolors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def main():

    img_basename = '5deg_'
    img_filt = 'Y106_'
    checkfig = True

    cat_header = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'ALPHA_J2000', 'DELTA_J2000', 
    'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_AUTO', 'MAGERR_AUTO', 'FLUX_RADIUS', 'FWHM_IMAGE']

    # Arrays to loop over
    pointings = np.arange(0, 1)
    detectors = np.arange(1, 4, 1)

    # looop over all SN added images
    for pt in tqdm(pointings, desc="Pointing"):
        for det in tqdm(detectors, desc="Detector", leave=False):

            img_suffix = img_filt + str(pt) + '_' + str(det)
            dir_img_name = img_sim_dir + img_basename + img_suffix + '_SNadded.fits'
            hdu = fits.open(dir_img_name)

            # Now run SExtractor automatically
            # Set up sextractor
            cat_filename = img_basename + img_filt + \
                           str(pt) + '_' + str(det) + '_SNadded.cat'
            img_filename = img_basename + img_filt + \
                           str(pt) + '_' + str(det) + '_SNadded.fits'
            checkimage = img_basename + img_filt + \
                           str(pt) + '_' + str(det) + '_segmap.fits'

            # Change directory to images directory
            os.chdir(img_sim_dir)

            tqdm.write(f"{bcolors.GREEN}" + "Running: " + "sex " + \
                img_filename + " -c" + " roman_sims_sextractor_config.txt" + \
                " -CATALOG_NAME " + cat_filename + \
                " -CHECKIMAGE_NAME " + checkimage + f"{bcolors.ENDC}")

            # Use subprocess to call sextractor.
            # The args passed MUST be passed in this way.
            # i.e., args that would be separated by a space on the 
            # command line must be passed here separated by commas.
            # It will not work if you join all of these args in a 
            # string with spaces where they are supposed to be; 
            # even if the command looks right when printed out.
            sextractor = subprocess.run(['sex', img_filename, 
                '-c', 'roman_sims_sextractor_config.txt', 
                '-CATALOG_NAME', cat_filename, 
                '-CHECKIMAGE_NAME', checkimage], check=True)

            # Go back to roman-slitless directory
            os.chdir(roman_slitless_dir)

            # ---------
            # Now set all pix not associated with objects detected by SExtractor to zero
            segmap_hdu = fits.open(img_sim_dir + checkimage)
            segmap = segmap_hdu[0].data

            # Copy direct image
            dir_img_copy = hdu[0].data

            for x in range(4088):
                for y in range(4088):
                    if segmap[x, y] == 0:
                        dir_img_copy[x, y] = 0.0

            # --------- Save and check with ds9
            denoised_img_hdu = fits.PrimaryHDU(header=hdu[0].header, data=dir_img_copy)
            savefile = dir_img_name.replace('.fits','_nonoise.fits')
            denoised_img_hdu.writeto(savefile, overwrite=True)
            tqdm.write('Saved: ' + savefile)

            # --------- Close all open fits files
            hdu.close()
            segmap_hdu.close()

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)