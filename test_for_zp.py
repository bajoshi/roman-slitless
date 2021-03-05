import numpy as np
from astropy.io import fits

import os
import sys

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

def create_regions_from_truth(truth_ra, truth_dec, truth_file):

    reg_file = truth_file.replace('.fits', '.reg')
    with open(reg_file, 'w') as fh:

        fh.write("# Region file format: DS9 version 4.1" + "\n")
        fh.write("global color=green dashlist=8 3 width=1 font=\"helvetica 10 normal roman\" ")
        fh.write("select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1" + "\n")
        fh.write("fk5" + "\n")

        for i in range(len(truth_ra)):
            fh.write("circle(" + "{:.7f}".format(truth_ra[i]) + "," + "{:.7f}".format(truth_dec[i])\
            + "," + "0.0002778" + "\") # color=green width=2" + "\n")  # size is 1 arcsec written in degrees
    
    print("Written:", reg_file)

    return None

def main():

    home = os.getenv('HOME')
    
    img_sim_dir = home + '/Documents/roman_direct_sims/sims2021/sextractor_mag_zp_test/'
    img_basename = 'test_5deg_'
    img_suffix = 'Y106_1_7'
    
    truth_dir = home + '/Documents/roman_direct_sims/sims2021/K_5degtruth/'
    truth_basename = '5deg_index_'
    
    test_obj_ids = np.arange(100)
    print("Chosen the following object ids (seg map ids):", test_obj_ids)
    print("Will check their SExtractor derived mags against truth mags.\n")
    
    # Read in SExtractor catalog
    cat_header = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'ALPHA_J2000', 'DELTA_J2000', \
    'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_AUTO', 'MAGERR_AUTO', 'FLUX_RADIUS', 'FWHM_IMAGE']
    cat = np.genfromtxt(img_sim_dir + img_basename + img_suffix + '.cat', \
        dtype=None, names=cat_header, encoding='ascii')
    
    # Read in truth file
    truth_file = truth_dir + truth_basename + img_suffix + '.fits'
    truth_hdu = fits.open(truth_file)
    
    # assign arrays
    ra  = truth_hdu[1].data['ra']  * 180/np.pi
    dec = truth_hdu[1].data['dec'] * 180/np.pi
    
    create_regions_from_truth(ra, dec, truth_file)
    sys.exit(0)
    
    # Matching tolerance
    tol = 0.3/3600  # arcseconds expressed in degrees since our ra-decs are in degrees
    
    # Now match each object
    for i in range(len(test_obj_ids)):
    
        current_id = int(test_obj_ids[i])
    
        # The -1 in the index is needed because the SExtractor 
        # catalog starts counting object ids from 1.
        ra_to_check = cat['ALPHA_J2000'][current_id - 1]
        dec_to_check = cat['DELTA_J2000'][current_id - 1]
    
        print("\nChecking coords:", ra_to_check, dec_to_check)
    
        # Now match and check if the magnitude is correct
        radiff = ra - ra_to_check
        decdiff = dec - dec_to_check
        idx = np.where( (np.abs(radiff) < tol) & (np.abs(decdiff) < tol) )[0]
    
        print("For object:", current_id)
        print("Matching indices:", idx)
    
        # The line I'd used to match for the SPZ paper
        # Seems to be equivalent to the line above. 
        # The line above needs more testing though. 
        # I know for sure that this line below works.
        #idx = np.where((ra >= ra_to_check - tol) & (ra <= ra_to_check + tol) & \
        #        (dec >= dec_to_check - tol) & (dec <= dec_to_check + tol))[0]

        if len(idx) != 1:
            print(f"{bcolors.WARNING}", "Match not found. Moving to next object.", f"{bcolors.ENDC}")
            continue

        idx = int(idx)
    
        print("Truth X:  ", "{:.4f}".format(truth_hdu[1].data['x'][idx]), "         ", "SExtractor X:  ", cat['X_IMAGE'][current_id - 1])
        print("Truth Y:  ", "{:.4f}".format(truth_hdu[1].data['y'][idx]), "         ", "SExtractor Y:  ", cat['Y_IMAGE'][current_id - 1])
        print("Truth RA: ", "{:.7f}".format(truth_hdu[1].data['ra'][idx] * 180/np.pi), "        ", "SExtractor RA: ", ra_to_check)
        print("Truth DEC:", "{:.7f}".format(truth_hdu[1].data['dec'][idx] * 180/np.pi), "       ", "SExtractor DEC:", dec_to_check)
        print("Truth MAG:", "{:.2f}".format(truth_hdu[1].data['mag'][idx]), "             ", "SExtractor MAG:", cat['MAG_AUTO'][current_id - 1])

        # Get the difference in magnitudes
        true_mag = truth_hdu[1].data['mag'][idx]
        # mag_diff = true_mag - cat['MAG_AUTO'][current_id - 1]

        flux = cat['FLUX_AUTO'][current_id - 1]
        print("FLUX (counts) from SExtractor:", flux)
        print("-2.5log(flux) = ", -2.5 * np.log10(flux))

        zp = true_mag + 2.5 * np.log10(flux)
        print(f"{bcolors.GREEN}", "ZP for this object:", zp, f"{bcolors.ENDC}")

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)













