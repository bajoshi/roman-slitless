import numpy as np
from astropy.io import fits

import os
import sys

import matplotlib.pyplot as plt

extdir = '/Volumes/Joshi_external_HDD/Roman/'
roman_direct_dir = extdir + 'roman_direct_sims/sims2021/'
assert os.path.isdir(roman_direct_dir)

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

    home = os.getenv('HOME')
    
    img_sim_dir = roman_direct_dir + 'sextractor_mag_zp_test/'
    img_basename = '5deg_'
    
    truth_dir = roman_direct_dir + 'K_5degtruth/'
    truth_basename = '5deg_index_'
    
    # Matching tolerance
    tol = 0.3/3600  # arcseconds expressed in degrees since our ra-decs are in degrees

    # List of images to test
    img_suffix_list = ['Y106_0_1', 'Y106_0_5', 'Y106_0_9', 'Y106_0_12', 'Y106_0_13']

    # Set up figure
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)

    ax.set_xlabel(r'$\Delta m, m_\mathrm{true} - m_\mathrm{SExtractor}$', fontsize=15)
    ax.set_ylabel('\# objects', fontsize=15)

    # Array to hold all of the mag diff
    all_mag_diff = []

    # Now loop over all images
    for img_suffix in img_suffix_list:

        # Read in SExtractor catalog
        cat_header = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'ALPHA_J2000', 'DELTA_J2000', \
        'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_AUTO', 'MAGERR_AUTO', 'FLUX_RADIUS', 'FWHM_IMAGE']
        cat = np.genfromtxt(img_sim_dir + img_basename + img_suffix + '_cps.cat', \
            dtype=None, names=cat_header, encoding='ascii')
    
        test_obj_ids = np.arange(len(cat))
        print("Chosen the following object ids (seg map ids):", test_obj_ids)
        print("Will check their SExtractor derived mags against truth mags.\n")

        # Read in truth file
        truth_file = truth_dir + truth_basename + img_suffix + '.fits'
        truth_hdu = fits.open(truth_file)
    
        # assign arrays
        ra  = truth_hdu[1].data['ra']  * 180/np.pi
        dec = truth_hdu[1].data['dec'] * 180/np.pi
    
        #create_regions_from_truth(ra, dec, truth_file)
        #sys.exit(0)
    
        # EMpty list ot hold zeropoints
        zp_list = []
        mag_diff = []
    
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
            mag_diff.append(true_mag - cat['MAG_AUTO'][current_id - 1])

            flux = cat['FLUX_AUTO'][current_id - 1]
            print("FLUX (counts) from SExtractor:", flux)
            print("-2.5log(flux) = ", -2.5 * np.log10(flux))

            zp = true_mag + 2.5 * np.log10(flux)
            print(f"{bcolors.GREEN}", "ZP for this object:", zp, f"{bcolors.ENDC}")
            print(f"{bcolors.WARNING}", "Be careful! The quoted ZP combines a SExtractor")
            print("quantity with a true quantity. Mag diff is a better statistic.(?)", f"{bcolors.ENDC}")

            zp_list.append(zp)

        # Make a histogram of the zeropoints for all objects in a given image
        zp = np.asarray(zp_list)

        """
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.set_xlabel('ZP', fontsize=15)
        ax.set_ylabel('\# objects', fontsize=15)

        ax.hist(zp, 50, range=(25.0, 27.5), histtype='step', color='k', lw=2.5)

        #plt.show()
        plt.clf()
        plt.cla()
        plt.close()
        """
        
        # Append to list first
        # For later when we need mean and median
        all_mag_diff.append(mag_diff)

        # Histogram of magnitude differences
        mag_diff = np.asarray(mag_diff)
        ax.hist(mag_diff, 40, histtype='step', lw=2.5, label=img_suffix.replace('_','-'), range=(-2.0, 2.0))

    ax.legend(fontsize=14)

    # Compute mean and median mag diff
    all_mag_diff = np.asarray(all_mag_diff, dtype=object)
    all_mag_diff = all_mag_diff.flatten()

    magdiff_mean = np.mean(np.mean(all_mag_diff))  # don't kknow why this has to be done twice. don't have time to figure it out.
    magdiff_median = np.median(np.median(all_mag_diff))  # don't kknow why this has to be done twice. don't have time to figure it out.

    print('\nMean mag diff:', magdiff_mean)
    print('\nMedian mag diff:', magdiff_median)

    fig.savefig('figures/mag_diff_hist_imagesims.pdf', dpi=300, bbox_inches='tight')

    plt.show()

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)













