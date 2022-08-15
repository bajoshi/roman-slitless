import numpy as np
from astropy.io import fits

import os
import sys

extdir = '/Volumes/Joshi_external_HDD/Roman/'
roman_direct_dir = extdir + 'roman_direct_sims/sims2021/'

assert os.path.isdir(roman_direct_dir)

dir_img_part = 'part1'
img_sim_dir = roman_direct_dir + 'K_5degimages_' + dir_img_part + '/'


if __name__ == '__main__':

    print('This code will add the inserted object IDs and their')
    print('(x,y) locations to the SExtractor catalog. No other ')
    print('properties are added for the inserted objects.')
    print('=================\n')

    # ---------------
    # Arrays to loop over
    pointings = np.arange(0, 1)
    detectors = np.arange(1, 19, 1)

    for pt in pointings:
        for det in detectors:

            img_suffix = 'Y106_' + str(pt) + '_' + str(det)
            dir_img_name = img_sim_dir + '5deg_' + img_suffix + '.fits'

            model_img_name = dir_img_name.replace('.fits', '_formodel.fits')
            cat_filename = model_img_name.replace('.fits', '.cat')

            # Read in catalog of inserted objects
            insert_cat = np.load(dir_img_name.replace('.fits', '_SNadded.npy'))

            with open(cat_filename, 'a') as fh:

                # Now construct line to append
                for i in range(len(insert_cat)):

                    line_to_append = (insert_cat[i][-1] + '   '
                                      + insert_cat[i][0] + '   '
                                      + insert_cat[i][1])

                    fh.write('      ' + line_to_append + '   '
                             + '-9999.0' + '   '
                             + '-9999.0' + '   '
                             + '-9999.0' + '   '
                             + '-9999.0' + '   '
                             + '-9999.0' + '   '
                             + '-9999.0' + '   '
                             + '-9999.0' + '   '
                             + '-9999.0' + '   '
                             + '-9999.0' + '   '
                             + '\n')

            print('Updated:', cat_filename)

    sys.exit(0)
