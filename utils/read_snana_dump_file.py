from astropy.table import Table
import numpy as np

import os
import sys

import matplotlib.pyplot as plt

home = os.getenv('HOME')


if __name__ == '__main__':

    # Read in large snana file
    snana_cat_path = home + \
        '/Desktop/PIP_WFIRST_STARTERKIT+SETEXP_WFIRST_SIMDATA_G10.DUMP'
    tbl = Table.read(snana_cat_path, format='ascii')

    # Get redshift and mags
    allz = tbl['ZCMB']
    allmag = tbl['MAGT0_Y']

    # Get all IA
    ia_idx = np.where(tbl['GENTYPE'] == 1)[0]

    # Get only valid mags
    mag_idx = np.where(allmag >= 10)[0]

    valid_idx = np.intersect1d(mag_idx, ia_idx)

    # Clip to only include type IA
    allz = allz[valid_idx]
    allmag = allmag[valid_idx]

    # Check stretch and color ranges
    all_color = tbl['S2c'][valid_idx]
    all_x1 = tbl['S2x1'][valid_idx]
    print('Color range:', np.min(all_color), np.max(all_color))
    print('X1 range:', np.min(all_x1), np.max(all_x1))

    # PLot to make sure
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel('Redshift', fontsize=15)
    ax.set_ylabel('Peak F106 mag', fontsize=15)

    # First axis to show KDE and actual data 
    ax.scatter(allz, allmag, color='k', alpha=0.15, s=0.2)

    plt.show()

    sys.exit(0)
