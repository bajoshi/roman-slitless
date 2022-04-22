import numpy as np
from scipy import stats
from astropy.table import Table
import sys
import os

import matplotlib.pyplot as plt

home = os.getenv('HOME')


if __name__ == '__main__':

    # snana_cat = get_snana_cat()
    # For now working with the small file from Phil
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

    # KDE for the distribution of points
    zmin = np.min(allz)
    zmax = np.max(allz)
    magmin = np.min(allmag)
    magmax = np.max(allmag)

    print('Redshift range:', zmin, zmax)
    print('m_F106 range:', magmin, magmax)

    X, Y = np.mgrid[zmin:zmax:100j, magmin:magmax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])

    values = np.vstack([allz, allmag])

    kernel = stats.gaussian_kde(values)

    # Now resample from kernel
    num_sne = 150 * 18
    print('Total', num_sne, 'resampled SNe mag and z.')
    samples = kernel.resample(size=num_sne)

    # Save out resampled mag and z as a txt file
    # This will be read in to insert SNe into sims.
    with open('sim_sn_mag_vs_z.txt', 'w') as fh:
        fh.write('# Redshift peak_mF106' + '\n')
        for i in range(num_sne):
            fh.write('{:.4f}'.format(samples[0][i]) + ' ')
            fh.write('{:.4f}'.format(samples[1][i]))
            fh.write('\n')

    # ----------------
    # PLot to make sure
    fig, (ax1, ax2) = plt.subplots(figsize=(12, 6), nrows=1, ncols=2)

    ax1.set_xlabel('Redshift', fontsize=15)
    ax1.set_ylabel('Peak F106 mag', fontsize=15)

    # First axis to show KDE and actual data 
    ax1.scatter(allz, allmag, color='k', alpha=0.08, s=0.2)

    Z = np.reshape(kernel(positions).T, X.shape)
    ax1.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
               extent=[zmin, zmax, magmin, magmax])

    ax1.set_aspect((zmax - zmin)/(magmax - magmin))

    # Second axis to show sampling from KDE
    ax2.set_xlabel('resampled Redshift', fontsize=15)
    ax2.set_ylabel('resampled Peak F106 mag', fontsize=15)

    ax2.scatter(samples[0], samples[1], color='k', alpha=0.4, s=1)

    # Force same limits and aspect ratio on both axes
    ax1.set_xlim(0.0, 3.0)
    ax1.set_ylim(18.0, 32.0)

    ax2.set_xlim(0.0, 3.0)
    ax2.set_ylim(18.0, 32.0)

    ax2.set_aspect((zmax - zmin)/(magmax - magmin))

    plt.show()

    sys.exit(0)
