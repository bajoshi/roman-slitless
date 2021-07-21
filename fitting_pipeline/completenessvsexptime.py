import numpy as np

import matplotlib.pyplot as plt

import os
import sys

def main():

    # Read in results file
    ext_spectra_dir = "/Volumes/Joshi_external_HDD/Roman/roman_slitless_sims_results/"
    results_dir = ext_spectra_dir + 'fitting_results/'
    
    resfile = results_dir + 'zrecovery_pylinear_sims_pt0.txt'
    cat = np.genfromtxt(resfile, dtype=None, names=True, encoding='ascii')

    # Create arrays for plotting
    deltamag = 0.5
    mag_bins = np.arange(17.5, 26.5, deltamag)  # left edges of mag bins
    mags = [(mag_bins[m] + mag_bins[m+1])/2 for m in range(len(mag_bins) - 1)]

    z_tol1 = 0.01  # abs val of delta-z/(1+z)
    z_tol2 = 0.001

    # Do this for each exposure time separately
    exptime_labels = ['z3600', 'z1800', 'z900']
    colors = ['crimson', 'dodgerblue', 'goldenrod']

    # Setup figure
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for e in range(3):

        et = exptime_labels[e]

        total_counts = np.zeros(len(mag_bins) - 1)
        ztol_counts1 = np.zeros(len(mag_bins) - 1)
        ztol_counts2 = np.zeros(len(mag_bins) - 1)

        for i in range(len(cat)):

            temp_z_true = cat['z_true'][i]
            temp_z = cat[et][i]

            mag = cat['Y106mag'][i]

            z_acc = np.abs(temp_z - temp_z_true) / (1 + temp_z_true)

            mag_idx = int((mag - 17.5) / deltamag)

            total_counts[mag_idx] += 1

            if z_acc <= z_tol1:
                ztol_counts1[mag_idx] += 1

            if z_acc <= z_tol2:
                ztol_counts2[mag_idx] += 1

        percent_complete1 = ztol_counts1 / total_counts
        percent_complete2 = ztol_counts2 / total_counts

        ax.plot(mags, percent_complete1, 'o-',  markersize=5, lw=2.0, color=colors[e])
        ax.plot(mags, percent_complete2, 'o--', markersize=5, lw=2.0, color=colors[e])

        # Cumulative completeness fraction
        # ONLY SHOWN FOR LONGEST EXPTIME
        if '3600' in et:
            ts = np.cumsum(total_counts)
            zs1 = np.cumsum(ztol_counts1)
            zs2 = np.cumsum(ztol_counts2)

            pc1 = zs1 / ts
            pc2 = zs2 / ts

            ax.plot(mags, pc1, '-', color='k')
            ax.plot(mags, pc2, '--', color='k')

    # Show the mag dist as a light histogram
    ax1 = ax.twinx()
    ax1.hist(cat['Y106mag'], bins=mag_bins, color='gray', alpha=0.3)

    # labels
    ax.set_ylabel(r'$\mathrm{Frac}.\ z\ \mathrm{completeness}$', fontsize=14)
    ax.set_xlabel(r'$m_{Y106}$', fontsize=14)
    ax1.set_ylabel(r'$\#\ \mathrm{objects}$', fontsize=14)

    plt.show()

    sys.exit(0)

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)