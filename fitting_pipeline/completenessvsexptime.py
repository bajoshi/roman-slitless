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
    low_maglim = 18.5
    high_maglim = 26.0

    mag_bins = np.arange(low_maglim, high_maglim, deltamag)  # left edges of mag bins
    mags = [(mag_bins[m] + mag_bins[m+1])/2 for m in range(len(mag_bins) - 1)]

    z_tol1 = 0.01  # abs val of delta-z/(1+z)
    z_tol2 = 0.001

    # Do this for each exposure time separately
    exptime_labels = ['z3600', 'z900']  # ['z3600', 'z1800', 'z900']
    colors = ['crimson', 'dodgerblue']  #, 'goldenrod']

    # Setup figure
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111)

    for e in range(len(exptime_labels)):

        et = exptime_labels[e]

        total_counts = np.zeros(len(mag_bins) - 1)
        ztol_counts1 = np.zeros(len(mag_bins) - 1)
        ztol_counts2 = np.zeros(len(mag_bins) - 1)

        for i in range(len(cat)):

            temp_z_true = cat['z_true'][i]
            temp_z = cat[et][i]

            mag = cat['Y106mag'][i]

            z_acc = np.abs(temp_z - temp_z_true) / (1 + temp_z_true)

            mag_idx = int((mag - low_maglim) / deltamag)

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
        #if '900' in et:
        #    ts = np.cumsum(total_counts)
        #    zs1 = np.cumsum(ztol_counts1)
        #    zs2 = np.cumsum(ztol_counts2)

        #    pc1 = zs1 / ts
        #    pc2 = zs2 / ts

        #    ax.plot(mags, pc1, '-', color='k', label=r'$\frac{\Delta z}{1+z} \leq 0.01$')
        #    ax.plot(mags, pc2, '--', color='k', label=r'$\frac{\Delta z}{1+z} \leq 0.001$')

    # Show the mag dist as a light histogram
    ax1 = ax.twinx()
    ax1.hist(cat['Y106mag'], bins=mag_bins, color='gray', alpha=0.3)

    #ax.legend(loc=6, frameon=False, fontsize=14)

    mag22_bin_idx = np.where((cat['Y106mag'] >= 22.0) & (cat['Y106mag'] < 22.5))[0]
    snr_mag22_3600 = np.mean(cat['SNR3600'][mag22_bin_idx])
    #snr_mag22_1800 = np.mean(cat['SNR1800'][mag22_bin_idx])
    snr_mag22_900 = np.mean(cat['SNR900'][mag22_bin_idx])

    # Text info
    ax.text(x=22.0, y=0.93, 
        s=r'$18000\ \mathrm{seconds; \left<SNR\right>}_{F106\sim22.25}=$' + '{:.1f}'.format(snr_mag22_3600), 
        verticalalignment='top', horizontalalignment='left', 
        transform=ax.transData, color='crimson', size=14)
    #ax.text(x=22.0, y=0.65, 
    #    s=r'$1800\ \mathrm{seconds; \left<SNR\right>}_{F106\sim22.25}=$' + '{:.1f}'.format(snr_mag22_1800), 
    #    verticalalignment='top', horizontalalignment='left', 
    #    transform=ax.transData, color='dodgerblue', size=14)
    ax.text(x=22.0, y=0.87, 
        s=r'$4500\ \mathrm{seconds; \left<SNR\right>}_{F106\sim22.25}=$' + '{:.1f}'.format(snr_mag22_900), 
        verticalalignment='top', horizontalalignment='left', 
        transform=ax.transData, color='dodgerblue', size=14)

    ax.text(x=low_maglim, y=0.2, s=r'$\mbox{---}\ \frac{\Delta z}{1+z} \leq 0.01$', 
        verticalalignment='top', horizontalalignment='left', 
        transform=ax.transData, color='k', size=14)
    ax.text(x=low_maglim, y=0.1, s=r'$-- \frac{\Delta z}{1+z} \leq 0.001$',
        verticalalignment='top', horizontalalignment='left', 
        transform=ax.transData, color='k', size=14)

    # labels
    ax.set_ylabel(r'$\mathrm{Frac}.\ z\ \mathrm{completeness}$', fontsize=14)
    ax.set_xlabel(r'$m_{F106}$', fontsize=14)
    ax1.set_ylabel(r'$\#\ \mathrm{objects}$', fontsize=14)

    # Limits
    ax.set_xlim(low_maglim - 0.5, high_maglim + 0.5)

    # save
    fig.savefig(results_dir + 'pylinearrecovery_completeness.pdf', 
        dpi=200, bbox_inches='tight')

    plt.show()

    sys.exit(0)

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)