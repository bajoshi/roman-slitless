import numpy as np
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

import os
import sys

home = os.getenv('HOME')
roman_slitless = home + '/Documents/GitHub/roman-slitless/'

sys.path.append(roman_slitless)
from gen_sed_lst import get_sn_z

def get_z_for_mag(m):
    # assuming the arg m passed here is an array
    zarr = np.zeros(len(m))

    for i in range(len(m)):
        mag = m[i]
        zarr[i] = get_sn_z(mag)

    return zarr

def sigmoid(x, T, mc, s):

    sigmoid_fn = T / (1 - np.exp((x - mc)/s))

    return sigmoid_fn

def main():

    # Read in results file
    ext_spectra_dir = "/Volumes/Joshi_external_HDD/Roman/roman_slitless_sims_results/"
    results_dir = ext_spectra_dir + 'fitting_results/'
    
    resfile = results_dir + 'zrecovery_pylinear_sims_pt0.txt'
    cat = np.genfromtxt(resfile, dtype=None, names=True, encoding='ascii')

    # ---------------------------- Completeness plot
    # Create arrays for plotting
    deltamag = 0.25
    low_maglim = 19.0
    high_maglim = 30.0

    mag_bins = np.arange(low_maglim, high_maglim, deltamag)  # left edges of mag bins
    mags = [(mag_bins[m] + mag_bins[m+1])/2 for m in range(len(mag_bins) - 1)]

    z_tol1 = 0.01  # abs val of delta-z/(1+z)
    z_tol2 = 0.001

    # Do this for each exposure time separately
    exptime_labels = ['z10800', 'z3600', 'z1200', 'z400']
    colors = ['crimson', 'dodgerblue', 'seagreen', 'goldenrod']
    ls = 'o-'

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

            #print(i, mag_idx, mag)

            total_counts[mag_idx] += 1

            if z_acc <= z_tol1:
                ztol_counts1[mag_idx] += 1

            if z_acc <= z_tol2:
                ztol_counts2[mag_idx] += 1

        percent_complete1 = ztol_counts1 / total_counts
        percent_complete2 = ztol_counts2 / total_counts

        ax.plot(mags, percent_complete1, 'o-',  markersize=3, color=colors[e])
        #ax.plot(mags, percent_complete2, 'x', markersize=5, color=colors[e])

        # ----------- Fit sigmoid curves and plot
        # fix nan to zero
        percent_complete1[np.isnan(percent_complete1)] = 0.0
        # init guess
        p0 = [1.0, 25.0, 0.05]  # in order: T, mc, s
        popt, pcov = curve_fit(sigmoid, mags, percent_complete1, p0)

        print('----'*3)
        print(popt)

        #ax.plot(mags, sigmoid(mags, popt[0], popt[1], popt[2]), lw=2.0, color=colors[e])

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

    # ----------- Top redshift axis
    # Don't need to plot anything
    # just set the correct redshift 
    # corresponding to bottom x-axis mag
    ax2 = ax.twiny()
    # --- Set mags and get corresponding redshift    
    # Since we're not plotting anything the default ticks
    # are at [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # SO the on the bottom x-axis need to be transformed to [0,1]
    # i.e., the range [20.5, 28.0] --> [0,1]
    mt = np.arange(18.5, 31.5, 1.0)
    mags_for_z_axis_transform = (mt - 18.5)/13.0
    # the denominator here corresponds to the difference 
    # on the bottom x-axis that is shown in the figure
    # NOT the difference between final and init values in mags_for_z_axis
    redshift_ticks = get_z_for_mag(mt)

    ax2.set_xticks(mags_for_z_axis_transform)
    ax2.set_xticklabels(['{:.2f}'.format(z) for z in redshift_ticks], rotation=30)
    ax2.set_xlabel(r'$\mathrm{Redshift\ (assumed\ SN\ at\ peak)}$', fontsize=14)
    ax2.minorticks_off()

    print('Magnitudes:', mt)
    print('Redshifts at above magnitudes:', redshift_ticks)
    #print('z at mag 23 and 23.5:', get_z_for_mag([23.0]), get_z_for_mag([23.5]))

    print('Total sample size:', len(cat))

    # Text info
    #ax.text(x=22.0, y=0.37, 
    #    s=r'$18000\ \mathrm{seconds;}$' + '\n' + \
    #    r'$\mathrm{\left<SNR\right>}^{z\sim1}_{F106\sim23.25}=$' + '{:.1f}'.format(snr_mag23_3600), 
    #    verticalalignment='top', horizontalalignment='left', 
    #    transform=ax.transData, color='crimson', size=14)
    #ax.text(x=22.0, y=0.23, 
    #    s=r'$4500\ \mathrm{seconds;}$' + '\n' + \
    #    r'$\mathrm{\left<SNR\right>}^{z\sim1}_{F106\sim23.25}=$' + '{:.1f}'.format(snr_mag23_900), 
    #    verticalalignment='top', horizontalalignment='left', 
    #    transform=ax.transData, color='dodgerblue', size=14)

    ax.text(x=low_maglim-0.2, y=0.28, s=r'$\mbox{---}\ \frac{\Delta z}{1+z} \leq 0.01$', 
        verticalalignment='top', horizontalalignment='left', 
        transform=ax.transData, color='k', size=14)
    #ax.text(x=low_maglim, y=0.1, s=r'$-- \frac{\Delta z}{1+z} \leq 0.001$',
    #    verticalalignment='top', horizontalalignment='left', 
    #    transform=ax.transData, color='k', size=14)

    # labels
    ax.set_ylabel(r'$\mathrm{Frac}.\ z\ \mathrm{completeness}$', fontsize=14)
    ax.set_xlabel(r'$m_{F106}$', fontsize=14)
    ax1.set_ylabel(r'$\#\ \mathrm{objects}$', fontsize=14)

    # Limits
    ax.set_xlim(low_maglim - 0.5, high_maglim + 0.5)
    ax.set_ylim(-0.04, 1.04)

    # save
    fig.savefig(roman_slitless + 'figures/pylinearrecovery_completeness.pdf', 
                dpi=200, bbox_inches='tight')
    # Also save in paper figures directory
    fig.savefig('/Users/baj/Library/Mobile Documents/com~apple~CloudDocs/Papers/my_papers/Roman_slitless_sims/figures/' \
                + 'pylinearrecovery_completeness.pdf', 
                dpi=200, bbox_inches='tight')

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)