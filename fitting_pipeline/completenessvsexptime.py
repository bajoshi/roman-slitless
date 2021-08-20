import numpy as np

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

def main():

    # Read in results file
    ext_spectra_dir = "/Volumes/Joshi_external_HDD/Roman/roman_slitless_sims_results/"
    results_dir = ext_spectra_dir + 'fitting_results/'
    
    resfile = results_dir + 'zrecovery_pylinear_sims_pt0.txt'
    cat = np.genfromtxt(resfile, dtype=None, names=True, encoding='ascii')

    # ---------------------------- Completeness plot
    # Create arrays for plotting
    deltamag = 0.5
    low_maglim = 18.5
    high_maglim = 26.0

    mag_bins = np.arange(low_maglim, high_maglim, deltamag)  # left edges of mag bins
    mags = [(mag_bins[m] + mag_bins[m+1])/2 for m in range(len(mag_bins) - 1)]

    z_tol1 = 0.01  # abs val of delta-z/(1+z)
    z_tol2 = 0.001

    # Do this for each exposure time separately
    exptime_labels = ['z3600', 'z900']
    colors = ['crimson', 'dodgerblue']

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
        #ax.plot(mags, percent_complete2, 'o--', markersize=5, lw=2.0, color=colors[e])

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

    mag23_bin_idx = np.where((cat['Y106mag'] >= 23.0) & (cat['Y106mag'] < 23.5))[0]
    snr_mag23_3600 = np.mean(cat['SNR3600'][mag23_bin_idx])
    snr_mag23_900 = np.mean(cat['SNR900'][mag23_bin_idx])

    print('\nFor SNe at approx z=1:')
    print('  IMG   SegID   Mag   z-true   z-wide   z-deep   SNR-wide   SNR-deep') 
    for s in range(len(mag23_bin_idx)):
        print(cat['img_suffix'][mag23_bin_idx][s], '  ', 
              cat['SNSegID'][mag23_bin_idx][s], '  ',
              cat['Y106mag'][mag23_bin_idx][s], '  ',
              cat['z_true'][mag23_bin_idx][s], '  ',
              cat['z900'][mag23_bin_idx][s], '  ',
              cat['z3600'][mag23_bin_idx][s], '  ',
              cat['SNR900'][mag23_bin_idx][s], '  ',
              cat['SNR3600'][mag23_bin_idx][s])
    print('\n')

    #mag24_idx = np.where(cat['Y106mag'] >= 24.0)[0]
    #print(mag24_idx)
    #snr_mag24 = np.mean(cat['SNR3600'][mag24_idx])
    #print(snr_mag24)

    # ----------- Top redshift axis
    # Don't need to plot anything
    # just set the correct redshift 
    # corresponding to bottom x-axis mag
    ax2 = ax.twiny()
    # --- Set mags and get corresponding redshift    
    # Since we're not plotting anything the default ticks
    # are at [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # SO the on the bottom x-axis need to be transformed to [0,1]
    # i.e., the range [18.0, 26.5] --> [0,1]
    mt = np.array([18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 26.5])
    mags_for_z_axis_transform = (mt - 18.0)/8.5
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

    print('z at mag 23 and 23.5:', get_z_for_mag([23.0]), get_z_for_mag([23.5]))

    print('Total sample size:', len(cat))

    # Text info
    ax.text(x=23.5, y=0.37, 
        s=r'$18000\ \mathrm{seconds;}$' + '\n' + \
        r'$\mathrm{\left<SNR\right>}^{z\sim1}_{F106\sim23.25}=$' + '{:.1f}'.format(snr_mag23_3600), 
        verticalalignment='top', horizontalalignment='left', 
        transform=ax.transData, color='crimson', size=14)
    ax.text(x=23.5, y=0.23, 
        s=r'$4500\ \mathrm{seconds;}$' + '\n' + \
        r'$\mathrm{\left<SNR\right>}^{z\sim1}_{F106\sim23.25}=$' + '{:.1f}'.format(snr_mag23_900), 
        verticalalignment='top', horizontalalignment='left', 
        transform=ax.transData, color='dodgerblue', size=14)

    ax.text(x=low_maglim, y=0.28, s=r'$\mbox{---}\ \frac{\Delta z}{1+z} \leq 0.01$', 
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

    # save
    fig.savefig(results_dir + 'pylinearrecovery_completeness.pdf', 
                dpi=200, bbox_inches='tight')
    # Also save in paper figures directory
    fig.savefig('/Users/baj/Library/Mobile Documents/com~apple~CloudDocs/Papers/my_papers/romansims_sne/figures/' \
                + 'pylinearrecovery_completeness.pdf', 
                dpi=200, bbox_inches='tight')

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)