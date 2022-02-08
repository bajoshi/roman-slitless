import numpy as np
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

import os
import sys

home = os.getenv('HOME')
roman_slitless = home + '/Documents/GitHub/roman-slitless/'

sys.path.append(roman_slitless)
from gen_sed_lst import get_sn_z  # noqa: E402


def get_z_for_mag(m):
    # assuming the arg m passed here is an array
    zarr = np.zeros(len(m))

    for i in range(len(m)):
        mag = m[i]
        zarr[i] = get_sn_z(mag)

    return zarr


# Sigmoid func fitting from Lou
def sigmoid(x, *p):
    x0, T, k = p
    # b = np.sqrt(b**2)
    # k = np.sqrt(k**2)
    b = 0.0
    T = np.sqrt(T**2)
    y = (T / (1 + np.exp(-k * (x - x0)))) - b
    return y


def main():

    # Read in results file
    extdir = "/Volumes/Joshi_external_HDD/Roman/"
    ext_spectra_dir = extdir + "roman_slitless_sims_results/"
    results_dir = ext_spectra_dir + 'fitting_results/'

    resfile = results_dir + 'zrecovery_pylinear_sims_pt0.txt'
    cat = np.genfromtxt(resfile, dtype=None, names=True, encoding='ascii')

    # ---------------------------- Completeness plot
    # Create arrays for plotting
    deltamag = 0.2
    low_maglim = 20.5
    high_maglim = 29.5

    classification_eff = 1.0

    # left edges of mag bins
    mag_bins = np.arange(low_maglim, high_maglim, deltamag)
    mags = []
    for m in range(len(mag_bins) - 1):
        mags.append((mag_bins[m] + mag_bins[m + 1]) / 2)
    # Need to convert to numpy array for where stmt in sigmoid fit
    mags = np.array(mags)

    print("Magnitude range for the SNe:",
          '{:.2f}'.format(np.min(cat['Y106mag'])),
          '{:.2f}'.format(np.max(cat['Y106mag'])))
    print('Total mag bins:', len(mags))

    z_tol = 0.01  # abs val of delta-z/(1+z)

    # Do this for each exposure time separately
    exptime_labels = ['z400']  # ['z10800', 'z3600', 'z1200', 'z400']
    colors = ['goldenrod']  # ['crimson', 'dodgerblue', 'seagreen', 'goldenrod']
    sigmoid_cols = ['peru']  # ['deeppink', 'navy', 'green', 'peru']

    # The above labels are col names in the catalog
    # and these labels below will be used in the plot
    all_exptimes = ['9h', '3h', '1h', '20m']

    # Setup figure
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    for e in range(len(exptime_labels)):

        et = exptime_labels[e]

        total_counts = np.zeros(len(mag_bins) - 1)
        ztol_counts = np.zeros(len(mag_bins) - 1)

        for i in range(len(cat)):

            temp_z_true = cat['z_true'][i]
            temp_z = cat[et][i]

            mag = cat['Y106mag'][i]

            z_acc = np.abs(temp_z - temp_z_true) / (1 + temp_z_true)

            mag_idx = int((mag - low_maglim) / deltamag)

            total_counts[mag_idx] += 1

            if z_acc <= z_tol:
                ztol_counts[mag_idx] += 1

            # print(i, et, z_acc, temp_z, temp_z_true,
            #       list(cat[i])[:10])

        # Now get effective completeness/exptime and plot
        percent_complete = ztol_counts / total_counts
        effective_completeness = percent_complete * classification_eff

        exptime = all_exptimes[e]
        ax.plot(mags, effective_completeness, 'o--', markersize=5,
                color=colors[e], label=r'$t_\mathrm{exp}\, =\ $' + exptime,
                zorder=2)

        # ----------- Fit sigmoid curves and plot
        # Remove NaNs
        completeness_valid_idx = np.where(~np.isnan(effective_completeness))[0]

        mags_tofit = mags[completeness_valid_idx]
        effective_completeness = effective_completeness[completeness_valid_idx]
        # init guess
        p0 = [25., 1.0, -0.4]

        popt, pcov = curve_fit(sigmoid, mags_tofit,
                               effective_completeness, p0=p0)
        perr = np.sqrt(np.diag(np.array(pcov)))

        ax.plot(mags_tofit, sigmoid(mags_tofit, *popt), lw=2.0,
                color=sigmoid_cols[e],
                label=r'$m_c=%.2f\pm%.2f$' % (popt[0], perr[0]),
                zorder=1)

        """
        Cumulative completeness fraction
        ONLY SHOWN FOR LONGEST EXPTIME
        if '10800' in et:
            ts = np.cumsum(total_counts)
            zs1 = np.cumsum(ztol_counts1)
            zs2 = np.cumsum(ztol_counts2)

            pc1 = zs1 / ts
            pc2 = zs2 / ts

            ax.plot(mags, pc1, '-', color='k',
                    label=r'$\frac{\Delta z}{1+z} \leq 0.01$')  # noqa
            ax.plot(mags, pc2, '--', color='k',
                    label=r'$\frac{\Delta z}{1+z} \leq 0.001$')  # noqa
        """

    plt.show()
    sys.exit(0)

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
    mags_for_z_axis_transform = (mt - 18.5) / 13.0
    # the denominator here corresponds to the difference
    # on the bottom x-axis that is shown in the figure
    # NOT the difference between final and init values in mags_for_z_axis
    redshift_ticks = get_z_for_mag(mt)

    ax2.set_xticks(mags_for_z_axis_transform)
    ax2.set_xticklabels(['{:.2f}'.format(z) for z in redshift_ticks],
                        rotation=30)
    ax2.set_xlabel(r'$\mathrm{Redshift\ (assumed\ SN\ at\ peak)}$',
                   fontsize=14)
    ax2.minorticks_off()

    print('Magnitudes:', mt)
    print('Redshifts at above magnitudes:', redshift_ticks)
    print('Total sample size:', len(cat))

    # ax.text(x=low_maglim - 0.2, y=0.28,
    #         s=r'$\mbox{---}\ \frac{\Delta z}{1+z} \leq 0.01$',
    #         verticalalignment='top', horizontalalignment='left',
    #         transform=ax.transData, color='k', size=14)

    # labels
    ax.set_ylabel(r'$\mathrm{Frac}.\ z\ \mathrm{completeness}$', fontsize=14)
    ax.set_xlabel(r'$m_{F106}$', fontsize=14)
    ax1.set_ylabel(r'$\#\ \mathrm{objects}$', fontsize=14)

    # Limits
    ax.set_xlim(low_maglim - 0.5, high_maglim + 0.5)
    ax.set_ylim(-0.04, 1.04)

    ax.legend(loc=0, fontsize=13, frameon=False)

    # save
    fig.savefig(roman_slitless + 'figures/pylinearrecovery_completeness.pdf',
                dpi=200, bbox_inches='tight')
    # Also save in paper figures directory
    clouddir = '/Users/baj/Library/Mobile Documents/com~apple~CloudDocs/'
    cloudfigdir = clouddir + 'Papers/my_papers/Roman_slitless_sims/figures/'
    fig.savefig(cloudfigdir + 'pylinearrecovery_completeness.pdf',
                dpi=200, bbox_inches='tight')

    return None


if __name__ == '__main__':
    main()
    sys.exit(0)
