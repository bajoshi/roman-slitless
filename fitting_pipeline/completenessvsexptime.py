import numpy as np
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

import os
import sys

import warnings
warnings.filterwarnings(action='ignore')

home = os.getenv('HOME')
roman_slitless = home + '/Documents/GitHub/roman-slitless/'
fitting_utils = roman_slitless + 'fitting_pipeline/utils/'

# Read in lookup table for getting redshifts on the top axis
# Lookup table name and zarr for lookup copied over
# from kcorr.py
lookup_table_fname = fitting_utils + 'sn_mag_z_lookup_comp_plot.txt'
lookup_table = np.genfromtxt(lookup_table_fname,
                             dtype=None, names=True,
                             encoding='ascii')
lookup_mags = lookup_table['mF106']
lookup_z = lookup_table['Redshift']


def get_sn_z_comp_plot(snmag):

    z_idx = np.argmin(abs(snmag - lookup_mags))
    snz = lookup_z[z_idx]

    return snz


def get_z_for_mag(m):

    # Get redshifts depending on whether a single
    # magnitude or an array has been passed.
    if len(m) > 1:
        zarr = np.zeros(len(m))

        for i in range(len(m)):
            mag = m[i]
            zarr[i] = get_sn_z_comp_plot(mag)

        return zarr

    elif len(m) == 1:
        return get_sn_z_comp_plot(mag)


# Sigmoid func fitting from Lou
def sigmoid(x, *p):
    x0, k = p  # x0, T, k = p
    T = 1  # np.sqrt(T**2)
    y = (T / (1 + np.exp(-k * (x - x0))))
    return y


def main():

    # Read in results file
    extdir = "/Volumes/Joshi_external_HDD/Roman/"
    ext_spectra_dir = extdir + "roman_slitless_sims_results/"
    results_dir = ext_spectra_dir + 'fitting_results/'

    resfile = results_dir + 'zrecovery_pylinear_sims_pt0.txt'
    cat = np.genfromtxt(resfile, dtype=None, names=True, encoding='ascii')

    # ---------------------------- Prep
    # Create arrays for plotting
    deltamag = 0.2
    low_maglim = 21.5
    high_maglim = 29.5

    # left edges of mag bins
    mag_bins = np.arange(low_maglim, high_maglim, deltamag)
    mags = []
    for m in range(len(mag_bins) - 1):
        mags.append((mag_bins[m] + mag_bins[m + 1]) / 2)
    # Need to convert to numpy array for where stmt in sigmoid fit
    mags = np.array(mags)
    print(mag_bins)
    print(mags)

    print("Magnitude range for the SNe:",
          '{:.2f}'.format(np.min(cat['Y106mag'])),
          '{:.2f}'.format(np.max(cat['Y106mag'])))
    print('Total mag bins:', len(mags))

    # ---------------------------- Plot to ensure that overlapping
    # isn't dependent on SN mag, i.e., SN at all mags have equal
    # likelihood of overlapping with their host.
    # So the plot below will show the histogram of the mag distribution
    # for all sources and an overlaid histogram of only those SN that
    # have an overlap with their host. The overlaid hist should follow
    # the parent hist distribution with just lower #s at all mags.
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    ax1.hist(cat['Y106mag'], bins=mag_bins, color='gray',
             alpha=0.3, label='All SNe')

    # Manually build up list of mags for those SNe that have overlap
    overlap_mags = []
    for k in range(len(cat)):
        if cat['overlap'][k]:
            overlap_mags.append(cat['Y106mag'][k])

    ax1.hist(overlap_mags, bins=mag_bins, color='navy',
             histtype='step', lw=3.0, label='SNe with host overlap')

    ax1.legend(loc=0, fontsize=13)

    # plt.show()
    fig1.clear()
    plt.close(fig1)

    # ---------------------------- Completeness plot
    classification_eff = 1.0

    z_tol = 0.01  # abs val of delta-z/(1+z)

    # ['z10800', 'z3600', 'z1200', 'z400']
    # ['crimson', 'dodgerblue', 'seagreen', 'goldenrod']
    # ['deeppink', 'navy', 'green', 'peru']

    # Do this for each exposure time separately
    exptime_labels = ['z3600', 'z1200', 'z400']

    # Colors from color brewer
    colors = ['#1b9e77', '#d95f02', '#7570b3']
    sigmoid_cols = ['green', 'peru', 'navy']
    sigmoid_err_cols = ['lightgreen', 'burlywood', 'skyblue']

    # The above labels are col names in the catalog
    # and these labels below will be used in the plot
    all_exptimes = ['3h', '1h', '20m']

    # Initial guesses for the sigmoid fitting
    init_guesses = [[26.5, -0.4],
                    [25., -0.4],
                    [24., -0.4]]

    # Setup figure
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    for e in range(len(exptime_labels)):

        print('\nWorking on exposure time:', all_exptimes[e])
        print('-----------------------------\n')

        et = exptime_labels[e]

        total_counts = np.zeros(len(mag_bins) - 1)
        ztol_counts = np.zeros(len(mag_bins) - 1)

        z_acc_list = np.zeros(len(cat))
        z_inferred_list = np.zeros(len(cat))

        for i in range(len(cat)):

            #
            # if cat['overlap'][i]:
            #     continue

            temp_z_true = cat['z_true'][i]
            temp_z = cat[et][i]

            mag = cat['Y106mag'][i]

            z_acc = np.abs(temp_z - temp_z_true) / (1 + temp_z_true)

            mag_idx = int((mag - low_maglim) / deltamag)

            total_counts[mag_idx] += 1

            if z_acc <= z_tol:
                ztol_counts[mag_idx] += 1

                # This passing string is just for debugging
                passing = 'PASSING'  # noqa
            else:
                passing = 'NOT PASSING'  # noqa

            # Append to our lists we need to get stats
            z_acc_list[i] = z_acc
            z_inferred_list[i] = temp_z

            # Printing some debugging info
            # DO NOT DELETE!
            # It took a lot of effort to get the alignment right
            # if cat['overlap'][i]:
            #     overlap_printvar = 'OVERLAP'
            # else:
            #     overlap_printvar = ''

            # if mag > 26.0:
            # print('{:>2d}'.format(i), '  ',
            #       all_exptimes[e], '  ',
            #       cat['SNSegID'][i], '  ',
            #       '{:.2f}'.format(mag), '  ',
            #       '{:6.2f}'.format(cat['SNR3600'][i]), '  ',
            #       '{:>10.4f}'.format(z_acc), '  ',
            #       '{:>10.4f}'.format(temp_z), '  ',
            #       '{:>.4f}'.format(temp_z_true), '  ',
            #       '{:^10}'.format(overlap_printvar), '      ',
            #       passing)

        # Print some stats
        # Remove catastrophic failures and invalid values first
        invalid_idx = np.where(z_inferred_list == -9999.0)[0]
        z_acc_list[invalid_idx] = np.nan
        catas_fail_idx = np.where(z_acc_list > 0.1)[0]
        print(len(catas_fail_idx), 'out of',
              len(z_acc_list), 'are catastrophic failures.',
              'i.e.,', '{:.2f}'.format(len(catas_fail_idx)/len(z_acc_list)),
              'percent.')

        z_acc_list[catas_fail_idx] = np.nan

        print('Mean z-accuracy for this exptime:', np.nanmean(z_acc_list))
        print('Median z-accuracy for this exptime:', np.nanmedian(z_acc_list))

        # Now get effective completeness/exptime and plot
        percent_complete = ztol_counts / total_counts
        effective_completeness = percent_complete * classification_eff

        exptime = all_exptimes[e]
        ax.plot(mags, effective_completeness, 'o--', markersize=5,
                color=colors[e], label=r'$t_\mathrm{exp}\, =\ $' + exptime,
                zorder=2)

        # ----------- Fit sigmoid curves and plot
        # Remove NaNs
        completeness_valid_idx = \
            np.where(~np.isnan(effective_completeness))[0]

        mags_tofit = mags[completeness_valid_idx]
        effective_completeness = \
            effective_completeness[completeness_valid_idx]
        # init guess
        p0 = init_guesses[e]

        popt, pcov = curve_fit(sigmoid, mags_tofit,
                               effective_completeness, p0=p0)
        perr = np.sqrt(np.diag(np.array(pcov)))

        print('Fitted params:', popt)

        # Plot fitted sigmoid
        ax.plot(mags_tofit, sigmoid(mags_tofit, *popt), lw=1.0,
                color=sigmoid_cols[e],
                label=r'$m_c=%.2f\pm%.2f$' % (popt[0], perr[0]),  # noqa
                zorder=1)

        # ------ Plot error on sigmoid curves
        # within error with an alpha level specified.
        xx = np.arange(20.0, 31.0, 0.05)

        ps = np.random.multivariate_normal(popt, pcov, 100)
        ysample = np.asarray([sigmoid(xx, *pi) for pi in ps])

        lower = np.percentile(ysample, 15.9, axis=0)
        upper = np.percentile(ysample, 84.1, axis=0)

        ax.fill_between(xx, upper, lower,
                        color=sigmoid_err_cols[e], alpha=0.5)

        # Below: old code block for plotting error range.
        """
        for s in range(1000):
            params = []

            # Should we only vary the central magnitude?
            # Is that the most robustly measured param?
            mc_arr = np.arange(popt[0] - perr[0],
                               popt[0] + perr[0], 0.01)
            T_arr = np.arange(popt[1] - perr[1],
                              popt[1] + perr[1], 0.01)
            b_arr = np.arange(popt[2] - perr[2],
                              popt[2] + perr[2], 0.01)

            mc = np.random.choice(mc_arr)
            T = np.random.choice(T_arr)  # popt[1]
            b = np.random.choice(b_arr)  # popt[2]

            params.append(mc)
            params.append(T)
            params.append(b)

            ax.plot(mags_tofit, sigmoid(mags_tofit, *params), lw=0.7,
                    color=sigmoid_cols[e], zorder=1, alpha=0.02)
        """

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
    mt = np.arange(low_maglim - 0.5, high_maglim + 0.5, 0.5)
    mag_rng_diff = high_maglim - low_maglim + 1.0
    mags_for_z_axis_transform = (mt - (low_maglim - 0.5)) / mag_rng_diff
    # the denominator here corresponds to the difference
    # on the bottom x-axis that is shown in the figure
    # NOT the difference between final and init values in mags_for_z_axis
    redshift_ticks = get_z_for_mag(mt)

    ax2.set_xticks(mags_for_z_axis_transform)
    ax2.set_xticklabels(['{:.2f}'.format(z) for z in redshift_ticks],
                        rotation=30)
    redshift_axis_label = \
        r'$\mathrm{Approx.\ Redshift\ (assumed\ SN\ at\ peak,\ no\ dust)}$'
    ax2.set_xlabel(redshift_axis_label, fontsize=10)
    ax2.minorticks_off()

    print('\n')
    print('Magnitudes:', mt)
    print('Redshifts at above magnitudes:', redshift_ticks)
    print('Total sample size:', len(cat))

    # ax.text(x=low_maglim - 0.2, y=0.28,
    #         s=r'$\mbox{---}\ \frac{\Delta z}{1+z} \leq 0.01$',
    #         verticalalignment='top', horizontalalignment='left',
    #         transform=ax.transData, color='k', size=14)

    # labels
    ax.set_ylabel(r'$\mathrm{Frac}.\ z\ \mathrm{completeness}$', fontsize=14)
    ax.set_xlabel(r'$\mathrm{SN}\ m_{F106}$', fontsize=14)
    ax1.set_ylabel(r'$\#\ \mathrm{objects}$', fontsize=14)

    # Limits
    ax.set_xlim(low_maglim - 0.5, high_maglim + 0.5)
    ax.set_ylim(-0.04, 1.04)

    ax.legend(loc=0, fontsize=13, frameon=False)

    ax.minorticks_on()
    # Comment grid out for final plot.
    # Turn it on to see which mag corresponds
    # exactly to the redshifts tick shown.
    # Compare by looking at figure and lookup table side by side.
    # ax.grid()

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
