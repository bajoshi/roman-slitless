import numpy as np

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

extdir = '/Volumes/Joshi_external_HDD/Roman/'
gal_fit_dir = extdir + 'sn_sit_hackday/testv3/'
results_dir = gal_fit_dir + 'Prism_deep_hostIav3/results/'

def old_prior_comparison():
    # ----------------
    # Read in catalogs
    zcat_deep_noprior = np.genfromtxt(gal_fit_dir + 'Prism_deep_hostIav3/results/' + \
        'zrecovery_results_deep_nozprior.txt', dtype=None, names=True, encoding='ascii')
    zcat_deep_withzprior = np.genfromtxt(gal_fit_dir + 'Prism_deep_hostIav3/results/photzprior/' + \
        'zrecovery_results_deep_withzprior.txt', dtype=None, names=True, encoding='ascii')

    zcat_shallow_noprior = np.genfromtxt(gal_fit_dir + 'Prism_shallow_hostIav3/results/' + \
        'zrecovery_results_shallow_nozprior.txt', dtype=None, names=True, encoding='ascii')
    zcat_shallow_withzprior = np.genfromtxt(gal_fit_dir + 'Prism_shallow_hostIav3/results/photzprior/' + \
        'zrecovery_results_shallow_withzprior.txt', dtype=None, names=True, encoding='ascii')

    # Get error arrays in correct shape
    zerr_deep_noprior = np.vstack((zcat_deep_noprior['zerr_low'], zcat_deep_noprior['zerr_up']))
    zerr_deep_withzprior = np.vstack((zcat_deep_withzprior['zerr_low'], zcat_deep_withzprior['zerr_up']))

    zerr_shallow_noprior = np.vstack((zcat_shallow_noprior['zerr_low'], zcat_shallow_noprior['zerr_up']))
    zerr_shallow_withzprior = np.vstack((zcat_shallow_withzprior['zerr_low'], zcat_shallow_withzprior['zerr_up']))

    # Compute z accuracies
    zacc_deep_noprior = (zcat_deep_noprior['z_corner'] - zcat_deep_noprior['z_truth']) / \
                        (1 + zcat_deep_noprior['z_truth'])
    zacc_deep_withzprior = (zcat_deep_withzprior['z_corner'] - zcat_deep_withzprior['z_truth']) / \
                        (1 + zcat_deep_withzprior['z_truth'])

    zacc_shallow_noprior = (zcat_shallow_noprior['z_corner'] - zcat_shallow_noprior['z_truth']) / \
                        (1 + zcat_shallow_noprior['z_truth'])
    zacc_shallow_withzprior = (zcat_shallow_withzprior['z_corner'] - zcat_shallow_withzprior['z_truth']) / \
                        (1 + zcat_shallow_withzprior['z_truth'])

    # ----------------
    # Make figure
    fig = plt.figure(figsize=(7, 10))

    gs = fig.add_gridspec(nrows=11, ncols=1, left=0.05, right=0.95, wspace=0.1)

    ax1 = fig.add_subplot(gs[:8])
    ax2 = fig.add_subplot(gs[8:])

    ax2.set_ylabel(r'$(z_\mathrm{inferred} - z_\mathrm{true} ) / (1 + z_\mathrm{true})$', fontsize=20)
    ax2.set_xlabel(r'$z_\mathrm{true}$', fontsize=20)

    ax1.set_ylabel(r'$z_\mathrm{inferred}$', fontsize=20)

    x_arr = np.arange(0.0, 3.0, 0.01)
    ax1.plot(x_arr, x_arr, '--', color='gray', lw=2.0)

    ax1.errorbar(zcat_deep_noprior['z_truth'], zcat_deep_noprior['z_corner'], 
        yerr=zerr_deep_noprior, fmt='o', color='k', elinewidth=0.5,
        label='Deep no z prior')
    ax1.errorbar(zcat_deep_withzprior['z_truth'], zcat_deep_withzprior['z_corner'], 
        yerr=zerr_deep_withzprior, fmt='o', color='crimson', elinewidth=0.5,
        label='Deep with z prior')

    ax1.errorbar(zcat_shallow_noprior['z_truth'], zcat_shallow_noprior['z_corner'], 
        yerr=zerr_shallow_noprior, fmt='x', color='k', elinewidth=0.5,
        label='Shallow no z prior')
    ax1.errorbar(zcat_shallow_withzprior['z_truth'], zcat_shallow_withzprior['z_corner'], 
        yerr=zerr_shallow_withzprior, fmt='x', color='crimson', elinewidth=0.5,
        label='Shallow with z prior')


    ax2.errorbar(zcat_deep_noprior['z_truth'], zacc_deep_noprior, 
        yerr=zerr_deep_noprior, fmt='o', color='k', elinewidth=0.5)
    ax2.errorbar(zcat_deep_withzprior['z_truth'], zacc_deep_withzprior, 
        yerr=zerr_deep_withzprior, fmt='o', color='crimson', elinewidth=0.5)

    ax2.errorbar(zcat_shallow_noprior['z_truth'], zacc_shallow_noprior, 
        yerr=zerr_shallow_noprior, fmt='x', color='k', elinewidth=0.5)
    ax2.errorbar(zcat_shallow_withzprior['z_truth'], zacc_shallow_withzprior, 
        yerr=zerr_shallow_withzprior, fmt='x', color='crimson', elinewidth=0.5)

    ax2.axhline(y=0.0, ls='--', color='gray', lw=1.5)


    # Ticks
    ax1.set_xticklabels([])

    # LEgend
    ax1.legend(loc=0, frameon=False, fontsize=12)

    # Limits
    ax1.set_xlim(0.0, 3.0)
    ax1.set_ylim(0.0, 3.0)

    ax2.set_xlim(0.0, 3.0)
    ax2.set_ylim(-1.0, 1.0)

    # Save fig
    fig.savefig(gal_fit_dir + 'zrecovery_comparison_testv3.pdf', dpi=200, bbox_inches='tight')

    return None

cat = np.genfromtxt(results_dir + 'zrecovery_results_deep.txt', 
                    dtype=None, names=True, encoding='ascii')

# -------
fig = plt.figure(figsize=(7, 10))

gs = fig.add_gridspec(nrows=11, ncols=1, left=0.05, right=0.95, wspace=0.1)

ax1 = fig.add_subplot(gs[:8])
ax2 = fig.add_subplot(gs[8:])

ax2.set_ylabel(r'$(z_\mathrm{inferred} - z_\mathrm{true} ) / (1 + z_\mathrm{true})$', fontsize=20)
ax2.set_xlabel(r'$z_\mathrm{true}$', fontsize=20)

ax1.set_ylabel(r'$z_\mathrm{inferred}$', fontsize=20)

x_arr = np.arange(0.0, 3.0, 0.01)
ax1.plot(x_arr, x_arr, '--', color='gray', lw=2.0)

# get error and accuracy
zerr_deep = np.vstack((cat['zerr_low'], cat['zerr_up']))
zacc_deep = (cat['z_corner'] - cat['z_truth']) / (1 + cat['z_truth'])

ax1.errorbar(cat['z_truth'], cat['z_corner'], 
    yerr=zerr_deep, markersize=3.0, fmt='o', color='k', elinewidth=0.5)

ax2.errorbar(cat['z_truth'], zacc_deep, 
    yerr=zerr_deep, markersize=3.0, fmt='o', color='k', elinewidth=0.5)
ax2.axhline(y=0.0, ls='--', color='gray', lw=2.0)

# Print info
fail_idx = np.where(np.abs(zacc_deep) >= 0.1)[0]
print('Fail IDs:', cat['ObjID'][fail_idx])

# remove outliers
zacc_deep[fail_idx] = np.nan

print('Mean acc:', np.nanmean(np.abs(zacc_deep)))
print('Median acc:', np.nanmedian(np.abs(zacc_deep)))

# Ticks
ax1.set_xticklabels([])

# Limits
ax1.set_xlim(0.0, 3.0)
ax1.set_ylim(0.0, 3.0)

ax2.set_xlim(0.0, 3.0)
ax2.set_ylim(-0.4, 0.4)

fig.savefig(results_dir + 'zrecovery_testv3.pdf', 
    dpi=200, bbox_inches='tight')

