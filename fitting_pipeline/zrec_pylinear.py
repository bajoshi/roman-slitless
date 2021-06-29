import numpy as np
import emcee

import os
import sys

import matplotlib.pyplot as plt

ext_spectra_dir = "/Volumes/Joshi_external_HDD/Roman/roman_slitless_sims_results/"
results_dir = ext_spectra_dir + 'fitting_results/'

resfile = results_dir + 'zrecovery_pylinear_sims.txt'
cat = np.genfromtxt(resfile, dtype=None, names=True, encoding='ascii')

# Remove invalid measures
z900 = cat['z900']
z900[z900 == -9999.0] = np.nan

z1800 = cat['z1800']
z1800[z1800 == -9999.0] = np.nan

z3600 = cat['z3600']
z3600[z3600 == -9999.0] = np.nan


# --------------------
fig = plt.figure(figsize=(7, 9))

gs = fig.add_gridspec(nrows=11, ncols=1, left=0.05, right=0.95, wspace=0.1)

ax1 = fig.add_subplot(gs[:8])
ax2 = fig.add_subplot(gs[8:])

ax2.set_ylabel(r'$(z_\mathrm{inferred} - z_\mathrm{true} ) / (1 + z_\mathrm{true})$', fontsize=20)
ax2.set_xlabel(r'$z_\mathrm{true}$', fontsize=20)

ax1.set_ylabel(r'$z_\mathrm{inferred}$', fontsize=20)

x_arr = np.arange(0.0, 3.0, 0.01)
ax1.plot(x_arr, x_arr, '--', color='gray', lw=2.0)

# --- EXPTIME 900 seconds
# get error and accuracy
z900err = np.vstack((cat['z900_lowerr'], cat['z900_uperr']))
z900acc = (z900 - cat['z_true']) / (1 + cat['z_true'])

ax1.errorbar(cat['z_true'], cat['z900'], 
    yerr=z900err, markersize=3.0, fmt='o', color='goldenrod', elinewidth=0.5)

ax2.errorbar(cat['z_true'], z900acc, 
    yerr=z900err, markersize=3.0, fmt='o', color='goldenrod', elinewidth=0.5)
ax2.axhline(y=0.0, ls='--', color='gray', lw=2.0)

# Print info
fail_idx900 = np.where(np.abs(z900acc) >= 0.1)[0]

print('\nCatastrophic failure fraction 900 seconds:', "{:.4f}".format(len(fail_idx900) / len(cat)))

# remove outliers
z900acc[fail_idx900] = np.nan

print('Mean acc 900 seconds:', "{:.4f}".format(np.nanmean(np.abs(z900acc))))
print('Median acc 900 seconds:', "{:.4f}".format(np.nanmedian(np.abs(z900acc))))





# --- EXPTIME 1800 seconds
# get error and accuracy
z1800err = np.vstack((cat['z1800_lowerr'], cat['z1800_uperr']))
z1800acc = (z1800 - cat['z_true']) / (1 + cat['z_true'])

ax1.errorbar(cat['z_true'], cat['z1800'], 
    yerr=z1800err, markersize=3.0, fmt='o', color='dodgerblue', elinewidth=0.5)

ax2.errorbar(cat['z_true'], z1800acc, 
    yerr=z1800err, markersize=3.0, fmt='o', color='dodgerblue', elinewidth=0.5)

# Print info
fail_idx1800 = np.where(np.abs(z1800acc) >= 0.1)[0]

print('\nCatastrophic failure fraction 1800 seconds:', "{:.4f}".format(len(fail_idx1800) / len(cat)))

# remove outliers
z1800acc[fail_idx1800] = np.nan

print('Mean acc 1800 seconds:', "{:.4f}".format(np.nanmean(np.abs(z1800acc))))
print('Median acc 1800 seconds:', "{:.4f}".format(np.nanmedian(np.abs(z1800acc))))





# --- EXPTIME 3600 seconds
# get error and accuracy
z3600err = np.vstack((cat['z3600_lowerr'], cat['z3600_uperr']))
z3600acc = (z3600 - cat['z_true']) / (1 + cat['z_true'])

ax1.errorbar(cat['z_true'], cat['z3600'], 
    yerr=z3600err, markersize=3.0, fmt='o', color='crimson', elinewidth=0.5)

ax2.errorbar(cat['z_true'], z3600acc, 
    yerr=z3600err, markersize=3.0, fmt='o', color='crimson', elinewidth=0.5)

# Print info
fail_idx3600 = np.where(np.abs(z3600acc) >= 0.1)[0]

print('\nCatastrophic failure fraction 3600 seconds:', "{:.4f}".format(len(fail_idx3600) / len(cat)))

# remove outliers
z3600acc[fail_idx3600] = np.nan

print('Mean acc 3600 seconds:', "{:.4f}".format(np.nanmean(np.abs(z3600acc))))
print('Median acc 3600 seconds:', "{:.4f}".format(np.nanmedian(np.abs(z3600acc))))


# Ticks
ax1.set_xticklabels([])

# Limits
ax1.set_xlim(0.0, 3.0)
ax1.set_ylim(0.0, 3.0)

ax2.set_xlim(0.0, 3.0)
#ax2.set_ylim(-0.01, 0.01)

#plt.show()

fig.savefig(results_dir + 'zrecovery_pylinear.pdf', 
    dpi=200, bbox_inches='tight')

# -------------------- z accuracy vs SNR plt







