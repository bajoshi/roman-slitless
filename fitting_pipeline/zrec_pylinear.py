import numpy as np
import emcee

import os
import sys

import matplotlib.pyplot as plt

ext_spectra_dir = "/Volumes/Joshi_external_HDD/Roman/roman_slitless_sims_results/"
results_dir = ext_spectra_dir + 'fitting_results/'

resfile = results_dir + 'zrecovery_pylinear_sims_pt0.txt'
cat = np.genfromtxt(resfile, dtype=None, names=True, encoding='ascii')

# Remove invalid measures
z900 = cat['z900']
z900[z900 == -9999.0] = np.nan

z1800 = cat['z1800']
z1800[z1800 == -9999.0] = np.nan

z3600 = cat['z3600']
z3600[z3600 == -9999.0] = np.nan

# ---
phase900 = cat['phase900']
phase900[phase900 == -9999.0] = np.nan

phase1800 = cat['phase1800']
phase1800[phase1800 == -9999.0] = np.nan

phase3600 = cat['phase3600']
phase3600[phase3600 == -9999.0] = np.nan

# ---
av900 = cat['Av900']
av900[av900 == -9999.0] = np.nan

av1800 = cat['Av1800']
av1800[av1800 == -9999.0] = np.nan

av3600 = cat['Av3600']
av3600[av3600 == -9999.0] = np.nan

###########################################
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

ax1.errorbar(cat['z_true'], z900, 
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

ax1.errorbar(cat['z_true'], z1800, 
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

ax1.errorbar(cat['z_true'], z3600, 
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

fig.savefig(results_dir + 'pylinearrecovery_z.pdf', 
    dpi=200, bbox_inches='tight')

fig.clear()
plt.close(fig)

print('IDs for 3600 sec catastrophic fails:')
for i in range(len(fail_idx3600)):
    print(cat['img_suffix'][fail_idx3600][i], '  ', cat['SNSegID'][fail_idx3600][i], '  ', \
        cat['z_true'][fail_idx3600][i], '  ', cat['z3600'][fail_idx3600][i])

sys.exit(0)

###########################################
# -------------------- phase recovery plt
fig = plt.figure(figsize=(7, 9))

gs = fig.add_gridspec(nrows=11, ncols=1, left=0.05, right=0.95, wspace=0.1)

ax1 = fig.add_subplot(gs[:8])
ax2 = fig.add_subplot(gs[8:])

ax2.set_ylabel(r'$\Delta \mathrm{Phase}$', fontsize=20)
ax2.set_xlabel(r'$\mathrm{Phase_{true}}$', fontsize=20)

ax1.set_ylabel(r'$\mathrm{Phase_{inferred}}$', fontsize=20)

x_arr = np.arange(-19, 51)
ax1.plot(x_arr, x_arr, '--', color='gray', lw=2.0)

# Set up error arrays
phase900err = np.vstack((cat['phase900_lowerr'], cat['phase900_uperr']))
phase1800err = np.vstack((cat['phase1800_lowerr'], cat['phase1800_uperr']))
phase3600err = np.vstack((cat['phase3600_lowerr'], cat['phase3600_uperr']))

# --- EXPTIME 900 seconds
ax1.errorbar(cat['phase_true'], phase900, 
    yerr=phase900err, markersize=3.0, fmt='o', color='goldenrod', elinewidth=0.5)

ax2.errorbar(cat['phase_true'], phase900 - cat['phase_true'], 
    yerr=phase900err, markersize=3.0, fmt='o', color='goldenrod', elinewidth=0.5)
ax2.axhline(y=0.0, ls='--', color='gray', lw=2.0)

# --- EXPTIME 1800 seconds
ax1.errorbar(cat['phase_true'], phase1800, 
    yerr=phase1800err, markersize=3.0, fmt='o', color='dodgerblue', elinewidth=0.5)

ax2.errorbar(cat['phase_true'], phase1800 - cat['phase_true'], 
    yerr=phase1800err, markersize=3.0, fmt='o', color='dodgerblue', elinewidth=0.5)
ax2.axhline(y=0.0, ls='--', color='gray', lw=2.0)

# --- EXPTIME 3600 seconds
ax1.errorbar(cat['phase_true'], phase3600, 
    yerr=phase3600err, markersize=3.0, fmt='o', color='crimson', elinewidth=0.5)

ax2.errorbar(cat['phase_true'], phase3600 - cat['phase_true'], 
    yerr=phase3600err, markersize=3.0, fmt='o', color='crimson', elinewidth=0.5)
ax2.axhline(y=0.0, ls='--', color='gray', lw=2.0)

# Ticks
ax1.set_xticklabels([])

# Limits
ax1.set_xlim(-19, 50)
ax1.set_ylim(-19, 50)

ax2.set_xlim(-19, 50)

fig.savefig(results_dir + 'pylinearrecovery_phase.pdf', 
    dpi=200, bbox_inches='tight')

fig.clear()
plt.close(fig)

###########################################
# -------------------- Av recovery plt
fig = plt.figure(figsize=(7, 9))

gs = fig.add_gridspec(nrows=11, ncols=1, left=0.05, right=0.95, wspace=0.1)

ax1 = fig.add_subplot(gs[:8])
ax2 = fig.add_subplot(gs[8:])

ax2.set_ylabel(r'$\Delta \mathrm{A_v}$', fontsize=20)
ax2.set_xlabel(r'$\mathrm{A_{v;\, true}}$', fontsize=20)

ax1.set_ylabel(r'$\mathrm{A_{v;\, inferred}}$', fontsize=20)

x_arr = np.arange(0.01, 5.01, 0.01)
ax1.plot(x_arr, x_arr, '--', color='gray', lw=2.0)

# Set up error arrays
av900err = np.vstack((cat['Av900_lowerr'], cat['Av900_uperr']))
av1800err = np.vstack((cat['Av1800_lowerr'], cat['Av1800_uperr']))
av3600err = np.vstack((cat['Av3600_lowerr'], cat['Av3600_uperr']))

# --- EXPTIME 900 seconds
ax1.errorbar(cat['Av_true'], av900, 
    yerr=av900err, markersize=3.0, fmt='o', color='goldenrod', elinewidth=0.5)

ax2.errorbar(cat['Av_true'], av900 - cat['Av_true'], 
    yerr=av900err, markersize=3.0, fmt='o', color='goldenrod', elinewidth=0.5)
ax2.axhline(y=0.0, ls='--', color='gray', lw=2.0)

# --- EXPTIME 1800 seconds
ax1.errorbar(cat['Av_true'], av1800, 
    yerr=av1800err, markersize=3.0, fmt='o', color='dodgerblue', elinewidth=0.5)

ax2.errorbar(cat['Av_true'], av1800 - cat['Av_true'], 
    yerr=av1800err, markersize=3.0, fmt='o', color='dodgerblue', elinewidth=0.5)
ax2.axhline(y=0.0, ls='--', color='gray', lw=2.0)

# --- EXPTIME 3600 seconds
ax1.errorbar(cat['Av_true'], av3600, 
    yerr=av3600err, markersize=3.0, fmt='o', color='crimson', elinewidth=0.5)

ax2.errorbar(cat['Av_true'], av3600 - cat['Av_true'], 
    yerr=av3600err, markersize=3.0, fmt='o', color='crimson', elinewidth=0.5)
ax2.axhline(y=0.0, ls='--', color='gray', lw=2.0)

# Ticks
ax1.set_xticklabels([])

# Limits
ax1.set_xlim(0.0, 5.0)
ax1.set_ylim(0.0, 5.0)

ax2.set_xlim(0.0, 5.0)


fig.savefig(results_dir + 'pylinearrecovery_dust.pdf', 
    dpi=200, bbox_inches='tight')

fig.clear()
plt.close(fig)

###########################################
# -------------------- Plot SNR vs % accuracy
fig = plt.figure(figsize=(9, 5))

gs = fig.add_gridspec(nrows=12, ncols=1, left=0.15, right=0.95, wspace=0.1)

ax1 = fig.add_subplot(gs[:4])
ax2 = fig.add_subplot(gs[4:8])
ax3 = fig.add_subplot(gs[8:])

# Axis labels
ax1.set_ylabel(r'$\frac{z_\mathrm{inferred} - z_\mathrm{true}}{1 + z_\mathrm{true}}$', fontsize=15)
ax2.set_ylabel(r'$\Delta \mathrm{Phase}$', fontsize=15)
ax3.set_ylabel(r'$\Delta \mathrm{A_v}$', fontsize=15)

ax3.set_xlabel(r'$\mathrm{SNR}$', fontsize=15)

# Plotting
ax1.scatter(cat['SNR3600'], z3600acc, s=7, color='k',  zorder=2)
ax1.axhline(y=0.0, ls='--', lw=2.0, color='gray', zorder=1)

ax2.scatter(cat['SNR3600'], phase3600 - cat['phase_true'], s=7, color='k',  zorder=2)
ax2.axhline(y=0.0, ls='--', lw=2.0, color='gray', zorder=1)

ax3.scatter(cat['SNR3600'], av3600 - cat['Av_true'], s=7, color='k',  zorder=2)
ax3.axhline(y=0.0, ls='--', lw=2.0, color='gray', zorder=1)

#ax1.set_xlim(10, 150)
#ax2.set_xlim(10, 150)
#ax3.set_xlim(10, 150)

ax1.set_ylim(-0.015, 0.015)

fig.savefig(results_dir + 'pylinearrecovery_snr.pdf',
    dpi=200, bbox_inches='tight')







