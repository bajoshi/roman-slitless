import numpy as np

import os
import sys

import matplotlib.pyplot as plt

home = os.getenv('HOME')
roman_slitless = home + '/Documents/GitHub/roman-slitless/'

extdir = "/Volumes/Joshi_external_HDD/Roman/"

ext_spectra_dir = extdir + "roman_slitless_sims_results/"
results_dir = ext_spectra_dir + 'fitting_results/'

resfile = results_dir + 'zrecovery_pylinear_sims_pt0.txt'
cat = np.genfromtxt(resfile, dtype=None, names=True, encoding='ascii')

# Remove invalid measures
z400 = cat['z400']
z400[z400 == -9999.0] = np.nan

z1200 = cat['z1200']
z1200[z1200 == -9999.0] = np.nan

z3600 = cat['z3600']
z3600[z3600 == -9999.0] = np.nan

# ---
phase400 = cat['phase400']
phase400[phase400 == -9999.0] = np.nan

phase1200 = cat['phase1200']
phase1200[phase1200 == -9999.0] = np.nan

phase3600 = cat['phase3600']
phase3600[phase3600 == -9999.0] = np.nan

# ---
av400 = cat['Av400']
av400[av400 == -9999.0] = np.nan

av1200 = cat['Av1200']
av1200[av1200 == -9999.0] = np.nan

av3600 = cat['Av3600']
av3600[av3600 == -9999.0] = np.nan

###########################################
# --------------------
fig = plt.figure(figsize=(7, 9))

gs = fig.add_gridspec(nrows=11, ncols=1, left=0.05, right=0.95, wspace=0.1)

ax1 = fig.add_subplot(gs[:8])
ax2 = fig.add_subplot(gs[8:])

# For legend
all_exptimes = ['20m', '1h', '3h']

# Labels
resid_y_label = \
    r'$(z - z_\mathrm{true} ) / (1 + z_\mathrm{true})$'

ax2.set_ylabel(resid_y_label, fontsize=20)
ax2.set_xlabel(r'$z_\mathrm{true}$', fontsize=20)

ax1.set_ylabel(r'$z$', fontsize=20)

# Plot 1:1 lines
x_arr = np.arange(0.0, 3.5, 0.01)
ax1.plot(x_arr, x_arr, '--', color='gray', lw=2.0)
ax2.axhline(y=0.0, ls='--', color='gray', lw=2.0)

# --- EXPTIME 400 seconds
# get error and accuracy
z400err = np.vstack((cat['z400_lowerr'], cat['z400_uperr']))
z400acc = (z400 - cat['z_true']) / (1 + cat['z_true'])

ax1.errorbar(cat['z_true'], z400,
             yerr=z400err, markersize=6.5, fmt='o',
             color='white', ecolor='goldenrod',
             markeredgecolor='goldenrod', elinewidth=0.1,
             label=r'$t_\mathrm{exp}\, =\ $' + all_exptimes[0])

ax2.errorbar(cat['z_true'], z400acc,
             yerr=z400err, markersize=6.5, fmt='o',
             color='white', ecolor='goldenrod',
             markeredgecolor='goldenrod', elinewidth=0.1)

# Print info
fail_idx400 = np.where(np.abs(z400acc) >= 0.1)[0]

# print('\nCatastrophic failure fraction 400 seconds:',
#       "{:.4f}".format(len(fail_idx400) / len(cat)))

# remove outliers
z400acc[fail_idx400] = np.nan

print('Mean acc 400 seconds:',
      "{:.4f}".format(np.nanmean(np.abs(z400acc))))
# print('Median acc 400 seconds:',
#       "{:.4f}".format(np.nanmedian(np.abs(z400acc))))

# --- EXPTIME 1200 seconds
# get error and accuracy
z1200err = np.vstack((cat['z1200_lowerr'], cat['z1200_uperr']))
z1200acc = (z1200 - cat['z_true']) / (1 + cat['z_true'])

ax1.errorbar(cat['z_true'], z1200,
             yerr=z1200err, markersize=4.5, fmt='o',
             color='white', ecolor='seagreen',
             markeredgecolor='seagreen', elinewidth=0.1,
             label=r'$t_\mathrm{exp}\, =\ $' + all_exptimes[1])

ax2.errorbar(cat['z_true'], z1200acc,
             yerr=z1200err, markersize=4.5, fmt='o',
             color='white', ecolor='seagreen',
             markeredgecolor='seagreen', elinewidth=0.1)

# Print info
fail_idx1200 = np.where(np.abs(z1200acc) >= 0.1)[0]

# print('\nCatastrophic failure fraction 1200 seconds:',
#       "{:.4f}".format(len(fail_idx1200) / len(cat)))

# remove outliers
z1200acc[fail_idx1200] = np.nan

print('Mean acc 1200 seconds:',
      "{:.4f}".format(np.nanmean(np.abs(z1200acc))))
# print('Median acc 1200 seconds:',
#       "{:.4f}".format(np.nanmedian(np.abs(z1200acc))))


# --- EXPTIME 3600 seconds
# get error and accuracy
z3600err = np.vstack((cat['z3600_lowerr'], cat['z3600_uperr']))
z3600acc = (z3600 - cat['z_true']) / (1 + cat['z_true'])

ax1.errorbar(cat['z_true'], z3600,
             yerr=z3600err, markersize=2.0, fmt='o',
             color='white', ecolor='dodgerblue',
             markeredgecolor='dodgerblue', elinewidth=0.1,
             label=r'$t_\mathrm{exp}\, =\ $' + all_exptimes[2])

ax2.errorbar(cat['z_true'], z3600acc,
             yerr=z3600err, markersize=2.0, fmt='o',
             color='white', ecolor='dodgerblue',
             markeredgecolor='dodgerblue', elinewidth=0.1)

# Print info
fail_idx3600 = np.where(np.abs(z3600acc) >= 0.1)[0]

# print('\nCatastrophic failure fraction 3600 seconds:',
#       "{:.4f}".format(len(fail_idx3600) / len(cat)))

# remove outliers
z3600acc[fail_idx3600] = np.nan

print('Mean acc 3600 seconds:',
      "{:.4f}".format(np.nanmean(np.abs(z3600acc))))
# print('Median acc 3600 seconds:',
#       "{:.4f}".format(np.nanmedian(np.abs(z3600acc))))


# Ticks
ax1.set_xticklabels([])

# Limits
ax1.set_xlim(0.0, 3.2)
ax1.set_ylim(0.0, 3.2)

ax2.set_xlim(0.0, 3.2)
# ax2.set_ylim(-0.01, 0.01)

ax1.legend(loc=0, fontsize=18, frameon=False)

fig.savefig(roman_slitless + 'figures/pylinearrecovery_z.pdf',
            dpi=200, bbox_inches='tight')

fig.clear()
plt.close(fig)


###########################################
# -------------------- phase recovery plt
fig = plt.figure(figsize=(7, 9))

gs = fig.add_gridspec(nrows=11, ncols=1, left=0.05, right=0.95, wspace=0.1)

ax1 = fig.add_subplot(gs[:8])
ax2 = fig.add_subplot(gs[8:])

ax2.set_ylabel(r'$\Delta \mathrm{Phase}$', fontsize=20)
ax2.set_xlabel(r'$\mathrm{Phase_{true}}$', fontsize=20)

ax1.set_ylabel(r'$\mathrm{Phase}$', fontsize=20)

x_arr = np.arange(-19, 51)
ax1.plot(x_arr, x_arr, '--', color='gray', lw=2.0)
ax2.axhline(y=0.0, ls='--', color='gray', lw=2.0)

# Set up error arrays
phase400err = np.vstack((cat['phase400_lowerr'], cat['phase400_uperr']))
phase1200err = np.vstack((cat['phase1200_lowerr'], cat['phase1200_uperr']))
phase3600err = np.vstack((cat['phase3600_lowerr'], cat['phase3600_uperr']))

# --- EXPTIME 400 seconds
ax1.errorbar(cat['phase_true'], phase400,
             yerr=phase400err, markersize=6.5,
             fmt='o', color='white', ecolor='goldenrod',
             markeredgecolor='goldenrod', elinewidth=0.5,
             label=r'$t_\mathrm{exp}\, =\ $' + all_exptimes[0])

ax2.errorbar(cat['phase_true'], phase400 - cat['phase_true'],
             yerr=phase400err, markersize=6.5,
             fmt='o', color='white', ecolor='goldenrod',
             markeredgecolor='goldenrod', elinewidth=0.5)

# --- EXPTIME 1200 seconds
ax1.errorbar(cat['phase_true'], phase1200,
             yerr=phase1200err, markersize=4.5,
             fmt='o', color='white', ecolor='seagreen',
             markeredgecolor='seagreen', elinewidth=0.5,
             label=r'$t_\mathrm{exp}\, =\ $' + all_exptimes[1])

ax2.errorbar(cat['phase_true'], phase1200 - cat['phase_true'],
             yerr=phase1200err, markersize=4.5,
             fmt='o', color='white', ecolor='seagreen',
             markeredgecolor='seagreen', elinewidth=0.5)

# --- EXPTIME 3600 seconds
ax1.errorbar(cat['phase_true'], phase3600,
             yerr=phase3600err, markersize=2.0,
             fmt='o', color='white', ecolor='dodgerblue',
             markeredgecolor='dodgerblue', elinewidth=0.5,
             label=r'$t_\mathrm{exp}\, =\ $' + all_exptimes[2])

ax2.errorbar(cat['phase_true'], phase3600 - cat['phase_true'],
             yerr=phase3600err, markersize=2.0,
             fmt='o', color='white', ecolor='dodgerblue',
             markeredgecolor='dodgerblue', elinewidth=0.5)

# Ticks
ax1.set_xticklabels([])

# Limits
ax1.set_xlim(-10, 10)
ax1.set_ylim(-10, 10)

ax2.set_xlim(-10, 10)

ax1.legend(loc=0, fontsize=18, frameon=False)

fig.savefig(roman_slitless + 'figures/pylinearrecovery_phase.pdf',
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

ax1.set_ylabel(r'$\mathrm{A_{v}}$', fontsize=20)

x_arr = np.arange(0.01, 3.01, 0.01)
ax1.plot(x_arr, x_arr, '--', color='gray', lw=2.0)
ax2.axhline(y=0.0, ls='--', color='gray', lw=2.0)

# Set up error arrays
av400err = np.vstack((cat['Av400_lowerr'], cat['Av400_uperr']))
av1200err = np.vstack((cat['Av1200_lowerr'], cat['Av1200_uperr']))
av3600err = np.vstack((cat['Av3600_lowerr'], cat['Av3600_uperr']))

# --- EXPTIME 400 seconds
ax1.errorbar(cat['Av_true'], av400,
             yerr=av400err, markersize=6.5, fmt='o',
             color='white',
             markeredgecolor='goldenrod', ecolor='goldenrod',
             elinewidth=0.1,
             label=r'$t_\mathrm{exp}\, =\ $' + all_exptimes[0])

ax2.errorbar(cat['Av_true'], av400 - cat['Av_true'],
             yerr=av400err, markersize=6.5, fmt='o',
             color='white',
             markeredgecolor='goldenrod', ecolor='goldenrod',
             elinewidth=0.1)


# --- EXPTIME 1200 seconds
ax1.errorbar(cat['Av_true'], av1200,
             yerr=av1200err, markersize=4.5, fmt='o',
             color='white',
             markeredgecolor='seagreen', ecolor='seagreen',
             elinewidth=0.1,
             label=r'$t_\mathrm{exp}\, =\ $' + all_exptimes[1])

ax2.errorbar(cat['Av_true'], av1200 - cat['Av_true'],
             yerr=av1200err, markersize=4.5, fmt='o',
             color='white',
             markeredgecolor='seagreen', ecolor='seagreen',
             elinewidth=0.1)

# --- EXPTIME 3600 seconds
ax1.errorbar(cat['Av_true'], av3600,
             yerr=av3600err, markersize=2.0, fmt='o',
             color='white',
             markeredgecolor='dodgerblue', ecolor='dodgerblue',
             elinewidth=0.1,
             label=r'$t_\mathrm{exp}\, =\ $' + all_exptimes[2])

ax2.errorbar(cat['Av_true'], av3600 - cat['Av_true'],
             yerr=av3600err, markersize=2.0, fmt='o',
             color='white',
             markeredgecolor='dodgerblue', ecolor='dodgerblue',
             elinewidth=0.1)

# Ticks
ax1.set_xticklabels([])

# Limits
ax1.set_xlim(0.0, 3.0)
ax1.set_ylim(0.0, 3.0)

ax2.set_xlim(0.0, 3.0)

ax1.legend(loc=0, fontsize=18, frameon=False)

fig.savefig(roman_slitless + 'figures/pylinearrecovery_dust.pdf',
            dpi=200, bbox_inches='tight')

fig.clear()
plt.close(fig)

###########################################
# -------------------- Plot SNR vs % accuracy
# get error and accuracy
z3600err = np.vstack((cat['z3600_lowerr'], cat['z3600_uperr']))
z3600acc = (z3600 - cat['z_true']) / (1 + cat['z_true'])

fig = plt.figure(figsize=(9, 5))

gs = fig.add_gridspec(nrows=12, ncols=1, left=0.15, right=0.95, wspace=0.1)

ax1 = fig.add_subplot(gs[:6])
ax2 = fig.add_subplot(gs[6:])
# ax3 = fig.add_subplot(gs[8:])

# Axis labels
ax1.set_ylabel(r'$\frac{z_ - z_\mathrm{true}}{1 + z_\mathrm{true}}$',
               fontsize=15)
ax2.set_ylabel(r'$\Delta \mathrm{Phase}$', fontsize=15)
# ax3.set_ylabel(r'$\Delta \mathrm{A_v}$', fontsize=15)
ax2.set_xlabel(r'$\mathrm{SNR}$', fontsize=15)

# Plotting
ax1.scatter(cat['SNR3600'], z3600acc, s=7, color='k', zorder=2)
ax1.axhline(y=0.0, ls='--', lw=2.0, color='gray', zorder=1)

ax2.scatter(cat['SNR3600'], phase3600 - cat['phase_true'],
            s=7, color='k', zorder=2)
ax2.axhline(y=0.0, ls='--', lw=2.0, color='gray', zorder=1)

# ax3.scatter(cat['SNR6000'], av6000 - cat['Av_true'],
#             s=7, color='k', zorder=2)
# ax3.axhline(y=0.0, ls='--', lw=2.0, color='gray', zorder=1)

# ax1.set_xlim(10, 150)
# ax2.set_xlim(10, 150)
# ax3.set_xlim(10, 150)

# ax1.set_ylim(-0.015, 0.015)

fig.savefig(roman_slitless + 'figures/pylinearrecovery_snr.pdf',
            dpi=200, bbox_inches='tight')

fig.clear()
plt.close(fig)
sys.exit(0)

###########################################
# --------- For modifying FITRES file from Dan
# need a 'functional form' of z error vs SNR.
snr = cat['SNR3600']
snr_idx = np.where((snr >= 10.0) & (snr <= 20.0))[0]

print(z3600acc[snr_idx])
print(np.nanmax(z3600acc[snr_idx]))

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(snr[snr_idx], z3600acc[snr_idx], s=7, color='k', zorder=2)
ax.axhline(y=0.0, ls='--', lw=2.0, color='gray', zorder=1)

plt.show()
