import numpy as np
from astropy.io import fits

import os
import sys

import matplotlib.pyplot as plt

home = os.getenv('HOME')
roman_slitless = home + '/Documents/GitHub/roman-slitless/'

extdir = "/Volumes/Joshi_external_HDD/Roman/"

ext_spectra_dir = extdir + "roman_slitless_sims_results/"
fitting_resdir = ext_spectra_dir + 'run1/fitting_results/'
resdir = extdir + 'roman_slitless_sims_results/run1/'
img_sim_dir = extdir + 'roman_direct_sims/sims2021/K_5degimages_part1/'

##############################################################################
# ------------------- PREP
resfile = fitting_resdir + 'zrecovery_pylinear_sims_pt0.txt'
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

# Make these plots only for the uncontaminated SNe??
overlap_idx = cat['overlap']

# flag for axis limits later
consider_contam_sne = True

if not consider_contam_sne:
    z400[overlap_idx] = np.nan
    z1200[overlap_idx] = np.nan
    z3600[overlap_idx] = np.nan

    phase400[overlap_idx] = np.nan
    phase1200[overlap_idx] = np.nan
    phase3600[overlap_idx] = np.nan

    av400[overlap_idx] = np.nan
    av1200[overlap_idx] = np.nan
    av3600[overlap_idx] = np.nan

# Error bar width
ebarwidth = 0.00

# Colors from colorbrewer
exp_400s_color = '#7570b3'
exp_1200s_color = '#d95f02'
exp_3600s_color = '#1b9e77'

##############################################################################
# --------------------
fig = plt.figure(figsize=(7, 9))

gs = fig.add_gridspec(nrows=14, ncols=1, left=0.05, right=0.95, wspace=0.1)

ax1 = fig.add_subplot(gs[:8])
ax2 = fig.add_subplot(gs[8:])

if consider_contam_sne:
    ax1.set_title('All SNe', fontsize=30)
else:
    ax1.set_title('No blended host-galaxy light', fontsize=30)

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

# PLot lines for misidentified abs features 
# from host in SN spectrum
host_feature_wav = 3951.5  # H+K average
sn_misID_wav = 6150
misID_z = ((host_feature_wav * (1+x_arr)) / sn_misID_wav) - 1
# ax1.plot(x_arr, misID_z, '--', color='r', lw=2.0)

# --- EXPTIME 400 seconds
# get error and accuracy
z400err = np.vstack((cat['z400_lowerr'], cat['z400_uperr']))
z400acc = (z400 - cat['z_true']) / (1 + cat['z_true'])

ax1.errorbar(cat['z_true'], z400,
             yerr=z400err, markersize=6.5, fmt='o',
             color='white', ecolor=exp_400s_color,
             markeredgecolor=exp_400s_color, elinewidth=ebarwidth,
             label=r'$t_\mathrm{exp}\, =\ $' + all_exptimes[0])

ax2.errorbar(cat['z_true'], z400acc,
             yerr=z400err, markersize=6.5, fmt='o',
             color='white', ecolor=exp_400s_color,
             markeredgecolor=exp_400s_color, elinewidth=ebarwidth)

# Print info
fail_idx400 = np.where(np.abs(z400acc) >= 0.1)[0]

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
             color='white', ecolor=exp_1200s_color,
             markeredgecolor=exp_1200s_color, elinewidth=ebarwidth,
             label=r'$t_\mathrm{exp}\, =\ $' + all_exptimes[1])

ax2.errorbar(cat['z_true'], z1200acc,
             yerr=z1200err, markersize=4.5, fmt='o',
             color='white', ecolor=exp_1200s_color,
             markeredgecolor=exp_1200s_color, elinewidth=ebarwidth)

# Print info
fail_idx1200 = np.where(np.abs(z1200acc) >= 0.1)[0]

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
             color='white', ecolor=exp_3600s_color,
             markeredgecolor=exp_3600s_color, elinewidth=ebarwidth,
             label=r'$t_\mathrm{exp}\, =\ $' + all_exptimes[2])

ax2.errorbar(cat['z_true'], z3600acc,
             yerr=z3600err, markersize=2.0, fmt='o',
             color='white', ecolor=exp_3600s_color,
             markeredgecolor=exp_3600s_color, elinewidth=ebarwidth)

# ===========================
# Investigate underestimated redshifts
sys.path.append(roman_slitless + 'fitting_pipeline/utils/')
from inspect_obj_x1d import underest_z_figure  # noqa

# Uncomment the case you need
# underestimates = np.where(z3600acc < -0.5)[0]

# underestimates = np.where((cat['z_true'] > 0.7)
#                           & (cat['z_true'] < 1.2)
#                           & (cat['z3600'] > 0.2)
#                           & (cat['z3600'] < 0.5))[0]

underestimates = np.where((z3600acc < -0.35) & (z3600acc > -0.45))[0]

total_underest = len(underestimates)
print('Total', total_underest,
      'underestimates out of', len(cat))

det_list = np.zeros(total_underest, dtype=np.int64)
segid_list = np.zeros(total_underest, dtype=np.int64)

ztru_list = np.zeros(total_underest)
zest_list = np.zeros(total_underest)

phase_tru_list = np.zeros(total_underest)
phase_list = np.zeros(total_underest)


for u in range(total_underest):
    print('------------------')
    print('IMG and ID :', cat['img_suffix'][underestimates[u]],
          cat['SNSegID'][underestimates[u]])
    print('z-tru:', cat['z_true'][underestimates[u]])
    print('z-est:', z3600[underestimates[u]])
    print('z-acc:', z3600acc[underestimates[u]])
    print('------------------\n')

    d = cat['img_suffix'][underestimates[u]].split('_')[-1]
    det_list[u] = int(d)
    segid_list[u] = int(cat['SNSegID'][underestimates[u]])

    ztru_list[u] = float(cat['z_true'][underestimates[u]])
    zest_list[u] = float(z3600[underestimates[u]])

    phase_tru_list[u] = cat['phase_true'][underestimates[u]]
    phase_list[u] = cat['phase3600'][underestimates[u]]

# Run through the underestimates for some detectors
# A bit more than a third of all SNe fail this way so
# you don't want to plot all of them
exptime = '3600s'
# all_detectors = np.arange(1, 19)  # needs a string here

all_detectors = ['1']

for detector in all_detectors:

    detector = str(detector)

    # ---------------
    # Read in extracted spectra
    x1d = fits.open(resdir + 'romansim_prism_Y106_0_'
                    + detector + '_' + exptime + '_x1d.fits')
    
    # ---------------
    # Read in sedlst
    sedlst_path = (extdir + 'pylinear_lst_files/run1/sed_Y106_0_'
                   + detector + '.lst')
    sedlst = np.genfromtxt(sedlst_path, dtype=None,
                           names=['segid', 'sed_path'], encoding='ascii')
    
    # ---------------
    # Now grab all underestimated z IDs in this detector
    det_underest = np.where(det_list == int(detector))[0]
    
    figdir = roman_slitless + 'figures/linear_feature/'
    
    for v in range(len(det_underest)):
        sn_segid = segid_list[det_underest[v]]

        ztru = ztru_list[det_underest[v]]
        zest = zest_list[det_underest[v]]

        phase_tru = phase_tru_list[det_underest[v]]
        phase = phase_list[det_underest[v]]

        print('\nPlotting SN', sn_segid, 'on detector', detector)
        print('z-true:', ztru)
        print('z-est:', zest)

        print('True phase:', phase_tru)
        print('Phase:', phase)
    
        underest_z_figure(x1d, sn_segid, sedlst, detector, exptime,
                          figdir, ztru, zest, phase_tru, phase)

sys.exit(0)

# ===========================

# Print info
fail_idx3600 = np.where(np.abs(z3600acc) >= 0.1)[0]

# remove outliers
z3600acc[fail_idx3600] = np.nan

print('Mean acc 3600 seconds:',
      "{:.4f}".format(np.nanmean(np.abs(z3600acc))))
# print('Median acc 3600 seconds:',
#       "{:.4f}".format(np.nanmedian(np.abs(z3600acc))))

print('\nCatastrophic failure fraction 400 seconds:',
      "{:.4f}".format(len(fail_idx400) / len(cat)))
print('\nCatastrophic failure fraction 1200 seconds:',
      "{:.4f}".format(len(fail_idx1200) / len(cat)))
print('\nCatastrophic failure fraction 3600 seconds:',
      "{:.4f}".format(len(fail_idx3600) / len(cat)))

# Ticks
ax1.set_xticklabels([])

# Limits
ax1.legend(loc=0, fontsize=18, frameon=False)

ax1.set_xlim(0.0, 3.2)
ax1.set_ylim(0.0, 3.2)

ax2.set_xlim(0.0, 3.2)

# Limit based on consideration of contam/uncontam sne
if not consider_contam_sne:
    ax2.set_ylim(-0.002, 0.002)
    fig.savefig(roman_slitless + 'figures/pylinearrecovery_z_noblend.pdf',
                dpi=200, bbox_inches='tight')
else:
    fig.savefig(roman_slitless + 'figures/pylinearrecovery_z_allsne.pdf',
                dpi=200, bbox_inches='tight')

fig.clear()
plt.close(fig)


###########################################
# -------------------- phase recovery plt
fig = plt.figure(figsize=(7, 9))

gs = fig.add_gridspec(nrows=14, ncols=1, left=0.05, right=0.95, wspace=0.1)

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
             fmt='o', color='white', ecolor=exp_400s_color,
             markeredgecolor=exp_400s_color, elinewidth=ebarwidth,
             label=r'$t_\mathrm{exp}\, =\ $' + all_exptimes[0])

ax2.errorbar(cat['phase_true'], phase400 - cat['phase_true'],
             yerr=phase400err, markersize=6.5,
             fmt='o', color='white', ecolor=exp_400s_color,
             markeredgecolor=exp_400s_color, elinewidth=ebarwidth)

# --- EXPTIME 1200 seconds
ax1.errorbar(cat['phase_true'], phase1200,
             yerr=phase1200err, markersize=4.5,
             fmt='o', color='white', ecolor=exp_1200s_color,
             markeredgecolor=exp_1200s_color, elinewidth=ebarwidth,
             label=r'$t_\mathrm{exp}\, =\ $' + all_exptimes[1])

ax2.errorbar(cat['phase_true'], phase1200 - cat['phase_true'],
             yerr=phase1200err, markersize=4.5,
             fmt='o', color='white', ecolor=exp_1200s_color,
             markeredgecolor=exp_1200s_color, elinewidth=ebarwidth)

# --- EXPTIME 3600 seconds
ax1.errorbar(cat['phase_true'], phase3600,
             yerr=phase3600err, markersize=2.0,
             fmt='o', color='white', ecolor=exp_3600s_color,
             markeredgecolor=exp_3600s_color, elinewidth=ebarwidth,
             label=r'$t_\mathrm{exp}\, =\ $' + all_exptimes[2])

ax2.errorbar(cat['phase_true'], phase3600 - cat['phase_true'],
             yerr=phase3600err, markersize=2.0,
             fmt='o', color='white', ecolor=exp_3600s_color,
             markeredgecolor=exp_3600s_color, elinewidth=ebarwidth)

# Ticks
ax1.set_xticklabels([])

# Limits
ax1.set_xlim(-10, 10)
ax1.set_ylim(-10, 10)

ax2.set_xlim(-10, 10)

# ax1.legend(loc=0, fontsize=18, frameon=False)
ax1.legend(loc='upper left', fontsize=18, frameon=False)

# Limit based on consideration of contam/uncontam sne
if not consider_contam_sne:
    ax2.set_ylim(-2, 2)
    fig.savefig(roman_slitless + 'figures/pylinearrecovery_phase_noblend.pdf',
                dpi=200, bbox_inches='tight')
else:
    fig.savefig(roman_slitless + 'figures/pylinearrecovery_phase_allsne.pdf',
                dpi=200, bbox_inches='tight')

fig.clear()
plt.close(fig)

###########################################
# -------------------- Av recovery plt
fig = plt.figure(figsize=(7, 9))

gs = fig.add_gridspec(nrows=14, ncols=1, left=0.05, right=0.95, wspace=0.1)

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
             markeredgecolor=exp_400s_color, ecolor=exp_400s_color,
             elinewidth=ebarwidth,
             label=r'$t_\mathrm{exp}\, =\ $' + all_exptimes[0])

ax2.errorbar(cat['Av_true'], av400 - cat['Av_true'],
             yerr=av400err, markersize=6.5, fmt='o',
             color='white',
             markeredgecolor=exp_400s_color, ecolor=exp_400s_color,
             elinewidth=ebarwidth)


# --- EXPTIME 1200 seconds
ax1.errorbar(cat['Av_true'], av1200,
             yerr=av1200err, markersize=4.5, fmt='o',
             color='white',
             markeredgecolor=exp_1200s_color, ecolor=exp_1200s_color,
             elinewidth=ebarwidth,
             label=r'$t_\mathrm{exp}\, =\ $' + all_exptimes[1])

ax2.errorbar(cat['Av_true'], av1200 - cat['Av_true'],
             yerr=av1200err, markersize=4.5, fmt='o',
             color='white',
             markeredgecolor=exp_1200s_color, ecolor=exp_1200s_color,
             elinewidth=ebarwidth)

# --- EXPTIME 3600 seconds
ax1.errorbar(cat['Av_true'], av3600,
             yerr=av3600err, markersize=2.0, fmt='o',
             color='white',
             markeredgecolor=exp_3600s_color, ecolor=exp_3600s_color,
             elinewidth=ebarwidth,
             label=r'$t_\mathrm{exp}\, =\ $' + all_exptimes[2])

ax2.errorbar(cat['Av_true'], av3600 - cat['Av_true'],
             yerr=av3600err, markersize=2.0, fmt='o',
             color='white',
             markeredgecolor=exp_3600s_color, ecolor=exp_3600s_color,
             elinewidth=ebarwidth)

# Ticks
ax1.set_xticklabels([])

# Limits
ax1.set_xlim(0.0, 3.0)
ax1.set_ylim(0.0, 3.0)

ax2.set_xlim(0.0, 3.0)

ax1.legend(loc=0, fontsize=18, frameon=False)

# Limit based on consideration of contam/uncontam sne
if not consider_contam_sne:
    ax2.set_ylim(-0.2, 0.2)
    fig.savefig(roman_slitless + 'figures/pylinearrecovery_dust_noblend.pdf',
                dpi=200, bbox_inches='tight')
else:
    fig.savefig(roman_slitless + 'figures/pylinearrecovery_dust_allsne.pdf',
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

ax1 = fig.add_subplot(gs[:4])
ax2 = fig.add_subplot(gs[4:8])
ax3 = fig.add_subplot(gs[8:])

# Axis labels
ax1.set_ylabel(r'$\frac{z_ - z_\mathrm{true}}{1 + z_\mathrm{true}}$',
               fontsize=15)
ax2.set_ylabel(r'$\Delta \mathrm{Phase}$', fontsize=20)
ax3.set_ylabel(r'$\Delta \mathrm{A_v}$', fontsize=20)
ax3.set_xlabel(r'$\mathrm{S/N}$', fontsize=20)

# Plotting
# z
ax1.axhline(y=0.0, ls='--', lw=2.0, color='gray', zorder=1)

ax1.scatter(cat['SNR400'], z400acc, s=23, color=exp_400s_color,
            facecolors='None', zorder=2)
ax1.scatter(cat['SNR1200'], z1200acc, s=15, color=exp_1200s_color,
            facecolors='None', zorder=2)
ax1.scatter(cat['SNR3600'], z3600acc, s=5, color=exp_3600s_color,
            facecolors='None', zorder=2, alpha=0.5)

# Phase
ax2.axhline(y=0.0, ls='--', lw=2.0, color='gray', zorder=1)

ax2.scatter(cat['SNR400'], phase400 - cat['phase_true'],
            s=23, color=exp_400s_color,
            facecolors='None', zorder=2)
ax2.scatter(cat['SNR1200'], phase1200 - cat['phase_true'],
            s=15, color=exp_1200s_color,
            facecolors='None', zorder=2)
ax2.scatter(cat['SNR3600'], phase3600 - cat['phase_true'],
            s=5, color=exp_3600s_color,
            facecolors='None', zorder=2, alpha=0.5)

# Av
ax3.axhline(y=0.0, ls='--', lw=2.0, color='gray', zorder=1)

ax3.scatter(cat['SNR400'], av400 - cat['Av_true'],
            s=23, color=exp_400s_color,
            facecolors='None', zorder=2)
ax3.scatter(cat['SNR1200'], av1200 - cat['Av_true'],
            s=15, color=exp_1200s_color,
            facecolors='None', zorder=2)
ax3.scatter(cat['SNR3600'], av3600 - cat['Av_true'],
            s=5, color=exp_3600s_color,
            facecolors='None', zorder=2, alpha=0.5)

# Limit based on consideration of contam/uncontam sne
if not consider_contam_sne:
    ax1.set_ylim(-0.002, 0.002)
    ax2.set_ylim(-1.5, 1.5)
    ax3.set_ylim(-0.15, 0.15)

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
