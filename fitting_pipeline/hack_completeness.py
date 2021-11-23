import numpy as np

import os
import sys

import matplotlib.pyplot as plt

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

def apply_classification_eff(comp_arr, mags, improved_classeff_fac):

    new_comp_arr = np.zeros(len(comp_arr))

    for count, c in enumerate(comp_arr):
        mag = mags[count]
        redshift = get_sn_z(mag)

        if redshift < 0.25:
            classeff = 1.0
        elif redshift > 2.25:
            classeff = 0.2
        else:
            z_idx = np.argmin(abs(classification_eff_redshift - redshift))
            classeff = classification_eff_base[z_idx] * improved_classeff_fac
            # The assumed improvement in classification efficiency
            # is due to the longer exptimes and therefore higher SNR

        if classeff > 1.0: classeff = 1.0

        new_comp_arr[count] = c * classeff

    return new_comp_arr

# ---------------------------- Completeness plot
# Create arrays for plotting
deltamag = 0.2
low_maglim = 18.6
high_maglim = 30.0

mag_bins = np.arange(low_maglim, high_maglim, deltamag)  # left edges of mag bins
mags = [(mag_bins[m] + mag_bins[m+1])/2 for m in range(len(mag_bins) - 1)]

classification_eff_base = [0.89, 0.97, 0.85, 0.84, 0.8, 0.65, 0.45, 0.25]
classification_eff_redshift = np.arange(0.375, 2.375, 0.25)

comp_20min = np.array([0.95, np.nan, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, \
    0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.89, 0.95, 0.95, \
    0.95, 0.9, 0.92, 0.95, 0.95, 0.95, 0.85, 0.78, 0.95, 0.9, 0.91, 0.87, \
    0.63, 0.87, 0.78, 0.7, 0.68, 0.5, 0.65, 0.215, 0.61, 0.23, 0.37, 0.15, \
    0.14, 0.04, 0, 0, 0, 0, 0, 0, 0, np.nan, np.nan, np.nan])

comp_1hr = np.array([0.95, np.nan, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, \
    0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, \
    0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, \
    0.92, 0.95, 0.95, 0.91, 0.95, 0.95, 0.9, 0.95, 0.85, 0.87, 0.92, \
    0.83, 0.84, 0.815, 0.27, 0, 0, 0, 0, 0, 0, 0, np.nan, np.nan, np.nan])

comp_3hr = np.array([0.95, np.nan, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, \
    0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, \
    0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, \
    0.92, 0.95, 0.95, 0.91, 0.95, 0.95, 0.9, 0.95, 0.95, 0.95, 0.92, \
    0.95, 0.95, 0.93, 0.375, 0, 0, 0, 0, 0, 0, 0, np.nan, np.nan, np.nan])

comp_9hr = np.array([0.95, np.nan, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, \
    0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, \
    0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, \
    0.92, 0.95, 0.95, 0.91, 0.95, 0.95, 0.9, 0.95, 0.95, 0.95, 0.95, \
    0.95, 0.95, 0.95, 0.375, 0, 0, 0, 0, 0, 0, 0, np.nan, np.nan, np.nan])

# Fix it because a blanket class eff was applied previously
comp_20min /= 0.95
comp_1hr /= 0.95
comp_3hr /= 0.95
comp_9hr /= 0.95

# First write the basic completeness values to file
with open('completeness_20m_1h.txt', 'w') as fh:

    fh.write('#  magF106  comp_20m  comp_1h' + '\n')

    for i in range(len(mags)):
        fh.write('{:.3f}'.format(mags[i]) + '  '
               + '{:.2f}'.format(comp_20min[i]) + '  '
               + '{:.2f}'.format(comp_1hr[i])
               + '\n')

sys.exit(0)

# Now redo with more accurate class eff
comp_20min = apply_classification_eff(comp_20min, mags, 1.0)
comp_1hr = apply_classification_eff(comp_1hr, mags, 1.5)
comp_3hr = apply_classification_eff(comp_3hr, mags, 2.0)
comp_9hr = apply_classification_eff(comp_9hr, mags, 2.5)

mag_counts = np.array([1, 0, 4, 7, 5, 8, 11, 6, 12, 16, 12, 10, \
    17, 15, 18, 18, 14, 15, 17, 23, 17, 20, 24, 25, 15, 20, 21, \
    23, 18, 20, 23, 26, 25, 27, 23, 27, 22, 23, 24, 22, 22, 27, \
    25, 24, 31, 29, 26, 37, 32, 25, 36, 17, 8, 0, 0, 0])

# Setup figure
fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111)

ax.plot(mags, comp_9hr, 'o-',  markersize=3, color='crimson', zorder=2)
ax.plot(mags, comp_3hr, 'o-',  markersize=3, color='dodgerblue', zorder=2)
ax.plot(mags, comp_1hr, 'o-',  markersize=3, color='seagreen', zorder=2)
ax.plot(mags, comp_20min, 'o-',  markersize=3, color='goldenrod', zorder=2)

axt = ax.twinx()
axt.bar(x=mags, height=mag_counts, width=0.2,
    color='gray', alpha=0.3, zorder=1)

ax.text(x=low_maglim-0.2, y=0.28, s=r'$\mbox{---}\ \frac{\Delta z}{1+z} \leq 0.01$', 
    verticalalignment='top', horizontalalignment='left', 
    transform=ax.transData, color='k', size=14)

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
mt = np.arange(low_maglim-0.5, high_maglim+0.5, 1.0)
mag_ax_diff = high_maglim+0.5 - (low_maglim-0.5)
mags_for_z_axis_transform = (mt - (low_maglim-0.5) )/mag_ax_diff
# the denominator here corresponds to the difference 
# on the bottom x-axis that is shown in the figure
# NOT the difference between final and init values in mags_for_z_axis
redshift_ticks = get_z_for_mag(mt)

ax2.set_xticks(mags_for_z_axis_transform)
ax2.set_xticklabels(['{:.2f}'.format(z) for z in redshift_ticks], rotation=30)
ax2.set_xlabel(r'$\mathrm{Redshift\ (assumed\ SN\ at\ peak)}$', fontsize=14)
ax2.minorticks_off()

# Limits
ax.set_xlim(low_maglim - 0.5, high_maglim + 0.5)
ax.set_ylim(-0.04, 1.04)

# labels
ax.set_ylabel(r'$\mathrm{Frac}.\ z\ \mathrm{completeness}$', fontsize=14)
ax.set_xlabel(r'$m_{F106}$', fontsize=14)
axt.set_ylabel(r'$\#\ \mathrm{objects}$', fontsize=14)

# save
fig.savefig(roman_slitless + 'figures/pylinearrecovery_completeness.pdf', 
            dpi=200, bbox_inches='tight')


