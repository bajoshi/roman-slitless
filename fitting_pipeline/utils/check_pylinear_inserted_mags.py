import numpy as np
import matplotlib.pyplot as plt

import socket

# This code will only be run when testing on the laptop
# NOT on PLFFSN2. It requires that pylinear print mags
# to the terminal and the user copy-paste the output into
# a file. You have to run run_pylinear.py and load the sources
# so that pylinear (from sourcecollection.py) can print to the screen.
img_sim_dir = '/Volumes/Joshi_external_HDD/Roman/roman_direct_sims/sims2021/K_5degimages_part1/'
roman_slitless_dir = '/Users/baj/Documents/GitHub/roman-slitless/'

img_suffix = 'Y106_0_1'

pylinear_mag_file = img_sim_dir + '5deg_' + img_suffix + '_pylinearmags.txt'
cat_filename = img_sim_dir + '5deg_' + img_suffix + '_SNadded.cat'

cat_header = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'ALPHA_J2000', 'DELTA_J2000', 
'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_AUTO', 'MAGERR_AUTO', 'FLUX_RADIUS', 'FWHM_IMAGE']


pylinear_mcat = np.genfromtxt(pylinear_mag_file, delimiter=',', dtype=None, names=['segid', 'pylinear_mag'], encoding='ascii')
cat = np.genfromtxt(cat_filename, dtype=None, names=cat_header, encoding='ascii')

# ------------------------------------

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xlabel('SExtractor MAG-AUTO', fontsize=14)
ax.set_ylabel('pyLINEAR inferred MAG', fontsize=14)

ax.scatter(cat['MAG_AUTO'], pylinear_mcat['pylinear_mag'], s=5, color='k', facecolors=None)
ax.plot(np.arange(18.5, 30.0, 0.01), np.arange(18.5, 30.0, 0.01), color='r', ls='--')

ax.set_xlim(19.0, 30.0)
ax.set_ylim(19.0, 30.0)

fig.savefig(roman_slitless_dir + 'figures/sextractor_pylinear_mag_compare.pdf', dpi=200, bbox_inches='tight')

