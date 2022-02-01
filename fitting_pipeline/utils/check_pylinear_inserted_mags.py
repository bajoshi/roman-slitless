import numpy as np
import matplotlib.pyplot as plt
import sys

# This code will only be run when testing on the laptop
# NOT on PLFFSN2. It requires that pylinear print mags
# to the terminal and the user copy-paste the output into
# a file. You have to run run_pylinear.py and load the sources
# so that pylinear (from sourcecollection.py) can print to the screen.
extdir = '/Volumes/Joshi_external_HDD/Roman/'
img_sim_dir = extdir + 'roman_direct_sims/sims2021/K_5degimages_part1/'
roman_slitless_dir = '/Users/baj/Documents/GitHub/roman-slitless/'
results_dir = extdir + 'roman_slitless_sims_results/'

img_suffix = 'Y106_0_1'

pylinear_mag_file = results_dir + 'pylinear_inferred_mags.txt'
cat_filename = img_sim_dir + '5deg_' + img_suffix + '_SNadded.npy'

pylinear_mcat = np.genfromtxt(pylinear_mag_file, dtype=None,
                              names=True, encoding='ascii')
cat = np.load(cat_filename)

all_insert_mags = np.array(cat[:, 2], dtype=np.float64)
all_insert_segids = np.array(cat[:, -1], dtype=int)

all_pylinear_mags = []

for i in range(len(all_insert_segids)):
    current_segid = all_insert_segids[i]
    segid_idx = int(np.where(pylinear_mcat['segid'] == current_segid)[0])
    m = pylinear_mcat['pylinear_mag'][segid_idx]
    all_pylinear_mags.append(m)

# ------------------------------------

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xlabel('SExtractor MAG-AUTO', fontsize=14)
ax.set_ylabel('pyLINEAR inferred MAG', fontsize=14)

ax.scatter(all_insert_mags, all_pylinear_mags, s=5,
           color='k', facecolors=None)
ax.plot(np.arange(18.5, 30.0, 0.01), np.arange(18.5, 30.0, 0.01),
        color='r', ls='--')

ax.set_xlim(21.5, 29.5)
ax.set_ylim(21.5, 29.5)

plt.show()

# fig.savefig(roman_slitless_dir +
#             'figures/sextractor_pylinear_mag_compare.pdf',
#             dpi=200, bbox_inches='tight')
sys.exit(0)
