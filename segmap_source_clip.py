import numpy as np
from astropy.io import fits

import matplotlib.pyplot as plt

# Define sources you want to keep in the segmap
sources = [622, 629, 630]

# Read in original segmap
fname = '/Users/baj/Documents/roman_slitless_sims_results/two_source_testrun/akari_match_Y106_11_1_segmap.fits'
h = fits.open(fname)

# Find indices for sources that you want
idx = []
for s in sources:
    idx.append(np.where(h[0].data == s))

# Now create a mask and invert it
# to mask out all the sources that aren't required
mask = np.ones(h[0].data.shape, dtype=bool)

for i in idx:
    mask[i] = False

h[0].data[mask] = 0.0

# Save to new file
h.writeto(fname.replace('.fits', '_sourceclip.fits'), overwrite=True)

print("Done. Check results in ds9.")
