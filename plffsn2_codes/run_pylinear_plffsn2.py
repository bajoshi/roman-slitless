import pylinear
from astropy.io import fits
import numpy as np

import os
import sys
import time
import datetime as dt
import glob
import shutil

start = time.time()
print("Starting at:", dt.datetime.now())

home = os.getenv('HOME')

os.chdir(home + '/Documents/roman_slitless_sims_results/')

img_suffix = 'Y106_11_1'

# Define list files and other preliminary stuff
segfile = home + '/Documents/roman_direct_sims/K_akari_rotate_subset/akari_match_' + img_suffix + '_segmap.fits'
obslst = home + '/Documents/GitHub/roman-slitless/plffsn2_codes/obs_plffsn2_' + img_suffix + '.lst'
wcslst = home + '/Documents/GitHub/roman-slitless/plffsn2_codes/wcs_plffsn2_' + img_suffix + '.lst'
sedlst = home + '/Documents/GitHub/roman-slitless/plffsn2_codes/sed_plffsn2_' + img_suffix + '.lst'
beam = '+1'
maglim = 99.0
seddir = 'SEDs_' + img_suffix

fltlst = home + '/Documents/GitHub/roman-slitless/plffsn2_codes/flt_plffsn2_' + img_suffix + '.lst'

assert os.path.isfile(segfile)
assert os.path.isfile(obslst)
assert os.path.isfile(sedlst)
assert os.path.isfile(wcslst)
assert os.path.isfile(fltlst)

# Get sources
sources = pylinear.source.SourceCollection(segfile, obslst, detindex=0, maglim=maglim)

# Set up and tabulate
grisms = pylinear.grism.GrismCollection(wcslst, observed=False)
tabulate = pylinear.modules.Tabulate('pdt', ncpu=0)
tabnames = tabulate.run(grisms, sources, beam)

# Simulate
print("Simulating...")
simulate = pylinear.modules.Simulate(sedlst, gzip=False)
fltnames = simulate.run(grisms, sources, beam)
print("Simulation done.")

# ---------- Add noise
print("Adding noise...")
sig = 0.1    # noise RMS in e-/s (check Russell's notes in pylinear notebooks)

for oldf in glob.glob('*_flt.fits'):
    print("Working on...", oldf)

    # let's save the file in case we want to compare
    savefile = oldf.replace('_flt', '_flt_noiseless')
    shutil.copyfile(oldf, savefile)

    # open the fits file
    with fits.open(oldf) as hdul:
        sci = hdul[('SCI',1)].data    # the science image
        size = sci.shape              # dimensionality of the image

        # update the science extension with random noise
        hdul[('SCI',1)].data = sci + np.random.normal(loc=0., scale=sig, size=size)

        # update the uncertainty extension with the sigma
        hdul[('ERR',1)].data = np.full_like(sci, sig)

        # now write to a new file name
        hdul.writeto(oldf, overwrite=True)

print("Noise addition done. Check simulated images.")

# Extraction
grisms = pylinear.grism.GrismCollection(fltlst, observed=True)
path = home + '/Documents/roman_slitless_sims_results/tables'
tabulate = pylinear.modules.Tabulate('pdt', path=path, ncpu=0)
tabnames = tabulate.run(grisms, sources, beam)

extraction_parameters = grisms.get_default_extraction()

print('Default parameters: range = {},{} A, sampling = {} A'.format(**extraction_parameters))

# Set extraction params
sources.update_extraction_parameters(**extraction_parameters)
method = 'golden'
root = 'romansim2_ext'
logdamp = [-8, 1, 0.1]

print("Extracting...")
pylinear.modules.extract.extract1d(grisms, sources, beam, logdamp, method, root, path, group=False)

print("Simulation and extraction done.")
print("Total time taken:", "{:.2f}".format(time.time() - start), "seconds.")

sys.exit(0)
