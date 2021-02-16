import pylinear
from astropy.io import fits
import numpy as np

import os
import sys
import time
import datetime as dt
import glob
import shutil

# ---------------------- Preliminary stuff
print("\nThis code exists to compare speeds between versions of pylinear.")
print("See stuff in folder $HOME/Documents/roman_slitless_sims_results/few_sources_testrun/")
print("This code will only work on the laptop; NOT PLFFSN2 or MARCC.\n")

# Get starting time
start = time.time()
print("Starting at:", dt.datetime.now())

# Change directory to make sure results go in the right place
home = os.getenv('HOME')

testdir = home + '/Documents/roman_slitless_sims_results/few_sources_testrun/'
os.chdir(testdir)

# Define list files and other preliminary stuff
img_suffix = 'Y106_11_1'
segfile = testdir + 'akari_match_' + img_suffix + '_segmap_sourceclip.fits'

obslst = testdir + 'obs_fewsources.lst'
wcslst = testdir + 'wcs_fewsources.lst'
sedlst = testdir + 'sed_fewsources.lst'
beam = '+1'
maglim = 99.0

# make sure the files exist
assert os.path.isfile(segfile)
assert os.path.isfile(obslst)
assert os.path.isfile(sedlst)
assert os.path.isfile(wcslst)

print("Using the following paths to lst files and segmap:")
print("Segmentation map:", segfile)
print("OBS LST:", obslst)
print("SED LST:", sedlst)
print("WCS LST:", wcslst)

# ---------------------- Get sources
sources = pylinear.source.SourceCollection(segfile, obslst, detindex=0, maglim=maglim)

# Set up and tabulate
grisms = pylinear.grism.GrismCollection(wcslst, observed=False)
tabulate = pylinear.modules.Tabulate('pdt', ncpu=0) 
tabnames = tabulate.run(grisms, sources, beam)

## ---------------------- Simulate
print("Simulating...")
simulate = pylinear.modules.Simulate(sedlst, gzip=False, ncpu=0)
fltnames = simulate.run(grisms, sources, beam)
print("Simulation done.")

# ---------------------- Add noise
print("Adding noise...")
# check Russell's notes in pylinear notebooks
# also check WFIRST tech report TR1901
sky = 1.1      # e/s
npix = 4096 * 4096
sky /= npix    # e/s/pix

dark = 0.015   # e/s/pix
read = 10.0    # electrons

exptime = 900  # seconds

for oldf in glob.glob('*_flt.fits'):
    print("Working on...", oldf)
    print("Putting in an exposure time of:", exptime, "seconds.")

    # let's save the file in case we want to compare
    savefile = oldf.replace('_flt', '_flt_noiseless')
    shutil.copyfile(oldf, savefile)

    # open the fits file
    with fits.open(oldf) as hdul:
        sci = hdul[('SCI',1)].data    # the science image
        size = sci.shape              # dimensionality of the image

        # update the science extension with sky background and dark current
        signal = (sci + sky + dark)

        # Multiply the science image with the exptime
        # sci image originally in electrons/s
        signal = signal * exptime  # this is now in electrons

        # Randomly vary signal about its mean. Assuming Gaussian distribution
        # first get the uncertainty
        variance = signal + read**2
        sigma = np.sqrt(variance)
        new_sig = np.random.normal(loc=signal, scale=sigma, size=size)

        # now divide by the exptime and subtract the sky again 
        # to get back to e/s. LINEAR expects a background subtracted image
        final_sig = (new_sig / exptime) - sky

        # Assign updated sci image to the first [SCI] extension
        hdul[('SCI',1)].data = final_sig

        # update the uncertainty extension with the sigma
        err = np.sqrt(signal) / exptime

        hdul[('ERR',1)].data = err

        # now write
        hdul.writeto(oldf, overwrite=True)
        
    print("Written:", oldf)

print("Noise addition done. Check simulated images.")
ts = time.time()
print("Time taken for simulation:", "{:.2f}".format(ts - start), "seconds.")

# ---------------------- Extraction
fltlst = testdir + 'flt_fewsources.lst'
assert os.path.isfile(fltlst)
print("FLT LST:", fltlst)

grisms = pylinear.grism.GrismCollection(fltlst, observed=True)
path = home + '/Documents/roman_slitless_sims_results/few_sources_testrun/tables'

extraction_parameters = grisms.get_default_extraction()

print('\nDefault parameters: range = {lamb0}, {lamb1} A, sampling = {dlamb} A'.format(**extraction_parameters))

# Set extraction params
sources.update_extraction_parameters(**extraction_parameters)
method = 'golden'  # golden, grid, or single
root = 'romansim_fewsources_test_' + img_suffix + '_' + str(exptime) + 's'
logdamp = [-7, -1, 0.1]

print("Extracting...")
pylinear.modules.extract.extract1d(grisms, sources, beam, logdamp, method, root, path, ncpu=0, group=False)

print("Simulation and extraction done.")

te = time.time() - ts
print("Time taken for extraction:", "{:.2f}".format(te), "seconds.")
print("Total time taken:", "{:.2f}".format(time.time() - start), "seconds.")
print("Finished at:", dt.datetime.now())




