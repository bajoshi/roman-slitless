import pylinear
from astropy.io import fits
import numpy as np

import os
import sys
import time
import datetime as dt
import glob
import shutil
import socket

# ---------------------- Preliminary stuff
# Get starting time
start = time.time()
print("Starting at:", dt.datetime.now())

# Change directory to make sure results go in the right place
home = os.getenv('HOME')
os.chdir(home + '/Documents/roman_slitless_sims_results/')

# Define directories for imaging and lst files
pylinear_lst_dir = home + '/Documents/GitHub/roman-slitless/pylinear_lst_files/'
direct_img_dir = home + '/Documents/roman_direct_sims/K_akari_rotate_subset/'

# Figure out the correct filenames depending on which machine is being used
img_suffix = 'Y106_11_1'

# Define list files and other preliminary stuff
segfile = direct_img_dir + 'akari_match_' + img_suffix + '_segmap.fits'

# changing img_suffix can only be done after segfile is defined 
# because of the way segfile is named. Needs to be cleaned up.
hostname = socket.gethostname()
if 'plffsn2' in hostname:
    img_suffix = 'plffsn2_' + img_suffix

obslst = pylinear_lst_dir + 'obs_' + img_suffix + '.lst'
wcslst = pylinear_lst_dir + 'wcs_' + img_suffix + '.lst'
sedlst = pylinear_lst_dir + 'sed_' + img_suffix + '.lst'
beam = '+1'
maglim = 99.0

fltlst = pylinear_lst_dir + 'flt_' + img_suffix + '.lst'

# make sure the files exist
assert os.path.isfile(segfile)
assert os.path.isfile(obslst)
assert os.path.isfile(sedlst)
assert os.path.isfile(wcslst)
assert os.path.isfile(fltlst)

# ---------------------- Get sources
sources = pylinear.source.SourceCollection(segfile, obslst, detindex=0, maglim=maglim)

# Set up and tabulate
grisms = pylinear.grism.GrismCollection(wcslst, observed=False)
tabulate = pylinear.modules.Tabulate('pdt', ncpu=0) 
tabnames = tabulate.run(grisms, sources, beam)

# ---------------------- Simulate
print("Simulating...")
simulate = pylinear.modules.Simulate(sedlst, gzip=False)
fltnames = simulate.run(grisms, sources, beam)
print("Simulation done.")

# ---------------------- Add noise
print("Adding noise...")
# check Russell's notes in pylinear notebooks
# also check WFIRST tech report TR1901
sig = 0.001    # noise RMS in e-/s 

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

# ---------------------- Extraction
grisms = pylinear.grism.GrismCollection(fltlst, observed=True)
path = home + '/Documents/roman_slitless_sims_results/tables'
#tabulate = pylinear.modules.Tabulate('pdt', path=path, ncpu=0)
#tabnames = tabulate.run(grisms, sources, beam)

extraction_parameters = grisms.get_default_extraction()

print('\nDefault parameters: range = {lamb0}, {lamb1} A, sampling = {dlamb} A'.format(**extraction_parameters))

# Set extraction params
sources.update_extraction_parameters(**extraction_parameters)
method = 'grid'  # single
root = 'romansim1_ext'
logdamp = [-7, -1, 0.1]  # logdamp = -np.inf

print("Extracting...")
pylinear.modules.extract.extract1d(grisms, sources, beam, logdamp, method, root, path, group=False)

print("Simulation and extraction done.")
print("Total time taken:", "{:.2f}".format(time.time() - start), "seconds.")

sys.exit(0)




