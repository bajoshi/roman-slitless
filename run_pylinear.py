import pylinear

import os
import sys
import time
import datetime as dt

start = time.time()
print("Starting at:", dt.datetime.now())

home = os.getenv('HOME')

os.chdir(home + '/Documents/roman_slitless_sims_results/')

img_suffix = 'Y106_11_1'

# Define list files and other preliminary stuff
segfile = home + '/Documents/roman_direct_sims/K_akari_rotate_subset/akari_match_' + img_suffix + '_segmap.fits'
obslst = home + '/Documents/GitHub/roman-slitless/obs_' + img_suffix + '.lst'
wcslst = home + '/Documents/GitHub/roman-slitless/wcs_' + img_suffix + '.lst'
sedlst = home + '/Documents/GitHub/roman-slitless/sed_' + img_suffix + '.lst'
beam = '+1'
maglim = 99.0
seddir = 'SEDs_' + img_suffix

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

# Extraction
fltlst = home + '/Documents/GitHub/roman-slitless/flt_' + img_suffix + '.lst'
grisms = pylinear.grism.GrismCollection(fltlst, observed=True)
path = home + '/Documents/roman_slitless_sims_results/tables'
tabulate = pylinear.modules.Tabulate('pdt', path=path, ncpu=0)
tabnames = tabulate.run(grisms, sources, beam)

extraction_parameters = grisms.get_default_extraction()

print('Default parameters: range = {},{} A, sampling = {} A'.format(*extraction_parameters))

# Set extraction params
sources.update_extraction_parameters(*extraction_parameters)
method = 'grid'
root = 'romansim1_ext'
logdamp = [-4, -1, 0.1]

print("Extracting...")
pylinear.modules.extract.extract1d(grisms, sources, beam, logdamp, method, root, path, group=False)

print("Simulation and extraction done.")
print("Total time taken:", "{:.2f}".format(time.time() - start), "seconds.")

sys.exit(0)