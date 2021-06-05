print("A copy of this code exists in the roman-slitless repo.")
print("However, it is to be run from the pylinear_basic_test folder.")
print("Make sure both copies are the same.\n")

"""
To do this entire test for a small number of sources
just make sure to change the names accordingly in here
and in the lst files. Use the following script to include
as many sources as needed.

cd to pylinear_basic_test folder
>>> ipython
import numpy as np
import os, sys
from astropy.io import fits 

num_sources = 10
chosen_segids = np.random.randint(low=1, high=1005, size=num_sources)

segmap = fits.open('5deg_Y106_0_1_cps_segmap.fits')
new_segmap = np.zeros(segmap[0].data.shape)

for i in range(1,1006):  # check this range by eye in the catalog
    if i in chosen_segids:
        print('Adding SegID:', i)
        idx = np.where(segmap[0].data == i)
        new_segmap[idx] += segmap[0].data[idx]

hnew = fits.PrimaryHDU(header=segmap[0].header, data=new_segmap)
hnew.writeto('5deg_Y106_0_1_cps_segmap_small.fits')

# Also make sure that only the chosen segids remain in sed.lst
sedlst = np.genfromtxt('sed.lst', dtype=None, names=['segid','path'], skip_header=2, encoding='ascii')

with open('sed_small.lst','w') as fh:
    all_segids = sedlst['segid']
    for i in all_segids:
        if i in chosen_segids:
            idx = int(np.where(all_segids == i)[0])
            fh.write(str(i) + '  ' + sedlst['path'][idx] + '\n')
            print(str(i) + '  ' + sedlst['path'][idx])

# Move segmap and all lst files to the small_num_sources_test folder.

# To run the script for a small number of sources
# Change the following in here to the correct paths and names:
# 1. basic_testdir = '/Volumes/Joshi_external_HDD/Roman/roman_direct_sims/pylinear_basic_test/small_num_sources_test'
# 2. segfile = basic_testdir + '5deg_Y106_0_1_cps_segmap_small.fits'
# 3. sedlst = basic_testdir + 'sed_small.lst'
# 4. obs, wcs, and flt lists can stay the same.

# Also make sure that the contents of the lst files are consistent

"""


import numpy as np
from astropy.io import fits
import pylinear

import os
import sys

home = os.getenv('HOME')
basic_testdir = '/Volumes/Joshi_external_HDD/Roman/roman_direct_sims/pylinear_basic_test/small_num_sources_test/'

# Make sure to use the counts-per-second image
"""
cd to pylinear_basic_test folder
>>> ipython
from astropy.io import fits
h = fits.open('5deg_Y106_0_1.fits')
hnew_data = h[1].data
hnew_data /= float(h[1].header['EXPTIME'])
hnew = fits.PrimaryHDU(header=h[1].header, data=hnew_data)
hnew.writeto('5deg_Y106_0_1_cps.fits')
"""

# Run sextractor through the command line manually
# e.g., sex 5deg_Y106_0_1_cps.fits -c roman_sims_sextractor_config.txt 
segfile = basic_testdir + '5deg_Y106_0_1_cps_segmap_small.fits'
# Make sure to rename segmap and cat accordingly afterwards

# ALL lst files created manually in sublime
# sedlst created by  --
"""
cd to pylinear_basic_test folder
>>> ipython
import sys
import os
sys.path.append('/Users/baj/Documents/GitHub/roman-slitless/')
from gen_sed_lst import get_sn_spec_path, get_gal_spec_path
import numpy as np
cat_header = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'ALPHA_J2000', 'DELTA_J2000', \
'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_AUTO', 'MAGERR_AUTO', 'FLUX_RADIUS', 'FWHM_IMAGE']
cat = np.genfromtxt('/Volumes/Joshi_external_HDD/Roman/roman_direct_sims/pylinear_basic_test/5deg_Y106_0_1_cps.cat', \
dtype=None, names=cat_header, encoding='ascii')
random_sne = np.random.randint(low=1, high=1005, size=20)
with open('sed.lst','w') as fh:
    fh.write('# 1: SEGMENTATION ID' + '\n')
    fh.write('# 2: SED FILE' + '\n')
    for segid in range(1,1006):
        redshift = np.random.uniform(low=0.1, high=3.0)
        if segid in random_sne:
            pth = get_sn_spec_path(redshift)
        else:
            pth = get_gal_spec_path(redshift)
        fh.write(str(segid) + "  " + pth + "\n")
        print(segid, '  ', '{:.3f}'.format(redshift), '  ', os.path.basename(pth))
"""
# wcs coords copy pasted from header of image file
obslst = basic_testdir + 'obs.lst'
sedlst = basic_testdir + 'sed_small.lst'
wcslst = basic_testdir + 'wcs.lst'
fltlst = basic_testdir + 'flt.lst'

beam = '+1'
maglim = 99.0

# -------- Generate dispersed images
sources = pylinear.source.SourceCollection(segfile,obslst,detindex=0,maglim=maglim)

grisms = pylinear.grism.GrismCollection(wcslst, observed=False)
tabulate = pylinear.modules.Tabulate('pdt', ncpu=0)
tabnames = tabulate.run(grisms, sources, beam)

simulate = pylinear.modules.Simulate(sedlst, gzip=False, ncpu=0)
fltnames = simulate.run(grisms, sources, beam)

# -------- Extraction
grisms = pylinear.grism.GrismCollection(fltlst, observed=True)    
extraction_parameters = grisms.get_default_extraction()
    
extpar_fmt = 'Default parameters: range = {lamb0}, {lamb1} A, sampling = {dlamb} A'
print(extpar_fmt.format(**extraction_parameters))
    
# Set extraction params
sources.update_extraction_parameters(**extraction_parameters)
method = 'golden'  # golden, grid, or single
extroot = 'romansim_grism_basic_test'
logdamp = [-4, -1, 0.1]

pylinear.modules.extract.extract1d(grisms, sources, beam, logdamp, 
    method, extroot, path='tables/',
    inverter='lsqr', ncpu=0, group=False)










