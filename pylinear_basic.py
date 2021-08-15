print("A copy of this code exists in the roman-slitless repo.")
print("However, it is to be run from the pylinear_basic_test folder.")
print("Make sure both copies are the same.\n")

import numpy as np
from astropy.io import fits
import pylinear

import os
import sys
import glob
import shutil

import matplotlib.pyplot as plt

home = os.getenv('HOME')
basic_testdir = '/Volumes/Joshi_external_HDD/Roman/roman_direct_sims/pylinear_basic_test/'

"""
To do this entire test for a small number of sources
just make sure to change the names accordingly in here
and in the lst files. Use the following function to include
as many sources as needed.

cd to pylinear_basic_test folder and run the 
create_reqs_for_smalltest function

# Move segmap and all lst files to the small_num_sources_test folder.

# To run the script for a small number of sources
# Change the following in here to the correct paths and names:
# 1. basic_testdir = '/Volumes/Joshi_external_HDD/Roman/roman_direct_sims/pylinear_basic_test/small_num_sources_test'
# 2. segfile = basic_testdir + '5deg_Y106_0_1_cps_segmap_small.fits'
# 3. sedlst = basic_testdir + 'sed_small.lst'
# 4. obs, wcs, and flt lists can stay the same.

# Also make sure that the contents of the lst files are consistent
"""

def create_reqs_for_smalltest(sedlst, num_sources=100, num_sne=10, get_sne_sedlst=False):

    # --------------- Galaxies to be included
    chosen_segids = np.random.randint(low=1, high=len(sedlst), size=num_sources)

    # --------------- loop and find all SN segids
    if get_sne_sedlst:
        all_sn_segids = []
        for i in range(len(sedlst)):
            if 'salt' in sedlst['sed_path'][i]:
                all_sn_segids.append(sedlst['segid'][i])
        print('ALL SN segids in this file:', all_sn_segids)
    else:
        all_sn_segids = np.random.randint(low=1, high=len(sedlst), size=num_sne)

    chosen_segids = np.append(chosen_segids, all_sn_segids)  # SNe in the chosen image
    chosen_segids = np.unique(chosen_segids)

    print('Will only keep the following segmap ids in the sim:')
    print(chosen_segids)

    # --------------- Now edit the segmap to only include 
    # the sources chosen above.
    segmap = fits.open('5deg_Y106_0_1_segmap.fits')
    new_segmap = np.zeros(segmap[0].data.shape)

    for i in range(1,len(sedlst)):
        if i in chosen_segids:
            print('Adding SegID:', i)
            idx = np.where(segmap[0].data == i)
            new_segmap[idx] += segmap[0].data[idx]

    hnew = fits.PrimaryHDU(header=segmap[0].header, data=new_segmap)
    hnew.writeto('5deg_Y106_0_1_segmap_small.fits', overwrite=True)

    print('Segmap edited and saved.')

    # --------------- Edit SED LST
    # Also make sure that only the chosen segids remain in sed lst
    # This simply prints the sed lst format text to the terminal
    # which can be copy pasted in a new file.
    with open('sed.lst','w') as fh:
        all_segids = sedlst['segid']
        for i in all_segids:
            if i in chosen_segids:
                idx = int(np.where(all_segids == i)[0])
                fh.write(str(i) + '  ' + sedlst['sed_path'][idx] + '\n')
                print(str(i) + '  ' + sedlst['sed_path'][idx])

    return None

# Comment these three lines out once create_reqs_for_smalltest() is run
#sedlst = np.genfromtxt('sed_Y106_0_1.lst', dtype=None, names=['segid','sed_path'], encoding='ascii')
#create_reqs_for_smalltest(sedlst)
#sys.exit(0)

# This function isn't required anymore
def create_sedlst():
    
    sys.path.append('/Users/baj/Documents/GitHub/roman-slitless/')
    
    from gen_sed_lst import get_sn_spec_path, get_gal_spec_path

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

    return None

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
segfile = basic_testdir + '5deg_Y106_0_1_segmap_small.fits'
# Make sure to rename segmap and cat accordingly afterwards

#create_sedlst()
#sys.exit(0)

# --------
# ALL lst files created manually in sublime
# except sedlst see above
# wcs coords copy pasted from header of image file
obslst = basic_testdir + 'obs.lst'
sedlst = basic_testdir + 'sed.lst'
wcslst = basic_testdir + 'wcs.lst'
fltlst = basic_testdir + 'flt.lst'

beam = '+1'
maglim = 30.0

# ------- Other preliminaries
exptime = 6000 # s

sky  = 1.1     # e/s
npix = 4096 * 4096
#sky /= npix    # e/s/pix
    
dark = 0.015   # e/s/pix
read = 10.0    # electrons
read /= npix

#readtime = 600 # s
#nreads = int(exptime / readtime)
#print('NREADS:', nreads)
#readeff = read #* nreads  # effective read noise i.e., total electrons from read noise
#readeff /= npix

simroot = 'romansim_prism'

# -------- Generate dispersed images
sources = pylinear.source.SourceCollection(segfile,obslst,detindex=0,maglim=maglim)

grisms = pylinear.grism.GrismCollection(wcslst, observed=False)
tabulate = pylinear.modules.Tabulate('pdt', ncpu=0)
tabnames = tabulate.run(grisms, sources, beam)

simulate = pylinear.modules.Simulate(sedlst, gzip=False, ncpu=0)
fltnames = simulate.run(grisms, sources, beam)
print('Simulation done.')

do_back_est()

# -------- Add noise according to exptime
for fl in glob.glob(basic_testdir + simroot + '*flt.fits'):
    print('Adding noise to:', os.path.basename(fl))

    with fits.open(fl) as hdul:

        # get basic sci image
        sci = hdul[('SCI',1)].data    # the science image
        size = sci.shape              # dimensionality of the image

        nan_idx = np.where(np.isnan(sci))
        nan_idx = np.asarray(nan_idx)
        if nan_idx.size:
            print("* * * * * Found NaNs in SCI. Resolve this issue first. Exiting.")
            sys.exit(1)

        # update signal with sky and dark
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
    
        #fig = plt.figure(figsize=(8,8))
        #ax = fig.add_subplot(111)
        #print(np.min(final_sig), np.max(final_sig))
        #ax.imshow(final_sig, origin='lower', vmin=0.0, vmax=0.5)
        #plt.show()
        #sys.exit(0)

        # Assign updated sci image to the first [SCI] extension
        hdul[('SCI',1)].data = final_sig
    
        # update the uncertainty extension with the sigma
        err = np.sqrt(signal) / exptime
    
        hdul[('ERR',1)].data = err

        # check for nan
        nan_idx = np.where(np.isnan(final_sig))
        nan_idx = np.asarray(nan_idx)
        if nan_idx.size:
            print("* * * * * Found NaNs in FINAL SCI. Resolve this issue first. Exiting.")
            sys.exit(1)

        # If all is good
        # first save a copy
        shutil.copy(fl, fl.replace('.fits', '_noiseless.fits'))
        # now write
        # We are overwriting here to avoid redoing the tables in the extraction
        # since the flt names will be the same.
        hdul.writeto(fl, overwrite=True)
        print('Noised sim saved for:', os.path.basename(fl))

# -------- Extraction
grisms = pylinear.grism.GrismCollection(fltlst, observed=True)
#tabulate = pylinear.modules.Tabulate('pdt', ncpu=0)
#tabnames = tabulate.run(grisms, sources, beam)

extraction_parameters = grisms.get_default_extraction()

extraction_parameters['dlamb'] = 30.0

extpar_fmt = 'Default parameters: range = {lamb0}, {lamb1} A, sampling = {dlamb} A'
print(extpar_fmt.format(**extraction_parameters))
    
# Set extraction params
sources.update_extraction_parameters(**extraction_parameters)
method = 'golden'  # golden, grid, or single
extroot = simroot + '_basic_test'
logdamp = [-6, -1, 0.1]

pylinear.modules.extract.extract1d(grisms, sources, beam, logdamp, 
    method, extroot, path='tables/',
    inverter='lsqr', ncpu=1, group=False)










