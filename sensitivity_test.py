import numpy as np
from astropy.io import fits
import pylinear

# From Russell's pylinear notebooks
from astropy import wcs                # for the WCS of the output images
from astropy.modeling import models    # to create parametric models
from skimage.morphology import label   # to segment the image

import os
import sys
import glob
import shutil

import matplotlib.pyplot as plt

home = os.getenv('HOME')
pylinear_config_roman_dir = home + '/Documents/pylinear_ref_files/pylinear_config/Roman/'
pylinear_ref_dir = home + '/Documents/pylinear_ref_files/pylinear_config/Roman/'

from save_trans_curve_to_fits import save_thru_curve_to_fits

def gen_img_seg(crval, pixscl, size, segfile, imgfile):

    # create a grid of (x,y) pairs for the images
    x,y = np.meshgrid(np.arange(size[0]), np.arange(size[1]))

    # we only need a single source in this test
    # will use the reference star from Kevin's simulated 
    # images as our point source
    refhdu = fits.open('sensitivity_test/ref_cutout.fits')
    ref = refhdu[0].data

    # create the output image
    # Add some small background
    # Measured from reference cutout in ds9
    # Take a region not including the star and check its stats
    img = np.random.normal(loc=0.04, scale=0.01, size=size)
    
    # add the point source
    # at the center of the image
    img[450:550, 450:550] += ref

    # --------
    # Make sure the counts when multiplied by PHOTFLAM 
    # give the required AB mag of 23.0
    photflam = 3.0507e-20  # For WFC3/F105W
    lam_pivot = 10551.05  # angstroms
    speed_of_light_ang = 3e18

    # create a segmentation map from this image
    threshold = 50                       # threshold to apply to the image
    good = img > threshold                # these pixels belong to a source
    seg = label(good)                     # now these pixels have unique segmentation IDs

    counts = np.sum(img[seg==1], axis=None)
    print('\nTotal counts in img:', counts)
    flam = counts * photflam
    print('flam from photflam:', flam)
    fnu = flam * lam_pivot**2 / speed_of_light_ang
    print('fnu:', fnu)
    mag = -2.5 * np.log10(fnu) - 48.6
    print('AB mag:', mag)  # should be about 16th mag because that's the source we put in
    # will be a bit fainter than 16 because we have a high threshold
    # so not all pixels get correctly associated with the source
    # therefore summed counts are lower.

    # ------ Now do the scaling
    req_mag = 22.9  
    # Note on the req mag above: it is 
    # a little brighter than the actual req of 23.0 because again all pix 
    # will not be associated with the source and the resulting counts are lower
    req_fnu = 10**((req_mag + 48.6)/-2.5)
    print('\nRequired mag and fnu:', req_mag, req_fnu)
    req_flam = req_fnu * speed_of_light_ang / lam_pivot**2
    req_counts = req_flam / photflam
    print('Required counts:', req_counts)
    scaling_factor = req_counts / counts
    print('Scaling factor:', scaling_factor)

    img *= scaling_factor

    # ------ Redo the segmap for this fainter source
    threshold = 0.25
    good = img > threshold                # these pixels belong to a source
    seg = label(good)                     # now these pixels have unique segmentation IDs

    new_counts = np.sum(img[seg==1], axis=None)
    print('New counts:', new_counts)
    new_fnu = (new_counts * photflam) * lam_pivot**2 / speed_of_light_ang
    print('New AB mag:', -2.5 * np.log10(new_fnu) - 48.6)  # This is what we want to be close to 23.0

    # create a WCS for the image and segmentation
    w = wcs.WCS(naxis=2)                  # the WCS object
    w.wcs.crpix = [size[0]/2,size[1]/2]   # put the CRPIX at the center of the image
    w.wcs.crval = crval                   # set the RA,Dec of the center
    w.wcs.ctype = ['RA---TAN','DEC--TAN'] # use RA,Dec projection
    p = pixscl/3600.                      # change units from arcsec/pix to deg/pix
    w.wcs.cd = [[-p,0.],[0.,p]]           # set the CD matrix; the neg sign makes E-left
    
    # put the WCS into a fits header
    hdr = w.to_header()

    # write the images to disk
    fits.writeto(segfile,seg,header=hdr,overwrite=True) # write the segmap
    fits.writeto(imgfile,img,header=hdr,overwrite=True) # write the image

    return None

def save_salt2_test_template(pth, redshift=1):

    salt2_spec = np.genfromtxt("fitting_pipeline/utils/salt2_template_0.txt", 
        dtype=None, names=['day', 'lam', 'llam'], encoding='ascii')

    day0_idx = np.where(salt2_spec['day'] == 0)[0]

    day0_wav = salt2_spec['lam'][day0_idx] * (1+redshift)
    day0_llam = salt2_spec['llam'][day0_idx]

    # Save the spectrum
    dat = np.array([day0_wav, day0_llam])
    dat = dat.T
    np.savetxt(pth, dat, fmt=['%.1f', '%.5e'], header='lam  llam')

    return None

if __name__ == '__main__':
    # Check the 1 hour AB mag sensitivity limit for the prism here:
    # https://wfirst.ipac.caltech.edu/sims/Param_db.html#wfi_prism
    # We will simulate here a point source of the same magnitude 
    # and check what sensitivity level gets us the required SNR
    # of 10 sigma per pixel as stated above.

    # 1. Make sure to change the dlam in instruments.xml to 65A
    # 2. Reinstall pylinear

    # Follows the procedure given in the pylinear notebooks.
    savedir = '/Users/baj/Documents/GitHub/roman-slitless/sensitivity_test/'

    segfile = savedir + 'seg.fits'
    imgfile = savedir + 'img.fits'

    crval = [53.0,-27.0]
    pixscl = 0.1  # arcseconds per pixel
    size = (1000,1000)

    # Pylinear settings 
    maglim = 30.0
    roll_angles = [0.0, 5.0, 10.0]
    obslst = 'obs.lst'
    wcslst = 'wcs.lst'
    sedlst = 'sed.lst'
    fltlst = 'flt.lst'
    beam = '+1'

    exptime = 1200 # seconds 
    # We want a total time of 1 hour
    # Since we have three roll angles this is just 3600/3

    # -------- Generate image and segmap
    #gen_img_seg(crval, pixscl, size, segfile, imgfile)

    # -------- Read in SN Ia spectrum
    sed_path = '/Users/baj/Documents/GitHub/roman-slitless/sensitivity_test/salt2_day0.txt'
    #save_salt2_test_template(sed_path)

    # -------- Create LST files
    with open('sensitivity_test/obs.lst', 'w') as fo:
        print('img.fits    hst_wfc3_f105w', file=fo)

    with open('sensitivity_test/wcs.lst', 'w') as fw:
        # obligatory header info:
        print('# TELESCOPE = Roman',file=fw)   # specify the telescope
        print('# INSTRUMENT = WFI',file=fw) # specify the instrument
        print('# DETECTOR = WFI',file=fw)     # specify the detector 
        print('# GRISM = P127',file=fw)  # either roman prism or grism
        print('# BLOCKING = ',file=fw)       # specify the blocking filter (only for JWST)

        for r in range(len(roll_angles)):
            print('senstest' + str(r+1) + '  ' + \
                '{:.6f}'.format(crval[0]) + '  ' + '{:.6f}'.format(crval[1]) + \
                    '  ' + '{:.1f}'.format(roll_angles[r]) + '  P127', file=fw)

    with open('sensitivity_test/sed.lst', 'w') as fs:
        print('1  ' + sed_path, file=fs)

    # --------
    sens_arr = 1e16 * np.arange(1, 8)
    print('\nWill test the following sensitivities:', sens_arr)

    # -------- Run the test for the above set of sensitivities
    rt = np.genfromtxt(pylinear_config_roman_dir + 'roman_throughput_20190325.txt', \
    dtype=None, names=True, skip_header=3)

    for senslimit in sens_arr:

        # First save the modified sensitivity curve
        wp = rt['Wave'] * 1e4  # convert to angstroms from microns
        tp = rt['SNPrism'] * senslimit 

        save_thru_curve_to_fits(wp, tp, np.zeros(len(tp)), 'p127', pylinear_ref_dir)

        # Set up pylinear and run
        os.chdir(savedir)

        # Load sources
        sources = pylinear.source.SourceCollection(segfile, obslst, detindex=0, maglim=maglim)

        # Load grisms
        grisms = pylinear.grism.GrismCollection(wcslst, observed=False)

        # Make tables
        tabulate = pylinear.modules.Tabulate('pdt', ncpu=3)
        tabnames = tabulate.run(grisms, sources, beam)

        # Simulate
        simulate = pylinear.modules.Simulate(sedlst, gzip=False)
        fltnames = simulate.run(grisms, sources, beam)



        sys.exit(0)












