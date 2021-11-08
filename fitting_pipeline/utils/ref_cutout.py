import numpy as np
from astropy.io import fits

import subprocess
import os

from make_model_img import gen_model_img

def gen_reference_cutout(showref=False):

    img_basename = '5deg_'
    img_suffix = 'Y106_0_6'

    xloc = 2512
    yloc = 2268

    # NOTE: this is saved one level up from all the other sim images
    dir_img_name = roman_direct_dir + img_basename + img_suffix + '_model.fits'

    # Create model image if it doesn't exist
    if not os.path.isfile(dir_img_name):
        # ---------
        # Get the reference image first
        dname = img_sim_dir + img_basename + img_suffix + '.fits'
        ddat, dhdr = fits.getdata(dname, header=True)
        
        # Now scale image to get the image to counts per sec
        cps_sci_arr = ddat * dirimg_scaling

        # Save to be able to run sextractor to generate model image
        mhdu = fits.PrimaryHDU(data=cps_sci_arr, header=dhdr)
        model_img_name = dname.replace('.fits', '_scaled.fits')
        mhdu.writeto(model_img_name) 

        # ---------
        # Run SExtractor on this scaled image
        os.chdir(img_sim_dir)

        cat_filename = model_img_name.replace('.fits', '.cat')
        checkimage   = model_img_name.replace('.fits', '_segmap.fits')

        sextractor   = subprocess.run(['sex', model_img_name, 
            '-c', 'roman_sims_sextractor_config.txt', 
            '-CATALOG_NAME', os.path.basename(cat_filename), 
            '-CHECKIMAGE_NAME', checkimage], check=True)

        # Go back to roman-slitless directory
        os.chdir(roman_slitless_dir)

        # ---------
        # Now turn it into a model image
        # and add a small amount of background. See notes below on this.
        model_img = gen_model_img(model_img_name, checkimage)
        model_img += np.random.normal(loc=0.0, scale=back_scale, size=model_img.shape)

        # Save
        pref = fits.PrimaryHDU(data=model_img, header=dhdr)
        pref.writeto(dir_img_name)

    img_arr = fits.getdata(dir_img_name)

    r = yloc
    c = xloc

    s = 50

    ref_img = img_arr[r-s:r+s, c-s:c+s]

    # Save
    rhdu = fits.PrimaryHDU(data=ref_img)
    rhdu.writeto(img_sim_dir + 'ref_cutout_psf.fits', overwrite=True)

    if showref:

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(np.log10(ref_img), origin='lower')
        plt.show()

    return None
