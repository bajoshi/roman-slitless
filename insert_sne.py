import numpy as np
from astropy.io import fits

import os
import sys
import socket
import subprocess
from tqdm import tqdm

import matplotlib.pyplot as plt

# Get the dirs correct
if 'plffsn2' in socket.gethostname():
    extdir = '/astro/ffsn/Joshi/'
    modeldir = extdir + 'bc03_output_dir/'
    roman_direct_dir = extdir + 'roman_direct_sims/sims2021/'

    roman_slitless_dir = extdir + 'GitHub/roman-slitless/'
    utils_dir = roman_slitless_dir + 'fitting_pipeline/utils/'

else:
    extdir = '/Volumes/Joshi_external_HDD/Roman/'
    modeldir = extdir + 'bc03_output_dir/m62/'
    roman_direct_dir = extdir + 'roman_direct_sims/sims2021/'

    home = os.getenv('HOME')
    roman_slitless_dir = home + '/Documents/GitHub/roman-slitless/'
    utils_dir = roman_slitless_dir + 'fitting_pipeline/utils/'

assert os.path.isdir(modeldir)
assert os.path.isdir(roman_direct_dir)

dir_img_part = 'part1'
img_sim_dir = roman_direct_dir + 'K_5degimages_' + dir_img_part + '/'

sys.path.append(utils_dir)
from make_model_dirimg import gen_model_img

# ---------------------- GLOBAL DEFS
# Scaling factor for direct images
# The difference here that appears in the power of 10
# is the difference between the ZP of the current img
# and what I think the correct ZP is i.e., the WFC3/F105W ZP.
dirimg_scaling = 10**(-0.4 * (31.7956 - 26.264))
# ----------------------
back_scale = 0.001  # standard deviation for background to be added.
# ----------------------


def get_insertion_coords(num_to_insert, img_cat=None):

    x_ins = np.zeros(num_to_insert)
    y_ins = np.zeros(num_to_insert)

    x_ins = np.random.randint(low=110, high=3985, size=num_to_insert)
    y_ins = np.random.randint(low=110, high=3985, size=num_to_insert)

    return x_ins, y_ins

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

def main():

    # ---------------
    # some preliminary settings
    img_basename = '5deg_'
    ref_mag = 15.9180
    ref_counts = 13753.24  # read in mag and counts from SExtractor catalog on dir img
    ref_segid = 630
    s = 50  # same as the size of the cutout stamp  # cutout is 100x100; need half that here
    verbose = False

    # Mag limits for choosing random SN mag
    lowmag = 19.0
    highmag = 30.0

    # ---------------
    # Read in the reference image of the star from 
    # Y106_0_6 at 72.1175280, -53.5739388 (~16th mag)
    ref_cutout_path = img_sim_dir + 'ref_cutout_psf.fits'
    if not os.path.isfile(ref_cutout_path):
        gen_reference_cutout()

    ref_data = fits.getdata(ref_cutout_path)

    # ---------------
    # Arrays to loop over
    pointings = np.arange(0, 1)
    detectors = np.arange(1, 19, 1)

    for pt in tqdm(pointings, desc="Pointing"):
        for det in tqdm(detectors, desc="Detector", leave=False):

            # ---------------
            # Determine number of SNe to insert and open dir img
            num_to_insert = np.random.randint(low=200, high=300)

            img_suffix = 'Y106_' + str(pt) + '_' + str(det)
            dir_img_name = img_sim_dir + img_basename + img_suffix + '.fits'

            tqdm.write("Working on: " + dir_img_name)
            tqdm.write("Will insert " + str(num_to_insert) + " SNe in " + os.path.basename(dir_img_name))

            # First check that the files have been unzipped
            if not os.path.isfile(dir_img_name):
                tqdm.write("Unzipping file: " + dir_img_name + ".gz")
                subprocess.run(['gzip', '-fd', dir_img_name + '.gz'])

            # Open dir image
            dir_hdu = fits.open(dir_img_name)

            # ---------------
            # Now scale image to get the image to counts per sec
            cps_sci_arr = dir_hdu[1].data * dirimg_scaling
            cps_hdr = dir_hdu[1].header
            dir_hdu.close()
            #cps_hdr['BUNIT'] = 'ELECTRONS'
            # Save to be able to run sextractor to generate model image
            mhdu = fits.PrimaryHDU(data=cps_sci_arr, header=cps_hdr)
            model_img_name = dir_img_name.replace('.fits', '_formodel.fits')
            mhdu.writeto(model_img_name, overwrite=True) 

            # ---------------
            # First pass of SExtractor 
            # See notes on sextractor args for subprocess
            # in gen_sed_lst.py
            # Change directory to images directory
            os.chdir(img_sim_dir)

            cat_filename = model_img_name.replace('.fits', '.cat')
            checkimage   = model_img_name.replace('.fits', '_segmap.fits')

            sextractor   = subprocess.run(['sex', model_img_name, 
                '-c', 'roman_sims_sextractor_config.txt', 
                '-CATALOG_NAME', os.path.basename(cat_filename), 
                '-CHECKIMAGE_NAME', checkimage], check=True)

            # Go back to roman-slitless directory
            os.chdir(roman_slitless_dir)

            # ---------------
            # Convert to model image
            model_img = gen_model_img(model_img_name, checkimage)

            # ---------------
            # Add a small amount of background
            # The mean is zero and the standard deviation is
            # is about 80 times lower than the expected counts
            # for a 29th mag source (which are ~0.08 for mag=29.0).
            # By trial and error I found that this works best for
            # SExtractor being able to detect sources down to 27.0
            # Not sure why it needs to be that much lower...
            model_img += np.random.normal(loc=0.0, scale=back_scale, size=model_img.shape)

            # ---------------
            # Get a list of x-y coords to insert SNe at
            x_ins, y_ins = get_insertion_coords(num_to_insert)

            # ---------------
            # Now insert as many SNe as required
            tqdm.write("--"*16)
            tqdm.write("  x      y           mag")
            tqdm.write("--"*16)
            snmag_arr = np.zeros(num_to_insert)

            for i in range(num_to_insert):

                # Decide some random mag for the SN
                # This is a power law # previously uniform dist
                pow_idx = 1.5  # power law index # PDF given by: P(x;a) = a * x^(a-1)
                snmag = np.random.power(pow_idx, size=None)
                snmag = snmag * (highmag - lowmag) + lowmag
                snmag_arr[i] = snmag

                # Hack because Sextractor for some reason assigns 
                # fainter mags to these SNe # by about ~0.1 to 0.3 mag
                # depending on the inserted magnitude.
                #snmag -= 0.3
                # I think this problem is because when SExtractor is 
                # run again on the SNadded images the flux is summed 
                # within a smaller area NOT the whole cutout area (like
                # np.sum below in the new_cutout). Therefore the 
                # SExtractor count and consequently mag falls short i.e., fainter.
                # Hacked for now, will have to figure out some fix later.

                # Another hack because we know that SExtractor 
                # mags are fainter than truth mags by about 0.25 mag.
                # NOT just for SNe but for all objects.
                # This hack works only the SNe for now.
                # See result of test_for_zp.py
                #snmag_eff -= 0.25

                # Now scale reference
                delta_m = ref_mag - snmag
                sncounts = ref_counts * (1 / 10**(-0.4*delta_m) )

                scale_fac = sncounts / ref_counts
                new_cutout = ref_data * scale_fac

                if verbose:
                    tqdm.write('Inserted SN mag: ' + "{:.3f}".format(snmag))
                    tqdm.write('delta_m: ' + "{:.3f}".format(delta_m))
                    tqdm.write('Added SN counts: ' + "{:.3f}".format(sncounts))
                    tqdm.write('Scale factor: ' + "{:.3f}".format(scale_fac))

                # Now get coords
                xi = x_ins[i]
                yi = y_ins[i]

                r = yi
                c = xi

                # Add in the new SN
                model_img[r-s:r+s, c-s:c+s] = model_img[r-s:r+s, c-s:c+s] + new_cutout

                tqdm.write(str(xi) + "  " + str(yi) + "    " + "{:.3f}".format(snmag) + "    " + "{:.3f}".format(sncounts))

            # Save the locations and SN mag as a numpy array
            added_sn_data = np.c_[x_ins, y_ins, snmag_arr]
            snadd_fl = dir_img_name.replace('.fits', '_SNadded.npy')
            np.save(snadd_fl, added_sn_data)
            tqdm.write('Saved: ' + snadd_fl)

            # Save and check with ds9
            new_hdu = fits.PrimaryHDU(header=cps_hdr, data=model_img)
            savefile = dir_img_name.replace('.fits', '_SNadded.fits')
            new_hdu.writeto(savefile, overwrite=True)
            tqdm.write('Saved: ' + savefile)

            # Also add a regions file for the added SNe
            snadd_regfl = dir_img_name.replace('.fits', '_SNadded.reg')
            with open(snadd_regfl, 'w') as fhreg:

                fhreg.write("# Region file format: DS9 version 4.1" + "\n")
                fhreg.write("global color=red dashlist=8 3 width=3 font=\"helvetica 10 normal roman\" ")
                fhreg.write("select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 ")
                fhreg.write("delete=1 include=1 source=1" + "\n")
                fhreg.write("image" + "\n")

                for i in range(num_to_insert):

                    fhreg.write("circle(" + \
                                "{:.1f}".format(x_ins[i])  + "," + \
                                "{:.1f}".format(y_ins[i]) + "," + \
                                "9.5955367)" + " # color=red" + \
                                " width=3" + "\n")

            tqdm.write('Saved: ' + snadd_regfl)

            # Clean up intermediate files
            os.remove(checkimage)
            os.remove(cat_filename)
            os.remove(model_img_name)

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)


