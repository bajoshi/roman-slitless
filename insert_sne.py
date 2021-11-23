import numpy as np
from astropy.io import fits

import os
import sys
import socket
import subprocess
from tqdm import tqdm
import pdb
import warnings

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


def get_insertion_coords(num_to_insert, 
    img_cat=None, img_segmap=None, imdat=None, checkplot=False):

    x_ins = np.zeros(num_to_insert)
    y_ins = np.zeros(num_to_insert)

    if img_cat is None:  # i.e., insert SNe randomly

        x_ins = np.random.randint(low=110, high=3985, size=num_to_insert)
        y_ins = np.random.randint(low=110, high=3985, size=num_to_insert)

        return x_ins, y_ins

    else:  # i.e., insert SNe next to galaxies

        # Empty array for hsot galaxy magnitudes
        host_galaxy_mags = np.zeros(num_to_insert)

        # Read in catalog
        cat_header = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'ALPHA_J2000', 'DELTA_J2000', 
        'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_AUTO', 'MAGERR_AUTO', 'FLUX_RADIUS', 
        'FWHM_IMAGE', 'CLASS_STAR']
        cat = np.genfromtxt(img_cat, dtype=None, names=cat_header, 
            encoding='ascii')

        # Read in segmap
        segmap = fits.getdata(img_segmap)

        # Now insert each SN next to another object with 
        # at least 20 pixels in the segmap, as long as that 
        # many galaxies exist, otherwise move on to smaller
        # hosts.
        # First get the number of pixels for each source,
        # and sort in descending order.
        num_src_pix_array = np.zeros(len(cat))
        for i in tqdm(range(len(cat)), desc='Computing total src pix'):

            src_segid = cat['NUMBER'][i]
            # Get source indices in segmap
            # and count them. Indices will be 2d.
            src_idx = np.where(segmap == src_segid)[0]
            # The [0] here is okay. It simply counts the 
            # x coords of the pixels. The total number 
            # of pix is simply the total number of x coords.

            num_src_pix_array[i] = src_idx.size

        # Now sort the num_src_pix array in descending
        # order and keep track of source seg IDs
        num_src_pix_array_desc = np.sort(num_src_pix_array)[::-1]
        argsort_idx = np.argsort(num_src_pix_array)
        argsort_idx = argsort_idx[::-1]

        desc_src_segids = cat['NUMBER'][argsort_idx]
        class_star = cat['CLASS_STAR'][argsort_idx]
        mag_arr = cat['MAG_AUTO'][argsort_idx]

        # NOw loop over total SNe required
        count = 0
        for j in range(len(cat)):

            num_src_pix = num_src_pix_array_desc[j]
            src_segid = desc_src_segids[j]

            # Now insert SN close to the other object if all okay
            if num_src_pix >= 20:

                src_x, src_y = np.where(segmap == src_segid)

                # Get a bounding box for the source
                top    = np.max(src_y)
                bottom = np.min(src_y)

                right  = np.max(src_x)
                left   = np.min(src_x)

                # Put the SN shifted out 1 pix away from one 
                # of hte four corners of the bounding box
                xsn = np.random.choice([left, right]) + 1
                ysn = np.random.choice([top, bottom]) + 1

                # Put the SN somewhere within the bounding box
                # xsn = np.random.choice(np.arange(left, right))
                # ysn = np.random.choice(np.arange(bottom, top))

                # Check if it is a star
                star = class_star[j]
                if star > 0.25:
                    continue

                # Now we make sure that not all SNe are associated
                # with bright galaxies. Because we've sorted the 
                # num_src_pix array in descending order we're likely
                # to have all the brightest galaxies show up first
                # (stars are being filtered out above). So we need
                # to ensure that all SNe don't preferentially go to 
                # brighter galaxies.
                galaxy_mag = mag_arr[j]
                pow_prob = np.random.power(2.0, size=None)
                gmag = (galaxy_mag - 17) / (27 - 17)
                sn_prob = gmag * pow_prob
                # i.e., 17th mag is approx brightest galaxy
                # and 27th mag is approx faintest galaxy
                # Explanation: the first line picking the random
                # number from the power distribution will pick
                # a number between 0 and 1 according to the power law.
                # The second line will assign an updated probability
                # by saying that the galaxies that get assigned SNe
                # follow the power law distribution. 
                if sn_prob < 0.3:  # this limit was determined by trial and error
                    continue

                print(j, src_segid, num_src_pix, xsn, ysn, star, 
                    '{:.2f}'.format(sn_prob), galaxy_mag)

                x_ins[count] = xsn
                y_ins[count] = ysn
                host_galaxy_mags[count] = galaxy_mag

                count += 1
                if count >= num_to_insert: break

                # Check cutout of inserted SN and galaxy
                if checkplot:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")

                        fig = plt.figure()
                        ax = fig.add_subplot(111)

                        # Get cutout
                        im_cutout = imdat[left-10:right+10, bottom-10:top+10]

                        # Image extent
                        ext = [left-10, right+10, bottom-10, top+10]

                        # Ensure square extent
                        x_extent = right+10 - left-10
                        y_extent = top+10 - bottom-10

                        if x_extent > y_extent:
                            ext_diff = x_extent - y_extent
                            ext = [left-10, right+10, bottom-10-int(ext_diff/2), top+10+int(ext_diff/2)]
                        elif y_extent > x_extent:
                            ext_diff = y_extent - x_extent
                            ext = [left-10-int(ext_diff/2), right+10+int(ext_diff/2), bottom-10, top+10]

                        ax.imshow(np.log10(im_cutout), extent=ext, origin='lower')
                        ax.scatter(xsn, ysn, marker='x', lw=5.0, s=60, color='red')

                        plt.show()

                    if j > 50: sys.exit(0)

            else:
                continue

        # Plot distribution of host galaxy magnitudes
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.hist(host_galaxy_mags, 20, color='k', 
            range=(17.0, 27.0), histtype='step')

        plt.show()
        sys.exit(0)

        return x_ins, y_ins, host_galaxy_mags


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
            num_to_insert = np.random.randint(low=90, high=100)

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
            x_ins, y_ins = get_insertion_coords(num_to_insert, 
                img_cat=cat_filename, img_segmap=checkimage, imdat=cps_sci_arr)

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

                tqdm.write(str(xi) + "  " + str(yi) + "    " + \
                    "{:.3f}".format(snmag) + "    " + "{:.3f}".format(sncounts))

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


