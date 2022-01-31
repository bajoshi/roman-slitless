import numpy as np
from astropy.io import fits

import os
import sys
import socket
import subprocess
from tqdm import tqdm
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

# sys.path.append(utils_dir)
# from make_model_dirimg import gen_model_img  # noqa: E402

# ---------------------- GLOBAL DEFS
# Scaling factor for direct images
# The difference here that appears in the power of 10
# is the difference between the ZP of the current img
# and what I think the correct ZP is i.e., the WFC3/F105W ZP.
DIRIMAGE_SCALING = 10**(-0.4 * (31.7956 - 26.264))
# ----------------------
BACK_SCALE = 0.001  # standard deviation for background to be added.
# read in mag and counts from SExtractor catalog on dir img
REF_COUNTS = 13696.77
REF_MAG = 15.9225

CUTOUT_SIZE = 50
# ----------------------


def get_segpix(snmag):

    blank_smap = np.zeros((CUTOUT_SIZE * 2, CUTOUT_SIZE * 2), dtype=np.int64)

    if snmag > 26.0:
        # central 3x3 grid
        blank_smap[49:52, 49:52] = 1
    elif snmag > 24.0 and snmag <= 26.0:
        # central 4x4 grid
        blank_smap[48:52, 48:52] = 1
    elif snmag > 22.0 and snmag <= 24.0:
        # central 4x4 grid
        blank_smap[48:52, 48:52] = 1
        # add 4 "spokes"
        blank_smap[50, 47] = 1
        blank_smap[50, 52] = 1
        blank_smap[47, 50] = 1
        blank_smap[52, 50] = 1
    elif snmag > 19.0 and snmag <= 22.0:
        # central 6x6 grid
        blank_smap[47:53, 47:53] = 1
        # add 4 "spokes"
        blank_smap[50, 46] = 1
        blank_smap[50, 53] = 1
        blank_smap[46, 50] = 1
        blank_smap[53, 50] = 1
    else:  # STARS
        # central 10x10 grid
        blank_smap[45:55, 45:55] = 1

    segpix = np.where(blank_smap == 1)

    return segpix


def get_ref_segpix_counts(snmag):
    # OLD method: See notes in ref_cutout_segpix.py

    # ---------
    # Get the segmentation pixels
    segpix = get_segpix(snmag)
    # since there is only one detected object
    # in all the reference segmaps.

    # Ensure that when the counts in the segpix for
    # this supernova are summed you get the scaled
    # reference counts. You cannot simply do np.sum()
    # on the new cutout because it adds everything,
    # including pixels that won't be included in the
    # segpix and will therefore be an overestimate.

    # This can be done by up scaling the sncounts
    # from above such that it satifies the above condition.
    # refimgfile = segmap.replace('_segmap.fits', '.fits')
    # refimgdat = fits.getdata(refimgfile)
    # segcounts = np.sum(refimgdat[segpix[0], segpix[1]])

    delta_m = REF_MAG - snmag
    sncounts = REF_COUNTS * (1 / 10**(-0.4 * delta_m))

    # Now determine scaling and scale ref img to check
    # sf = sncounts / segcounts
    # new_ref_dat = refimgdat * sf
    # new_segcounts = np.sum(new_ref_dat[segpix[0], segpix[1]])

    # fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10,4))
    # ax1.imshow(np.log10(refimgdat), origin='lower', vmin=-0.01, vmax=2.0)
    # ax2.imshow(segdata)
    # ax3.imshow(np.log10(new_ref_dat), origin='lower', vmin=-0.01, vmax=2.0)
    # plt.show()
    # print('\nCounts from refimgdat:', segcounts)
    # print('Required SN mag and counts:', snmag, sncounts)
    # print('New scaled counts:', new_segcounts, sf, '\n')

    return sncounts, segpix


def get_insertion_coords(num_to_insert,
                         img_cat=None, img_segmap=None,
                         imdat=None, checkplot=False):

    x_ins = np.zeros(num_to_insert, dtype=np.int64)
    y_ins = np.zeros(num_to_insert, dtype=np.int64)

    if img_cat is None:  # i.e., insert SNe randomly

        x_ins = np.random.randint(low=110, high=3985, size=num_to_insert)
        y_ins = np.random.randint(low=110, high=3985, size=num_to_insert)

        return x_ins, y_ins

    else:  # i.e., insert SNe next to galaxies

        # Empty array for hsot galaxy magnitudes
        host_galaxy_mags = np.zeros(num_to_insert)
        host_galaxy_ids = np.zeros(num_to_insert, dtype=np.int64)

        # Read in catalog
        cat_header = ['NUMBER', 'X_IMAGE', 'Y_IMAGE',
                      'ALPHA_J2000', 'DELTA_J2000',
                      'FLUX_AUTO', 'FLUXERR_AUTO',
                      'MAG_AUTO', 'MAGERR_AUTO',
                      'FLUX_RADIUS', 'FWHM_IMAGE',
                      'CLASS_STAR']
        cat = np.genfromtxt(img_cat, dtype=None, names=cat_header,
                            encoding='ascii')

        # PUll out mags
        cat_mags = cat['MAG_AUTO']

        # Read in segmap
        segmap = fits.getdata(img_segmap)

        # Now insert each SN next to another object with
        # at least 15 pixels in the segmap, as long as that
        # many galaxies exist, otherwise move on to smaller
        # hosts.
        sn_added_count = 0
        while sn_added_count < num_to_insert:

            # First find a galaxy close to a chosen magnitude
            # This magnitude is drawn from a power law distribution
            # similar to the distribution for SNe below. This
            # ensures that SNe are not preferentially inserted
            # into faint/bright galaxies.
            pow_prob = np.random.power(2.0, size=None)
            galaxy_mag_to_match = pow_prob * (27 - 17) + 17.0
            # This line above assumes 17th mag is approx brightest galaxy
            # and 27th mag is approx faintest galaxy

            # Now find a galaxy in the catalog that is within
            # +- 0.2 mags of this above mag
            # 0.2 mag is initial search width
            mag_width = 0.2
            all_cat_idx = \
                np.where((cat_mags >= galaxy_mag_to_match - mag_width)
                         & (cat_mags <= galaxy_mag_to_match + mag_width))[0]

            # Randomly pick a galaxy that is within this range
            while not all_cat_idx.size:  # ie., no galaxies found in mag range
                mag_width += 0.05
                all_cat_idx = \
                    np.where((cat_mags >= galaxy_mag_to_match - mag_width)
                             & (cat_mags
                                <= galaxy_mag_to_match + mag_width))[0]

            cat_idx = np.random.choice(all_cat_idx)

            # Keep track of the host_galaxy IDS so there are no repeats
            src_segid = cat['NUMBER'][cat_idx]

            if src_segid in host_galaxy_ids:
                # print('SKIPPING REPEAT SOURCE.')
                continue

            # Get source indices in segmap
            # and count them. Indices will be 2d.
            src_rows, src_cols = np.where(segmap == src_segid)

            num_src_pix = src_rows.size

            # Check if it is a star
            star = cat['CLASS_STAR'][cat_idx]
            if star > 0.25:
                # print('SKIPPING STAR.')
                continue

            # Get a bounding box for the source
            top = np.max(src_rows)
            bottom = np.min(src_rows)

            right = np.max(src_cols)
            left = np.min(src_cols)

            # Ensure that host galaxy is not too close to the edge
            if (top > 3985) or (right > 3985) or \
               (left < 110) or (bottom < 110):
                # print('SKIPPING HOST TOO CLOSE TO EDGE.')
                continue

            # Now insert SN close to the other object if all okay
            if num_src_pix >= 15:

                # Put the SN shifted out some pix away from one
                # of hte four corners of the bounding box
                xsn = np.random.choice([left, right]) + 3
                ysn = np.random.choice([top, bottom]) + 3

                # Put the SN somewhere within the bounding box
                # xsn = np.random.choice(np.arange(left, right))
                # ysn = np.random.choice(np.arange(bottom, top))

                galaxy_mag = cat['MAG_AUTO'][cat_idx]

                # print(sn_added_count, src_segid, num_src_pix,
                #       xsn, ysn, star, galaxy_mag)

                x_ins[sn_added_count] = xsn
                y_ins[sn_added_count] = ysn
                host_galaxy_mags[sn_added_count] = galaxy_mag
                host_galaxy_ids[sn_added_count] = src_segid

                sn_added_count += 1

                # Check cutout of inserted SN and galaxy
                if checkplot:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")

                        fig = plt.figure()
                        ax = fig.add_subplot(111)

                        # Get cutout
                        im_cutout = imdat[bottom - 10:top + 10,
                                          left - 10:right + 10]

                        # Image extent
                        ext = [left - 10, right + 10, bottom - 10, top + 10]

                        # Ensure square extent
                        x_extent = right + 10 - left - 10
                        y_extent = top + 10 - bottom - 10

                        if x_extent > y_extent:
                            ext_diff = x_extent - y_extent
                            ext = [left - 10, right + 10,
                                   bottom - 10 - int(ext_diff / 2),
                                   top + 10 + int(ext_diff / 2)]
                        elif y_extent > x_extent:
                            ext_diff = y_extent - x_extent
                            ext = [left - 10 - int(ext_diff / 2),
                                   right + 10 + int(ext_diff / 2),
                                   bottom - 10, top + 10]

                        # Show image of galaxy and mark SN location
                        ax.imshow(np.log10(im_cutout), extent=ext,
                                  origin='lower')
                        ax.scatter(xsn, ysn, marker='x', lw=5.0,
                                   s=60, color='red')

                        plt.show()

                        if sn_added_count > 10:
                            sys.exit(0)

            else:
                # print('SKIPPING TOO SMALL SOURCE.')
                continue

        return x_ins, y_ins, host_galaxy_mags, host_galaxy_ids


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
        cps_sci_arr = ddat * DIRIMAGE_SCALING

        # Save to be able to run sextractor to generate model image
        rhdu = fits.PrimaryHDU(data=cps_sci_arr, header=dhdr)
        rhdu.writeto(dir_img_name)

    img_arr = fits.getdata(dir_img_name)

    r = yloc
    c = xloc

    ref_img = img_arr[r - CUTOUT_SIZE:r + CUTOUT_SIZE,
                      c - CUTOUT_SIZE:c + CUTOUT_SIZE]

    # Save
    rhdu = fits.PrimaryHDU(data=ref_img)
    ref_name = img_sim_dir + 'ref_cutout_psf.fits'
    rhdu.writeto(ref_name, overwrite=True)

    # --------------
    # Run SExtractor on the reference cutout
    os.chdir(img_sim_dir)

    cat_filename = ref_name.replace('.fits', '.cat')
    checkimage = ref_name.replace('.fits', '_segmap.fits')

    subprocess.run(['sex', ref_name,
                    '-c', 'roman_sims_sextractor_config.txt',
                    '-CATALOG_NAME',
                    os.path.basename(cat_filename),
                    '-CHECKIMAGE_NAME', checkimage],
                   check=True)

    # Go back to roman-slitless directory
    os.chdir(roman_slitless_dir)

    # --------------
    # Now read in the catalog just created to get
    # the reference's counts and magnitude
    ref_cat_hdr = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'ALPHA_J2000',
                   'DELTA_J2000', 'FLUX_AUTO', 'FLUXERR_AUTO',
                   'MAG_AUTO', 'MAGERR_AUTO', 'FLUX_RADIUS',
                   'FWHM_IMAGE', 'CLASS_STAR']
    ref_cat = np.genfromtxt(cat_filename, dtype=None,
                            names=ref_cat_hdr, encoding='ascii')
    assert ref_cat.size == 1
    print('Counts for reference star:', ref_cat['FLUX_AUTO'])
    print('MAG for reference star:', ref_cat['MAG_AUTO'])

    # --------------
    if showref:

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(np.log10(ref_img), origin='lower')
        plt.show()

    return None


def get_fullgrid_segpix(segpix, row, col):

    fullgrid_segpix_rows = row - 50 + segpix[0]
    fullgrid_segpix_cols = col - 50 + segpix[1]

    fullgrid_segpix = np.array([fullgrid_segpix_rows, fullgrid_segpix_cols])

    return fullgrid_segpix


if __name__ == '__main__':

    # ---------------
    # some preliminary settings
    img_basename = '5deg_'
    s = 50  # same as the size of the cutout stamp
    # cutout is 100x100; need half that here

    # Mag limits for choosing random SN mag
    # low and high limits here correspond to
    # almost exactly z=0.5 and z=3.0
    lowmag = 22.42
    highmag = 28.71

    s = CUTOUT_SIZE

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
    detectors = np.arange(1, 2, 1)

    for pt in pointings:
        for det in detectors:

            # ---------------
            # Determine number of SNe to insert and open dir img
            num_to_insert = np.random.randint(low=90, high=100)
            # Also decide number of stars
            # randomly insert approx 10 stars in each detector
            num_to_insert_stars = np.random.randint(low=8, high=12)

            # ---------------
            img_suffix = 'Y106_' + str(pt) + '_' + str(det)
            dir_img_name = img_sim_dir + img_basename + img_suffix + '.fits'

            print("Working on: " + dir_img_name)
            print("Will insert " + str(num_to_insert) + " SNe in "
                  + os.path.basename(dir_img_name))

            # First check that the files have been unzipped
            if not os.path.isfile(dir_img_name):
                print("Unzipping file: " + dir_img_name + ".gz")
                subprocess.run(['gzip', '-fd', dir_img_name + '.gz'])

            # Open dir image
            dir_hdu = fits.open(dir_img_name)

            # ---------------
            # Now scale image to get the image to counts per sec
            cps_sci_arr = dir_hdu[1].data * DIRIMAGE_SCALING
            cps_hdr = dir_hdu[1].header
            dir_hdu.close()
            # cps_hdr['BUNIT'] = 'ELECTRONS'
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
            checkimage = model_img_name.replace('.fits', '_segmap.fits')

            subprocess.run(['sex', model_img_name,
                            '-c', 'roman_sims_sextractor_config.txt',
                            '-CATALOG_NAME', os.path.basename(cat_filename),
                            '-CHECKIMAGE_NAME', checkimage], check=True)

            # Go back to roman-slitless directory
            os.chdir(roman_slitless_dir)

            # ---------------
            # Read in segmap. Will be used later
            segdata = fits.getdata(checkimage)

            # ---------------
            # Get a list of x-y coords to insert SNe at
            x_ins, y_ins, host_mags, host_segids = \
                get_insertion_coords(num_to_insert,
                                     img_cat=cat_filename,
                                     img_segmap=checkimage,
                                     imdat=cps_sci_arr)

            # ================================================
            # Now insert as many SNe as required
            # print("--"*16)
            # print("  x      y           mag")
            # print("--"*16)

            insert_mag = np.zeros(num_to_insert + num_to_insert_stars)
            object_type = np.empty(num_to_insert + num_to_insert_stars,
                                   dtype='<U4')
            insert_segid = np.zeros(num_to_insert + num_to_insert_stars,
                                    dtype=int)

            last_segid = np.max(segdata)

            for i in tqdm(range(num_to_insert), desc='Inserting SNe'):

                # Decide some random mag for the SN
                # This is a power law # previously uniform dist
                pow_idx = 1.2  # not too steep; 1 is uniform
                # power law index # PDF given by: P(x;a) = a * x^(a-1)
                snmag = np.random.power(pow_idx, size=None)
                snmag = snmag * (highmag - lowmag) + lowmag
                insert_mag[i] = snmag

                sncounts, segpix = get_ref_segpix_counts(snmag)

                scale_fac = sncounts / REF_COUNTS
                new_cutout = ref_data * scale_fac

                # Now update counts to recover the required mag
                # from summing only the segpix in the scaled ref data
                cutout_sum = np.sum(new_cutout[segpix[0], segpix[1]])
                eps = sncounts / cutout_sum
                new_cutout_scaled = eps * new_cutout

                # scaled_cutout_sum = np.sum(new_cutout_scaled[segpix[0],
                #                                              segpix[1]])
                # inferred_mag = -2.5 * np.log10(scaled_cutout_sum) + 26.264
                # print(i,
                #       '{:.3f}'.format(snmag),
                #       '{:.3f}'.format(inferred_mag),
                #       '{:.3f}'.format(snmag - inferred_mag))

                # Now get coords
                xi = x_ins[i]
                yi = y_ins[i]

                r = yi
                c = xi

                # Add in the new SN in the direct image
                cps_sci_arr[r - s:r + s, c - s:c + s] = \
                    cps_sci_arr[r - s:r + s, c - s:c + s] + new_cutout_scaled

                # Also add it into the segmap
                # First update segpix to reference the larger 4096 x 4096 grid
                segpix_big = get_fullgrid_segpix(segpix, r, c)

                # Recompute the sum of the segmap pixels for this source
                # Now that it has been added to the sci img the sum will
                # not be the same as the above scaled_cutout_sum
                sci_sum = np.sum(cps_sci_arr[segpix_big[0], segpix_big[1]])

                print(sci_sum)

                # Now update segid and add
                new_segid = last_segid + i + 1
                segdata[segpix_big[0], segpix_big[1]] = new_segid

                insert_segid[i] = new_segid

                # Now assign an object type. Used in gen_sed_lst
                # to assign spectra.
                object_type[i] = 'SNIa'

                # Print info to screen
                print(str(xi) + "  " + str(yi) + "    "
                      + "{:.3f}".format(snmag) + "    "
                      + "{:.3f}".format(sncounts) + "    "
                      + "{:.3f}".format(host_mags[i]))

                sys.exit()

            # ================================================
            # Now insert some bright stars. Same process as SNe.
            print('Inserting', num_to_insert_stars, 'stars...')
            stellar_mag_array = np.arange(15.0, 19.0, 0.1)

            last_segid = new_segid  # i.e., last segid from SNe

            star_x = np.zeros(num_to_insert_stars)
            star_y = np.zeros(num_to_insert_stars)

            for j in range(num_to_insert_stars):

                # Decide mag to insert
                stellar_mag = np.random.choice(stellar_mag_array)
                insert_mag[i + j + 1] = stellar_mag

                # Decide location to insert
                xs = np.random.randint(low=110, high=3985, size=1)
                ys = np.random.randint(low=110, high=3985, size=1)

                r = int(xs)
                c = int(ys)

                star_x[j] = xs
                star_y[j] = ys

                # Now get the segpix and counts
                starcounts, segpix = get_ref_segpix_counts(stellar_mag)

                # Scale and add
                scale_fac = starcounts / REF_COUNTS
                new_cutout = ref_data * scale_fac

                # NOT DOING THE ADDITIONAL SCALING HERE
                # Not really needed because these stars are
                # already quite bright.

                cps_sci_arr[r - s:r + s, c - s:c + s] = \
                    cps_sci_arr[r - s:r + s, c - s:c + s] + new_cutout

                # Also add it into the segmap
                segpix_big = get_fullgrid_segpix(segpix, r, c)
                new_segid = last_segid + j + 1
                segdata[segpix_big[0], segpix_big[1]] = new_segid

                insert_segid[i + j + 1] = new_segid

                # Now assign an object type.
                object_type[i + j + 1] = 'STAR'

            # Append star data to the SN data and save
            x_ins = np.append(x_ins, star_x)
            y_ins = np.append(y_ins, star_y)
            host_mags = np.append(host_mags,
                                  np.ones(num_to_insert_stars) * -99.0)
            host_segids = np.append(host_segids,
                                    np.ones(num_to_insert_stars) * -99.0)

            # Save the locations and SN mag as a numpy array
            added_sn_data = np.c_[x_ins, y_ins, insert_mag,
                                  host_mags, host_segids,
                                  object_type, insert_segid]
            snadd_fl = dir_img_name.replace('.fits', '_SNadded.npy')
            np.save(snadd_fl, added_sn_data)
            tqdm.write('Saved: ' + snadd_fl)

            # Save direct image and check with ds9
            new_hdu = fits.PrimaryHDU(header=cps_hdr, data=cps_sci_arr)
            savefile = dir_img_name.replace('.fits', '_SNadded.fits')
            new_hdu.writeto(savefile, overwrite=True)
            tqdm.write('Saved: ' + savefile)

            # Also save the edited segmentation map.
            # No more need to run SExtractor again on the same image.
            new_segmap = fits.PrimaryHDU(data=segdata, header=cps_hdr)
            new_segmap_file = savefile.replace('.fits', '_segmap.fits')
            new_segmap.writeto(new_segmap_file, overwrite=True)
            tqdm.write('Saved: ' + new_segmap_file)

            # Also add a regions file for the added SNe
            snadd_regfl = dir_img_name.replace('.fits', '_SNadded.reg')
            with open(snadd_regfl, 'w') as fhreg:

                hdr1 = "# Region file format: DS9 version 4.1" + "\n"
                hdr2 = "global color=red dashlist=8 3 width=3 " + \
                       "font=\"helvetica 10 normal roman\" "
                hdr3 = "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 "

                fhreg.write(hdr1)
                fhreg.write(hdr2)
                fhreg.write(hdr3)
                fhreg.write("delete=1 include=1 source=1" + "\n")
                fhreg.write("image" + "\n")

                for i in range(num_to_insert):

                    fhreg.write("circle("
                                + "{:.1f}".format(x_ins[i]) + ","
                                + "{:.1f}".format(y_ins[i]) + ","
                                + "9.5955367)" + " # color=red"
                                + " width=3" + "\n")

            tqdm.write('Saved: ' + snadd_regfl)

            # Clean up intermediate files
            os.remove(checkimage)
            # os.remove(cat_filename)
            os.remove(model_img_name)

    sys.exit(0)
