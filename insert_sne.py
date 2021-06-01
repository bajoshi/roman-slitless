import numpy as np
from astropy.io import fits

import os
import sys
import socket

import matplotlib.pyplot as plt

# Get the dirs correct
if 'plffsn2' in socket.gethostname():
    extdir = '/astro/ffsn/Joshi/'
    modeldir = extdir + 'bc03_output_dir/'
    roman_direct_dir = home + '/Documents/roman_direct_sims/sims2021/'
else:
    extdir = '/Volumes/Joshi_external_HDD/Roman/'
    modeldir = extdir + 'bc03_output_dir/m62/'
    roman_direct_dir = extdir + 'roman_direct_sims/sims2021/'

assert os.path.isdir(modeldir)
assert os.path.isdir(roman_direct_dir)

dir_img_part = 'part1'
img_sim_dir = roman_direct_dir + 'K_5degimages_' + dir_img_part + '/'

def get_insertion_coords(num_to_insert, img_cat=None):

    x_ins = np.zeros(num_to_insert)
    y_ins = np.zeros(num_to_insert)

    x_ins = np.random.randint(low=110, high=3985, size=num_to_insert)
    y_ins = np.random.randint(low=110, high=3985, size=num_to_insert)

    return x_ins, y_ins

def gen_reference_cutout():

    showref = False

    img_basename = '5deg_'
    img_suffix = 'Y106_0_6'

    xloc = 2512
    yloc = 2268

    dir_img_name = img_sim_dir + img_basename + img_suffix + '_cps.fits'
    dir_hdu = fits.open(dir_img_name)
    img_arr = dir_hdu[0].data

    r = yloc
    c = xloc

    s = 50

    ref_img = img_arr[r-s:r+s, c-s:c+s]

    rhdu = fits.PrimaryHDU(data=ref_img)
    rhdu.writeto(img_sim_dir + 'ref_cutout.fits', overwrite=True)

    if showref:

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(np.log10(ref_img), origin='lower')
        plt.show()

    return None

def main():

    num_to_insert = np.random.randint(low=8, high=20)

    # ---------------
    # some preliminary settings
    img_basename = '5deg_'
    img_suffix = 'Y106_0_6'
    ref_mag = 15.76
    s = 50  # same as the size of the cutout stamp  # cutout is 100x100; need half that here
    verbose = False

    # Open dir image
    dir_img_name = img_sim_dir + img_basename + img_suffix + '_cps.fits'
    dir_hdu = fits.open(dir_img_name)

    # Copy dir image for adding fake SNe
    img_arr = dir_hdu[0].data

    # ---------------
    # Get a list of x-y coords to insert SNe at
    print("Will insert", num_to_insert, "SNe in", os.path.basename(dir_img_name))

    # first get center coords of image
    ra_cen = float(dir_hdu[0].header['CRVAL1'])
    dec_cen = float(dir_hdu[0].header['CRVAL2'])

    x_ins, y_ins = get_insertion_coords(num_to_insert)

    # ---------------
    # Read in the reference image of the star from 
    # Y106_0_6 at 72.1175280, -53.5739388 (~16th mag)
    ref_cutout_path = img_sim_dir + 'ref_cutout.fits'
    if not os.path.isfile(ref_cutout_path):
        gen_reference_cutout()

    ref_cutout = fits.open(ref_cutout_path)
    ref_data = ref_cutout[0].data

    # ---------------
    # Now insert as many SNe as required
    print("--"*16)
    print("  x      y           mag")
    print("--"*16)
    for i in range(num_to_insert):

        # Decide some random mag for the SN
        snmag = np.random.uniform(low=19.0, high=24.0)

        # Now scale reference
        ref_flux = np.sum(ref_data, axis=None)
        delta_m = ref_mag - snmag
        snflux = ref_flux * (1/np.power(10, -1*0.4*delta_m))
        
        scale_fac = snflux / ref_flux
        new_cutout = ref_data * scale_fac

        if verbose:
            print('Inserted SN mag:', snmag)
            print('delta_m:', delta_m)
            print('Added SN flux:', snflux)
            print('Scale factor:', scale_fac)
            print('New flux:', np.sum(new_cutout, axis=None))

        # Now get coords
        xi = x_ins[i]
        yi = y_ins[i]

        r = yi
        c = xi

        # Add in the new SN
        img_arr[r-s:r+s, c-s:c+s] = img_arr[r-s:r+s, c-s:c+s] + new_cutout

        print(xi, yi, "    ", "{:.3f}".format(snmag))

    # Save and check with ds9
    new_hdu = fits.PrimaryHDU(header=dir_hdu[0].header, data=img_arr)
    savefile = dir_img_name.replace('.fits', '_SNadded.fits')
    new_hdu.writeto(savefile, overwrite=True)

    # Also add a regions file for the added SNe
    with open(dir_img_name.replace('.fits', '_SNadded.reg'), 'w') as fhreg:

        fhreg.write("# Region file format: DS9 version 4.1" + "\n")
        fhreg.write("global color=red dashlist=8 3 width=3 font=\"helvetica 10 normal roman\" ")
        fhreg.write("select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 ")
        fhreg.write("delete=1 include=1 source=1" + "\n")
        fhreg.write("image" + "\n")

        for i in range(num_to_insert):

            fhreg.write("circle(" + \
                        "{:.1f}".format(x_ins[i])  + "," + \
                        "{:.1f}".format(y_ins[i]) + "," + \
                        "9.5955367" + "# color=red" + \
                        " width=3" + "\n")

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)


