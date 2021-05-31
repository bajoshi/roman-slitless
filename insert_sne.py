import numpy as np
from astropy.io import fits

import os
import sys
import socket

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



    return x_ins, y_ins

def main():

    num_to_insert = np.random.randint(low=8, high=20)

    # ---------------
    # some preliminary settings
    img_basename = '5deg_'
    img_suffix = 'Y106_0_6'
    ref_mag = 15.76

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

    # ---------------
    # Now insert as many SNe as required
    print("--"*16)
    print("  x      y           mag")
    print("--"*16)
    for i in range(num_to_insert):

        # Decide some random mag for the SN
        snmag = np.random.uniform(low=19.0, high=24.0)

        # Now scale reference
        ref_flux = np.sum(ref_cutout[0].data)
        delta_m = ref_mag - snmag
        snref = (1/ref_flux) * np.power(10, 0.4*delta_m)

        print(ref_flux, snref, delta_m)

        scale_fac = snref / ref_flux

        new_cutout = ref_cutout * scale_fac

        # Now get coords
        xi = x_ins[i]
        yi = y_ins[i]

        rowidx = 
        colidx = 

        # Add in the new SN
        img_arr[rowidx, colidx] = img_arr[rowidx, colidx] + new_cutout


        print(xi, yi, "    ", "{:.3f}".format(snmag))

    # Save and check with ds9
    new_hdu = fits.PrimaryHDU(header=dir_hdu[0].header, data=img_arr)
    savefile = dir_img_name.replace('.fits', '_SNadded.fits')
    new_hdu.writeto(savefile, overwrite=True)

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)


