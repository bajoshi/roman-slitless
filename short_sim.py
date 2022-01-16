import numpy as np
from astropy.io import fits

import os
import sys

import pylinear

from gen_sed_lst import get_sn_z, get_sn_spec_path

extdir = "/Volumes/Joshi_external_HDD/Roman/"
img_sim_dir = extdir + "roman_direct_sims/sims2021/K_5degimages_part1/"
savedir = img_sim_dir + 'shortsim/'
if not os.path.isdir(savedir):
    os.mkdir(savedir)

# Params of sim
NPIX = 4096
NUM_SNE = 100
REF_COUNTS = 13696.77
SIZE = 50  # half of cutout
# LOWMAG = 22.42
# HIGHMAG = 28.71
ZP = 26.264 

MAGLIM = 29.0
BEAM = '+1'
EXPTIME = 400  # seconds per FLT file

# Noise budget
SKY = 1.1
DARK = 0.015
READNOISE = 8


def get_counts(mag):
    counts = np.power(10, -0.4 * (mag - ZP))
    return counts


def insert_sne_getsedlst_shortsim():

    # Get a blank image of the correct size
    full_img = np.zeros((NPIX, NPIX))

    # ---------------
    # Get reference image
    # Read in the reference image of the star from 
    # Y106_0_6 at 72.1175280, -53.5739388 (~16th mag)
    ref_cutout_path = img_sim_dir + 'ref_cutout_psf.fits'

    ref_data = fits.getdata(ref_cutout_path)

    # Empty array to store inserted sne properties
    insert_cat = np.zeros((NUM_SNE, 3))

    # ---------------
    # Loop over sne to insert
    for i in range(NUM_SNE):

        # Scale SN and add to full image
        # Decide some random mag for the SN
        snmag = np.random.normal(loc=27.0, scale=0.1, size=None)
        sncounts = get_counts(snmag)
        scale_fac = sncounts / REF_COUNTS
        sn_img = ref_data * scale_fac

        # Get SN coords # random
        r = np.random.randint(low=110, high=3985, size=None)
        c = np.random.randint(low=110, high=3985, size=None)

        full_img[r-SIZE:r+SIZE, c-SIZE:c+SIZE] += sn_img

        print(i+1, '  ', r, '  ', c, '  ', '{:.3f}'.format(snmag))

        insert_cat[i][0] = r
        insert_cat[i][1] = c
        insert_cat[i][2] = snmag

    # Save the file
    savefile = savedir + 'shortsim_image.fits'

    # Get a header with WCS
    hdr = fits.getheader(img_sim_dir + '5deg_Y106_0_1.fits')

    hdu = fits.PrimaryHDU(data=full_img, header=hdr)
    hdu.writeto(savefile, overwrite=True)

    # =====================================
    # Create SED LST file
    sedlst_filename = savedir + 'sed_shortsim.lst'

    # Open an empty file for writing sed lst
    with open(sedlst_filename, 'w') as fh:

        # ---- Write header
        fh.write("# 1: SEGMENTATION ID" + "\n")
        fh.write("# 2: SED FILE" + "\n")

        # ---------------
        # Loop over sne to insert
        for j in range(NUM_SNE):

            # First we need the mag for the jth SN
            m = insert_cat[j][2]

            # Also assign the SED
            # First we need the correct z
            z = get_sn_z(m)
            spec_path = get_sn_spec_path(z)

            # ------------ Write to file
            fh.write(str(j+1) + " " + spec_path + "\n")

    return None


def run_pylinear_shortsim():

    segfile = savedir + 'test_segmap.fits'
    obslst = savedir + 'obs_shortsim.lst'
    wcslst = savedir + 'wcs_shortsim.lst'
    sedlst = savedir + 'sed_shortsim.lst'
    fltlst = savedir + 'flt_shortsim.lst'

    # ---------------------- Get sources
    sources = pylinear.source.SourceCollection(segfile, obslst, 
                                               detindex=0, maglim=MAGLIM)

    # Set up
    grisms = pylinear.grism.GrismCollection(wcslst, observed=False)
    tabulate = pylinear.modules.Tabulate('pdt', ncpu=0)
    tabulate.run(grisms, sources, BEAM)
    print("Done with tabulation.")

    # ---------------------- Simulate
    print("Simulating...")
    simulate = pylinear.modules.Simulate(sedlst, gzip=False, ncpu=0)
    simulate.run(grisms, sources, BEAM)
    print("Simulation done.")

    # ---------------------- Add noise
    for f in range(3):
        flt = savedir + 'shortsim' + str(f+1) + '_flt.fits'
        scidata = fits.getdata(flt)

    return None


if __name__ == '__main__':

    # ======================
    # Insert SNe and assign the SEDs
    # insert_sne_getsedlst_shortsim()

    # print('\nNow run SExtractor and create the LST files by hand.')
    # print('Run the following command for SExtractor in the shortsim folder:')
    # print('>> sex shortsim_image.fits -c default_config.txt')
    # print('Make sure to comment out the insert code before running the next',
    #       'function to run pylinear on the short sim.')
    # sys.exit(0)

    # Run pyLINEAR
    run_pylinear_shortsim()

    sys.exit(0)
