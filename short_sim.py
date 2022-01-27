import numpy as np
from astropy.io import fits
from astropy.modeling import models
from skimage.morphology import label
import matplotlib.pyplot as plt

import os
import sys
import socket
import subprocess

import pylinear

from gen_sed_lst import get_sn_z, get_sn_spec_path

if 'plffsn2' in socket.gethostname():
    extdir = "/astro/ffsn/Joshi/"
else:
    extdir = "/Volumes/Joshi_external_HDD/Roman/"

img_sim_dir = extdir + "roman_direct_sims/sims2021/K_5degimages_part1/"
savedir = img_sim_dir + 'shortsim/'
if not os.path.isdir(savedir):
    os.mkdir(savedir)

# Params of sim
NPIX = 4096
NUM_SNE = 1
REF_COUNTS = 13696.77
SIZE = 50  # half of cutout
SNMAG_CENTRAL = 27.0
ZP = 26.264  # For WFC3/F105W and for WFI/F106
PIXSCL = 0.108  # arcsec per pix
MODEL = 'GAUSS'  # model to use for fake SNe; GAUSS or REFSTAR

MAGLIM = 29.0
BEAM = '+1'
EXPTIME = 334  # seconds per FLT file; we have 3 exposures

# Noise budget
# See the prism info file from Jeff Kruk
# for all these numbers.
# Readnoise is effective readnoise rate
# Assumed average 1.1 factor for Zodi
BCK_ZODIACAL = 1.047 # e/pix/sec
BCK_THERMAL = 0.0637249  # e/pix/sec
DARK = 0.005  # e/pix/sec
READNOISE = 0.031  # e/pix/sec
# Effective sky
SKY = BCK_ZODIACAL + BCK_THERMAL


# Change to working directory
os.chdir(savedir)


def get_counts(mag):
    counts = np.power(10, -0.4 * (mag - ZP))
    return counts


def insert_sne_getsedlst_shortsim():

    # Get a blank image of the correct size
    full_img = np.zeros((NPIX, NPIX))

    # ---------------
    if MODEL == 'REFSTAR':
        # Get reference image
        # Read in the reference image of the star from 
        # Y106_0_6 at 72.1175280, -53.5739388 (~16th mag)
        ref_cutout_path = img_sim_dir + 'ref_cutout_psf.fits'
        ref_data = fits.getdata(ref_cutout_path)
    elif MODEL == 'GAUSS':
        # Generate Gaussian model from astropy
        # See Russell's simulating images notebook

        # Create Gaussian function model
        # stddev in pixels
        gaussfunc = models.Gaussian2D(x_stddev=3, y_stddev=3)

    # Empty array to store inserted sne properties
    insert_cat = np.zeros((NUM_SNE, 3))

    # Get a header with WCS
    hdr = fits.getheader(img_sim_dir + '5deg_Y106_0_1.fits')

    # ---------------
    # Loop over sne to insert
    for i in range(NUM_SNE):

        # Decide some random mag for the SN and get counts
        # Position of SN
        # and adding to image
        # Get SN coords # random
        if NUM_SNE == 1:
            snmag = SNMAG_CENTRAL

            r = 2048
            c = 2048
        else:
            snmag = np.random.normal(loc=SNMAG_CENTRAL, scale=0.01, size=None)

            r = np.random.randint(low=110, high=3985, size=None)
            c = np.random.randint(low=110, high=3985, size=None)

        sncounts = get_counts(snmag)

        if MODEL == 'GAUSS':
            
            # Put the SN in the center of the image
            gaussfunc.x_mean = c
            gaussfunc.y_mean = r

            x, y = np.meshgrid(np.arange(NPIX), np.arange(NPIX))

            full_img += gaussfunc(x, y)

            # Also ensure that the number of pixels in 
            # manual segmap above threshold sum up to required
            # counts. 
            # See Russell's pyLINEAR notebooks for creating segmap
            # create a segmentation map from this image
            threshold = 0.2  # threshold to apply to the image
            good = full_img > threshold  # these pixels belong to a source
            segmap = label(good)  # now these pixels have unique segmentation IDs

            print('Total good pixels:', len(np.where(good)[0]))
            actual_counts = np.sum(full_img[good])
            print('Total counts within good pixels:', actual_counts)
            print('Required counts:', sncounts)
            scale_fac = sncounts / actual_counts
            # with np.printoptions(precision=2):
            #     print(full_img[good])
            
            print('Scaling factor:', scale_fac)
            full_img *= scale_fac

            print('New scaled counts:', np.sum(full_img[good]))

            # with np.printoptions(precision=4):
            #     print(full_img[good])

            # print(segmap[2048-10:2048+10, 
            #              2048-10:2048+10])

            # fig = plt.figure(figsize=(7, 7))
            # ax = fig.add_subplot(111)
            # ax.imshow(full_img[2048-25:2048+25, 
            #                    2048-25:2048+25], origin='lower')
            # plt.show()

            # ------- Now save segmap
            fits.writeto(savedir + 'test_segmap.fits',
                         segmap, header=hdr, overwrite=True)

        else:
            # Scale SN and add to full image
            scale_fac = sncounts / REF_COUNTS
            sn_img = ref_data * scale_fac

            full_img[r-SIZE:r+SIZE, c-SIZE:c+SIZE] += sn_img

            print(i+1, '  ', r, '  ', c, '  ', '{:.3f}'.format(snmag))

            insert_cat[i][0] = r
            insert_cat[i][1] = c
            insert_cat[i][2] = snmag

    # Save the file
    savefile = savedir + 'shortsim_image.fits'

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
            if NUM_SNE == 1:
                m = SNMAG_CENTRAL
            else:
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
        flthdu = fits.open(flt)

        sci = flthdu[1].data

        print('Data range:', np.min(sci), '  ', np.max(sci))

        size = sci.shape

        # Add sky and dark
        signal = (sci + SKY + DARK)

        # Scale by exptime
        signal = signal * EXPTIME

        variance = signal + READNOISE**2
        sigma = np.sqrt(variance)

        # Photometric variation
        new_sig = np.random.normal(loc=signal, 
                                   scale=sigma, size=size)

        # Subtract background and go back to e/s
        final_sig = (new_sig / EXPTIME) - SKY

        # Save
        # 0th extension with just the header
        noised_hdul = fits.HDUList()
        noised_hdul.append(fits.PrimaryHDU(header=flthdu[0].header))
        # 1st extension with the SCI data
        noised_hdul.append(fits.ImageHDU(data=final_sig, 
                                         header=flthdu[1].header))
        # 2nd extension with the ERR
        noised_hdul.append(fits.ImageHDU(data=sigma/EXPTIME, 
                                         header=flthdu[2].header))
        # 3rd extension with the DQ Array
        noised_hdul.append(fits.ImageHDU(data=flthdu[3].data, 
                                         header=flthdu[3].header))

        fltname = flt.replace('_flt', '_flt_noised')
        noised_hdul.writeto(fltname, overwrite=True)

        # Close open HDU
        flthdu.close()

    # Generate FLT LST and start extraction
    with open(fltlst, 'w') as fh:
        hdr1 = "# Path to each flt image" + "\n"
        hdr2 = "# This has to be a simulated or " + \
               "observed dispersed image" + "\n"

        fh.write(hdr1)
        fh.write(hdr2)

        for i in range(3):
            flt = savedir + 'shortsim' + str(i+1) + '_flt_noised.fits'
            fh.write('\n' + flt)

    print('Noised FLTs and written FLT LST.')

    # ---------------------- Extraction
    print('Starting extraction...')

    tablespath = savedir + 'tables/'

    grisms = pylinear.grism.GrismCollection(fltlst, observed=True)
    tabulate = pylinear.modules.Tabulate('pdt', 
                                         path=tablespath, ncpu=0)
    tabulate.run(grisms, sources, BEAM) 
    
    extraction_parameters = grisms.get_default_extraction()
    
    extpar_fmt = 'Default parameters: range = {lamb0}, {lamb1} A,' + \
                 ' sampling = {dlamb} A'
    print(extpar_fmt.format(**extraction_parameters))
    
    # Set extraction params
    sources.update_extraction_parameters(**extraction_parameters)
    method = 'golden'  # golden, grid, or single
    extroot = 'shortsim_'
    logdamp = [-6, -1, 0.1]

    pylinear.modules.extract.extract1d(grisms, sources, BEAM, logdamp, 
                                       method, extroot, tablespath, 
                                       inverter='lsqr', ncpu=6, 
                                       group=False)

    return None


if __name__ == '__main__':

    # ======================
    # Insert SNe and assign the SEDs
    insert_sne_getsedlst_shortsim()

    # Run SExtractor to generate segmap
    if MODEL == 'REFSTAR':
        subprocess.run(['sex', 'shortsim_image.fits', 
                        '-c', 'default_config.txt'], 
                       check=True)

    # Run pyLINEAR
    run_pylinear_shortsim()

    sys.exit(0)
