import numpy as np
from astropy.io import fits
from astropy.modeling import models
from skimage.morphology import label

import copy
import os
import sys
import subprocess

# import pylinear

from gen_sed_lst import get_sn_z, get_sn_spec_path
from run_pylinear import noise_img_save

datadir = os.getcwd() + '/' + 'short_sim_test/'
if not os.path.isdir(datadir):
    os.mkdir(datadir)

# Params of sim
NPIX = 4096
NUM_SNE = 1
REF_COUNTS = 13696.77
SIZE = 50  # half of cutout
SNMAG_CENTRAL = 24.5
ZP = 26.264  # For WFC3/F105W and for WFI/F106
PIXSCL = 0.108  # arcsec per pix
MODEL = 'GAUSS'  # model to use for fake SNe; GAUSS or REFSTAR

rollangles = [0.0, 70.0, 140.0]

MAGLIM = 29.0
BEAM = '+1'
EXPTIME = 3600  # seconds per FLT file; we have 3 exposures

# Get a header with WCS
extdir = '/Volumes/Joshi_external_HDD/Roman/'
roman_sims_seds = extdir + "roman_slitless_sims_seds/"
img_sim_dir = extdir + 'roman_direct_sims/sims2021/K_5degimages_part1/'
dirimg_hdr = fits.getheader(img_sim_dir + '5deg_Y106_0_1.fits')

obs_ra = float(dirimg_hdr['CRVAL1'])
obs_dec = float(dirimg_hdr['CRVAL2'])

print('Obs coords:', obs_ra, obs_dec)

# Noise budget
# See the prism info file from Jeff Kruk
# for all these numbers.
# Readnoise is effective readnoise rate
# Assumed average 1.1 factor for Zodi
BCK_ZODIACAL = 1.047  # e/pix/sec
BCK_THERMAL = 0.0637249  # e/pix/sec
DARK = 0.005  # e/pix/sec
READNOISE = 15  # electrons RMS
# Effective sky
SKY = BCK_ZODIACAL + BCK_THERMAL


# Change to working directory
os.chdir(datadir)


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
        ref_cutout_path = datadir + 'ref_cutout_psf.fits'
        ref_data = fits.getdata(ref_cutout_path)
    elif MODEL == 'GAUSS':
        # Generate Gaussian model from astropy
        # See Russell's simulating images notebook

        # Create Gaussian function model
        # stddev in pixels
        gaussfunc = models.Gaussian2D(x_stddev=3, y_stddev=3)

    # Empty array to store inserted sne properties
    insert_cat = np.zeros((NUM_SNE, 3))

    # ---------------
    # Loop over sne to insert
    for i in range(NUM_SNE):

        print('\nWorking on SN:', i+1)

        # Decide some random mag for the SN and get counts
        # Position of SN
        # and adding to image
        # Get SN coords # random
        if NUM_SNE == 1:
            snmag = SNMAG_CENTRAL

            r = 2048
            c = 2048
        else:
            snmag = np.random.normal(loc=SNMAG_CENTRAL, scale=0.2, size=None)

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
            threshold = 0.4  # threshold to apply to the image
            good = full_img > threshold  # these pixels belong to a source
            segmap = label(good)  # now these pixels have unique SegIDs

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
    savefile = datadir + 'shortsim_image.fits'

    # Create a header (pyLINEAR needs WCS)
    hdr = dirimg_hdr

    ihdul = fits.HDUList()
    ext_sci = fits.ImageHDU(data=full_img, header=hdr, name='SCI')
    ihdul.append(ext_sci)
    ext_err = fits.ImageHDU(data=np.sqrt(full_img), header=hdr, name='ERR')
    ihdul.append(ext_err)
    ext_dq = fits.ImageHDU(data=np.ones(full_img.shape),
                           header=hdr, name='DQ')
    ihdul.append(ext_dq)
    ihdul.writeto(savefile, overwrite=True)

    # ------- Now save segmap
    shdul = fits.HDUList()
    ext1 = fits.ImageHDU(data=segmap, header=hdr, name='SCI')
    shdul.append(ext1)
    shdul.writeto(datadir + 'test_segmap.fits', overwrite=True)

    # =====================================
    # Create SED LST file
    sedlst_filename = datadir + 'sed_shortsim.lst'

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
            # spec_path = gen_delta_comb_spec()

            # ------------ Write to file
            fh.write(str(j+1) + " " + spec_path + "\n")

    return None


def gen_delta_comb_spec(plot=False, scalefac=1e-17):

    # define an overall wav grid
    lamgrid = np.arange(7000, 20000)

    # define the spacing between the comb "teeth"
    stepsize = 80.0  # angstroms

    spec = np.zeros(len(lamgrid))

    for i in range(len(lamgrid)):
        current_wav = lamgrid[i]
        if (current_wav % stepsize) == 0:
            spec[i] = 1.0

    # Test figure
    if plot:
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        ax.plot(lamgrid, spec, color='k', lw=1.2)
        plt.show()

    # Scale to some sensible flam units
    spec *= scalefac

    # Now save
    # This is to match the behaviour of the other funcs
    # that generate spectra.
    combspec_file = roman_sims_seds + 'comb_spec.txt'
    with open(combspec_file, 'w') as fh:
        fh.write("#  lam  flux")
        fh.write("\n")

        for j in range(len(lamgrid)):
            fh.write("{:.2f}".format(lamgrid[j]) + " " + str(spec[j]))
            fh.write("\n")

    return combspec_file


def run_pylinear_shortsim():

    segfile = datadir + 'test_segmap.fits'
    obslst = datadir + 'obs_shortsim.lst'
    wcslst = datadir + 'wcs_shortsim.lst'
    sedlst = datadir + 'sed_shortsim.lst'
    fltlst = datadir + 'flt_shortsim.lst'
    '''
    # ---------------------- Get sources
    sources = pylinear.source.SourceCollection(segfile, obslst,
                                               detindex=0,
                                               maglim=MAGLIM)

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
    '''
    # ---------------------- Add noise
    for f in range(3):
        flt = datadir + 'shortsim' + str(f+1) + '_flt.fits'
        flthdu = fits.open(flt)

        sci = flthdu[1].data
        size = sci.shape
        print('Data range:', np.min(sci), '  ', np.max(sci))

        # make a copy of the sci img before sending it off
        # to be modified in two different ways. If this copy
        # isn't made then the first func directly modifies the
        # sci data and the second noising actually does it twice.
        sci_copy = copy.deepcopy(sci)

        # ---- WITH STIPS
        # This is very similar to how it is done in STIPS
        # We need this to send to IPAC but for pyLINEAR internal
        # purposes we will still use the older method for now.
        noise_dict = {'filename': os.path.basename(flt),
                      'filesuffix': '_noised_stips',
                      'exptime': EXPTIME,
                      'oldhdr': flthdu[1].header,
                      'sky': SKY,
                      'dark': DARK,
                      'readnoise': READNOISE}

        noise_img_save(sci_copy, noise_dict)
        print("Noised 2D FLT images saved.")

        # ---- MANUAL WAY
        # Add sky and dark
        signal = (sci + SKY + DARK)

        # Scale by exptime
        signal = signal * EXPTIME

        variance = signal + READNOISE**2
        sigma = np.sqrt(variance)

        # Photometric variation
        # This resulting image is now in electrons
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

        sys.exit(0)

    sys.exit(0)

    # Generate FLT LST and start extraction
    with open(fltlst, 'w') as fh:
        hdr1 = "# Path to each flt image" + "\n"
        hdr2 = "# This has to be a simulated or " + \
               "observed dispersed image" + "\n"

        fh.write(hdr1)
        fh.write(hdr2)

        for i in range(3):
            flt = datadir + 'shortsim' + str(i+1) + '_flt_noised.fits'
            fh.write('\n' + flt)

    print('Noised FLTs and written FLT LST.')

    # ---------------------- Extraction
    print('Starting extraction...')

    tablespath = datadir + 'tables/'

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


def create_obs_wcs_lst():

    # -------- OBS
    obs_filt = 'hst_wfc3_f105w'
    obslst_file = datadir + 'obs_shortsim.lst'
    imgfile = datadir + 'shortsim_image.fits'

    with open(obslst_file, 'w') as fho:
        fho.write('# Image File name' + '\n')
        fho.write('# Observing band' + '\n')

        fho.write('\n' + imgfile + '  ' + obs_filt)

    # -------- WCS
    wcslst_file = datadir + 'wcs_shortsim.lst'

    with open(wcslst_file, 'w') as fhw:

        hdr = ('# TELESCOPE = Roman' + '\n'
               '# INSTRUMENT = WFI' + '\n'
               '# DETECTOR = WFI' + '\n'
               '# GRISM = P127' + '\n'
               '# BLOCKING = ' + '\n')

        fhw.write(hdr + '\n')

        for r in range(len(rollangles)):
            obs_roll = rollangles[r]

            fhw.write('shortsim' + str(r+1))
            fhw.write('  ' + str(obs_ra))
            fhw.write('  ' + str(obs_dec))
            fhw.write('  ' + str(obs_roll))
            fhw.write('  P127' + '\n')

    return None


if __name__ == '__main__':
    # ======================
    # Insert SNe and assign the SEDs
    insert_sne_getsedlst_shortsim()
    sys.exit(0)

    # Create required lists
    create_obs_wcs_lst()

    # Run SExtractor to generate segmap
    if MODEL == 'REFSTAR':
        subprocess.run(['sex', 'shortsim_image.fits',
                        '-c', 'default_config.txt'],
                       check=True)

    # Run pyLINEAR
    run_pylinear_shortsim()

    sys.exit(0)
