import numpy as np
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
import pylinear

import yaml
import os
import sys
import socket
import subprocess
import pdb

from pprint import pprint
from tqdm import tqdm

# -----------------
if 'plffsn2' in socket.gethostname():
    extdir = '/astro/ffsn/Joshi/'
    roman_slitless_dir = extdir + "GitHub/roman-slitless/"
else:
    extdir = '/Volumes/Joshi_external_HDD/Roman/'
    home = os.getenv("HOME")
    roman_slitless_dir = home + "/Documents/GitHub/roman-slitless/"
    
pylinear_lst_dir = extdir + "pylinear_lst_files/"
roman_direct_dir = extdir + 'roman_direct_sims/sims2021/'
result_dir = extdir + 'survey_sim/'

fitting_utils = roman_slitless_dir + "fitting_pipeline/utils/"
survey_dir = roman_slitless_dir + "survey/"

# One more step to defining the directory with the image sims
img_sim_dir = roman_direct_dir + 'K_5degimages_part1/'

assert os.path.isdir(pylinear_lst_dir)
assert os.path.isdir(img_sim_dir)
assert os.path.isdir(fitting_utils)
assert os.path.isdir(result_dir)
# -----------------

######## CUSTOM IMPORTS

sys.path.append(fitting_utils)
from make_model_dirimg import gen_model_img
from ref_cutout import gen_reference_cutout
from get_insertion_coords import get_insertion_coords
from get_sn_mag import get_sn_mag_F106
from get_sn_spec_path import get_sn_spec_path
from get_gal_spec_path import get_gal_spec_path

################## END IMPORTS

# Colored text on terminal
# This class came from stackoverflow
# SEE:
# https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-python
class bcolors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# Assign spectra to all detected objects in an image
def assign_spectra(dir_img_name, sn_prop, visit):
    """
    This function will run sextractor to detect
    all objects in a given image, and assign 
    either a galaxy or SN Ia spectrum to each 
    object depending on the object type.
    """

    # Run Sextractor ONLY ON THE FIRST VISIT
    # for visits beyond the first one we will 
    # simply read in the catalog from the first visit
    cat_filename = dir_img_name.replace('.fits', '.cat')
    if visit == 1:
        # See notes on sextractor args for subprocess
        # in gen_sed_lst.py
        # Change directory to images directory
        os.chdir(img_sim_dir)

        checkimage = dir_img_name.replace('.fits', '_segmap.fits')
            
        sextractor = subprocess.run(['sex', os.path.basename(dir_img_name), 
            '-c', 'roman_sims_sextractor_config.txt', 
            '-CATALOG_NAME', os.path.basename(cat_filename), 
            '-CHECKIMAGE_NAME', os.path.basename(checkimage)], check=True)
            
        # Go back to roman-slitless directory
        os.chdir(survey_dir)

    # Now read in the SExtractor catalog 
    cat_header = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'ALPHA_J2000', 'DELTA_J2000', 
    'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_AUTO', 'MAGERR_AUTO', 'FLUX_RADIUS', 
    'FWHM_IMAGE']
    cat = np.genfromtxt(cat_filename, dtype=None, 
        names=cat_header, encoding='ascii')

    # Assign spectra 
    # First construct the path of the SED.lst file
    ibase = os.path.basename(dir_img_name)  # file base name
    ibase = ibase.replace('.fits','')  # remove extension
    isplt = ibase.split('_')  # split by underscores and rejoin
    img_suffix = isplt[1] + '_' + isplt[2] + '_' + isplt[3]

    sedlst_filename = pylinear_lst_dir + 'sed_' + img_suffix + '.lst'

    # Read in previous SED LST file if visit > 1
    if visit > 1:
        # Get all lines
        alllines = open(sedlst_filename, 'r').readlines()

    # On the first visit also  record all the randomly
    # chosen Av. The same values will continue to be 
    # used afterwards.
    av_list = np.zeros(len(sn_prop))

    # Now loop over all objects
    with open(sedlst_filename, 'w') as fh:

        # Write header
        fh.write("# 1: SEGMENTATION ID" + "\n")
        fh.write("# 2: SED FILE" + "\n")

        for i in tqdm(range(len(cat)), desc="Object SegID"):

            current_sextractor_id = int(cat['NUMBER'][i])

            # Check if the object is a SN
            obj_x = cat['X_IMAGE'][i]
            obj_y = cat['Y_IMAGE'][i]

            # If there is a SN within 5 pix of this object 
            # then assign this object a SN spectrum 
            # otherwise assign a galaxy spectrum
            sn_idx = np.where((abs(sn_prop['xc'] - obj_x) < 5) & \
                              (abs(sn_prop['yc'] - obj_y) < 5))[0]
            
            if sn_idx.size:
                sn_idx = int(sn_idx)
                sn_z = sn_prop['redshift'][sn_idx]
                starting_phase = sn_prop['phase'][sn_idx]
                cosmic_time_dilation_rest_frame = 5 / (1 + sn_z)

                # For visit 1 the current phase is equal 
                # to the starting phase
                current_phase = starting_phase + \
                                cosmic_time_dilation_rest_frame * (visit - 1)
                # Ensure that the phase is an integer
                current_phase = int(current_phase)
                if visit == 1:
                    sn_spec_path, _r, _d, _av = \
                    get_sn_spec_path(cosmo, sn_z, day_chosen=current_phase)
                    
                    av_list[sn_idx] = _av

                elif visit > 1:
                    sn_av = sn_prop['Av'][sn_idx]
                    sn_spec_path, _r, _d, _av = \
                    get_sn_spec_path(cosmo, sn_z, 
                        day_chosen=current_phase, chosen_av=sn_av)

                # Write to file
                _w = str(current_sextractor_id) + ' ' + sn_spec_path + '\n'
                fh.write(_w)
            else:
                if visit == 1:
                    # Pick a random redshift for this galaxy
                    # the z_arr variable is global above so
                    # we can simply use that again
                    random_z = np.random.choice(z_arr)
                    gal_spec_path = get_gal_spec_path(cosmo, random_z)

                    # Write to file
                    _w = str(current_sextractor_id) + ' ' + gal_spec_path + '\n'
                    fh.write(_w)
                elif visit > 1:
                    fh.write(alllines[i+2])  # shifted by two header lines

    # Finally if it is the first visit update the sn_prop file
    # with the dust extinction applied to the SN spectrum
    if visit == 1:
        # Put the old data in arrays first
        snx  = sn_prop['xc']
        sny  = sn_prop['yc']
        snph = sn_prop['phase']
        snz  = sn_prop['redshift']
        snm  = sn_prop['magF106']

        new_dat = np.c_[snx, sny, snph, snz, snm, av_list]

        np.savetxt('inserted_sn_props.txt', new_dat, 
            fmt=['%.3f', '%.3f', '%d', '%.3f', '%.2f', '%.3f'], 
            header='xc  yc  phase  redshift  magF106  Av')

    return None


# Function to update the inserted SNe mags according to LC evolution
def update_sn_visit_mag(visit, sn_prop):

    for i in range(insert_num):
    
        sn_z = sn_prop['redshift'][i]
        starting_phase = sn_prop['phase'][i]
    
        cosmic_time_dilation_rest_frame = 5 / (1 + sn_z)
    
        # Get current phase in rest frame
        current_phase = starting_phase + \
                        cosmic_time_dilation_rest_frame * (visit - 1)
    
        # Get new apparent magnitude at this phase
        snmag = get_sn_mag_F106(current_phase, sn_z, fitting_utils)
    
        # Add the reference cutout again at this new magnitude
        delta_m = ref_mag - snmag
        sncounts = ref_counts * (1 / 10**(-0.4*delta_m) )
    
        scale_fac = sncounts / ref_counts
        new_cutout = ref_data * scale_fac
    
        # Now get coords
        xi = sn_prop['xc'][i]
        yi = sn_prop['yc'][i]
    
        r = yi
        c = xi
    
        # Add in the new SN
        model_img[r-s:r+s, c-s:c+s] = model_img[r-s:r+s, c-s:c+s] + new_cutout

    return None


def create_lst_files(dir_img_name, visit, config):

    # First construct the img suffix
    ibase = os.path.basename(dir_img_name)  # file base name
    ibase = ibase.replace('.fits','')  # remove extension
    isplt = ibase.split('_')  # split by underscores and rejoin
    img_suffix = isplt[1] + '_' + isplt[2] + '_' + isplt[3]

    # Paths to each file
    fltlst_deep = pylinear_lst_dir + 'flt_' + img_suffix + '_deep.lst'
    fltlst_wide = pylinear_lst_dir + 'flt_' + img_suffix + '_wide.lst'
    obslst = pylinear_lst_dir + 'obs_' + img_suffix + '.lst'
    wcslst = pylinear_lst_dir + 'wcs_' + img_suffix + '.lst'
    sedlst = pylinear_lst_dir + 'sed_' + img_suffix + '.lst'

    # Some other stuff needed from config
    disp_elem = config['disp_elem']
    ra_cen_fmt = config['ra_cen']
    dec_cen_fmt = config['dec_cen']

    # ---------- OBSLST
    with open(obslst, 'w') as fh:
        fh.write("# Image File name" + "\n")
        fh.write("# Observing band" + "\n")

        fh.write("\n" + dir_img_name + '  ' + 'hst_wfc3_f105w')

    # ---------- WCSLST
    with open(wcslst, 'w') as fh:
        fh.write("# TELESCOPE = Roman" + "\n")
        fh.write("# INSTRUMENT = WFI" + "\n")
        fh.write("# DETECTOR = WFI" + "\n")
        fh.write("# GRISM = " + disp_elem + "\n")
        fh.write("# BLOCKING = " + "\n")

        # Get roll angle dependent on vist
        roll_angle = 5 * (visit - 1)

        str_to_write = "\n" + 'romanprism_hltds_' + img_suffix + \
        '  ' + ra_cen_fmt + '  ' + dec_cen_fmt + \
        '  ' + str(roll_angle) + '  ' + disp_elem
            
        fh.write(str_to_write)

    # ---------- FLTLST
    # One for each exposure time -- deep and wide 
    for e in [3600, 900]:

        if e == 3600:
            fltlst = fltlst_deep
        else:
            fltlst = fltlst_wide

        with open(fltlst, 'w') as fh:
            h1 = "# Path to each flt image"
            h2 = "# This has to be a simulated or observed dispersed image"
            fh.write(h1 + "\n")
            fh.write(h2 + "\n")

            str_to_write = "\n" + result_dir + 'romanprism_hltds_' + \
                    img_suffix + '_' + str(e) + 's_flt.fits'

            fh.write(str_to_write)

    # ---------- SEDLST
    # Doesn't need anything here. SED LST is created and edited 
    # by the assign_spectra function above.

    # Return all paths
    return fltlst_deep, fltlst_wide, obslst, wcslst, sedlst


def run_sim(dir_img_name, visit, config):

    # ------------------
    # Create LST files
    fltlst_deep, fltlst_wide, obslst, wcslst, sedlst = \
    create_lst_files(dir_img_name, visit, config)

    # ------------------
    # Now run sim
    # Change directory to where the simulation results will go
    os.chdir(result_dir)

    # Get pylinear config params
    beam = config['pylin']['beam']
    maglim = config['pylin']['maglim']

    segfile = dir_img_name.replace('.fits', '_segmap.fits')

    # ---------------------- Get sources
    sources = pylinear.source.SourceCollection(segfile, obslst, 
        detindex=0, maglim=maglim)

    # Set up
    grisms = pylinear.grism.GrismCollection(wcslst, observed=False)
    tabulate = pylinear.modules.Tabulate('pdt', ncpu=0)
    tabnames = tabulate.run(grisms, sources, beam)

    ## ---------------------- Simulate
    simulate = pylinear.modules.Simulate(sedlst, gzip=False, ncpu=0)
    fltnames = simulate.run(grisms, sources, beam)
    print(f'{bcolors.CYAN}', 'Simulation done.', f'{bcolors.ENDC}')

    # ---------------------- Noise 2D dispersed image
    npix = config['pylin']['npix']
    sky = config['pylin']['sky']
    dark = config['pylin']['dark']
    rdnoise = config['pylin']['rdnoise']

    for exptime in [3600, 900]:

        oldf = 'romanprism_hltds_' + img_suffix + '_flt.fits'

        with fits.open(oldf) as hdul:
            sci = hdul[('SCI',1)].data
            size = sci.shape

            # update the science extension with sky background and dark current
            signal = (sci + sky + dark)

            # Multiply the science image with the exptime
            # sci image originally in electrons/s
            signal = signal * exptime  # this is now in electrons
    
            # Randomly vary signal about its mean. Assuming Gaussian distribution
            # first get the uncertainty
            variance = signal + read**2
            sigma = np.sqrt(variance)
            new_sig = np.random.normal(loc=signal, scale=sigma, size=size)

            # now divide by the exptime and subtract the sky again 
            # to get back to e/s. LINEAR expects a background subtracted image
            final_sig = (new_sig / exptime) - sky
    
            # Stop if you find nans
            nan_idx = np.where(np.isnan(final_sig))
            nan_idx = np.asarray(nan_idx)
            if nan_idx.size:
                logger.critical("Found NaNs. Resolve this issue first. Exiting.")
                sys.exit(1)

            # Assign updated sci image to the first [SCI] extension
            hdul[('SCI',1)].data = final_sig
    
            # update the uncertainty extension with the sigma
            err = np.sqrt(signal) / exptime
    
            hdul[('ERR',1)].data = err
    
            # now write to a new file name
            newfilename = oldf.replace('_flt', '_' + str(exptime) + 's' + '_flt')
            hdul.writeto(newfilename, overwrite=True)

        # ------------------
        # Extract
        print(f'{bcolors.CYAN}', 'Beginning extraction...', f'{bcolors.ENDC}')
        if e == 3600:
            fltlst = fltlst_deep
        else:
            fltlst = fltlst_wide

        grisms = pylinear.grism.GrismCollection(fltlst, observed=True)
        tabulate = pylinear.modules.Tabulate('pdt', path=tablespath, ncpu=0)
        tabnames = tabulate.run(grisms, sources, beam)
    
        extraction_parameters = grisms.get_default_extraction()
    
        extpar_fmt = \
        'Default parameters: range = {lamb0}, {lamb1} A, sampling = {dlamb} A'
        print(extpar_fmt.format(**extraction_parameters))
    
        # Set extraction params
        sources.update_extraction_parameters(**extraction_parameters)
        method = 'golden'  # golden, grid, or single
        extroot = 'romanprism_hltds_' + img_suffix + '_' + str(exptime) + 's' \
                  + '_visit' + str(visit)
        logdamp = [-6, -1, 0.1]
    
        print(f'{bcolors.CYAN}', "Extracting...", f'{bcolors.ENDC}')
        pylinear.modules.extract.extract1d(grisms, sources, beam, logdamp, 
            method, extroot, tablespath, 
            inverter='lsqr', ncpu=1, group=False)

    return None


################## END FUNCTION DEFS

##### TO DO LIST
print(f'{bcolors.WARNING}')
print('TODO LIST:')
print('0. Ensure that directory structure is only ')
print('   setup and checked once (in this code).')
print('1. Convert YAML config dict to a class object.')
print('2. Write the colors class as a util and import.')
print(f'{bcolors.ENDC}')

############################## 
# Get configuration
config_flname = 'survey_config.yaml'
with open(config_flname, 'r') as fh:
    cfg = yaml.safe_load(fh)

print('Received the following configuration for the simulation:')
pprint(cfg)

################################################################################
################################################################################
# Insert SNe in visit 1
# ---------------
# Read in the reference image of the star from 
# Y106_0_6 at 72.1175280, -53.5739388 (~16th mag)
ref_cutout_path = img_sim_dir + 'ref_cutout_psf.fits'
if not os.path.isfile(ref_cutout_path):
    gen_reference_cutout()

ref_data = fits.getdata(ref_cutout_path)

ref_mag = cfg['insert']['ref_mag']
ref_counts = cfg['insert']['ref_counts']

s = cfg['insert']['ref_size']  # reference cutout size

# For now we're simulating all SNE on a single Roman detector
# so no need to loop over all 18 or multiple pointings.
# Arrays to loop over
#pointings = np.arange(0, 1)
#detectors = np.arange(1, 19, 1)

pointing = cfg['pointing']
detector = cfg['detector']

# ---------------
# Determine number of SNe to insert and open dir img
ins_low_lim = cfg['insert']['numlow']
ins_high_lim = cfg['insert']['numhigh']

insert_num = np.random.randint(low=ins_low_lim, high=ins_high_lim)

print(f'{bcolors.CYAN}')
print("Will insert " + str(insert_num) + " SNe in detector " + str(detector))
print(f'{bcolors.ENDC}')

# Open dir image
img_basename = cfg['img']['basename']
img_filt = cfg['img']['filt']

dir_img_name = (img_sim_dir + img_basename + img_filt + 
               str(pointing) + '_' + str(detector) + '.fits')
dir_hdu = fits.open(dir_img_name)

# ---------------
# Now scale image to get the image to counts per sec
# Scaling factor for direct images
# The difference here that appears in the power of 10
# is the difference between the ZP of the current img
# and what I think the correct ZP is i.e., the WFC3/F105W ZP.
img_zp = cfg['img']['img_zp']
wfc3_f105w_zp = cfg['img']['wfc3_f105w_zp']
dirimg_scaling = 10**(-0.4 * (img_zp - wfc3_f105w_zp))

cps_sci_arr = dir_hdu[1].data * dirimg_scaling
cps_hdr = dir_hdu[1].header
dir_hdu.close()

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

# Go back to survey directory
os.chdir(survey_dir)

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
back_scale = cfg['img']['back_scale']

model_img += np.random.normal(loc=0.0, scale=back_scale, size=model_img.shape)

# ---------------
# Get a list of x-y coords to insert SNe at
x_ins, y_ins = get_insertion_coords(insert_num)

# ---------------
# Choose phases and redshifts to insert
# Mags will be dependent on phase and z to be inserted
# Visit 2 and beyond will then simply evolve these inserted SNe

# Redshift array from which random redshift will be chosen
z_arr = np.arange(0.5, 3.01, 0.001)

phase_chosen = np.zeros(insert_num)
redshift_chosen = np.zeros(insert_num)
sn_magnitudes = np.zeros(insert_num)

# Loop over all SNe to be inserted
for i in tqdm(range(insert_num), desc='Inserting SNe'):

    # Pick a phase with equal probability from
    # [-7, -6, -5, -4, -3]
    phase = np.random.randint(low=-7, high=-2)

    # Pick a redshift 
    # Equal probability for any z within [0.5, 3.0]
    redshift = np.random.choice(z_arr)

    # Now figure out what F106 magnitude this SN will be
    snmag = get_sn_mag_F106(phase, redshift, fitting_utils)

    # Now scale reference
    delta_m = ref_mag - snmag
    sncounts = ref_counts * (1 / 10**(-0.4*delta_m) )

    scale_fac = sncounts / ref_counts
    new_cutout = ref_data * scale_fac

    # Now get coords
    xi = x_ins[i]
    yi = y_ins[i]

    r = yi
    c = xi

    # Add in the new SN
    model_img[r-s:r+s, c-s:c+s] = model_img[r-s:r+s, c-s:c+s] + new_cutout

    # Append chosen quantities to arrays to be saved
    phase_chosen[i] = phase
    redshift_chosen[i] = redshift
    sn_magnitudes[i] = snmag

# Save and check image with ds9 if needed
new_hdu = fits.PrimaryHDU(header=cps_hdr, data=model_img)
img_savefile = dir_img_name.replace('.fits', '_SNadded.fits')
new_hdu.writeto(img_savefile, overwrite=True)

# Save all needed quantities to a numpy array
dat = np.c_[x_ins, y_ins, phase_chosen, redshift_chosen, sn_magnitudes]

np.savetxt('inserted_sn_props.txt', dat, 
    fmt=['%.3f', '%.3f', '%d', '%.3f', '%.2f'], 
    header='xc  yc  phase  redshift  magF106')

print(f'{bcolors.CYAN}')
print('Inserted SNe and props file saved. Assigning spectra now...')
print(f'{bcolors.ENDC}')

# Read in the SN properties file just created
sn_prop = np.genfromtxt('inserted_sn_props.txt', 
    dtype=None, names=True, encoding='ascii')

assign_spectra(img_savefile, sn_prop, visit=1)

#check_simprep()

print(f'{bcolors.CYAN}')
print('Running visit 1 sim.')
print(f'{bcolors.ENDC}')

run_sim(img_savefile, visit=1, config=cfg)

sys.exit(0)

################################################################################
################################################################################
# Visit 2 and future visits
# ---------------
# Update image with new SN mags
update_sn_visit_mag(visit, sn_prop)
print('Ensure Av mag taken into account for dir img')

# ---------------
# Read in the updated SN properties file after the first visit
sn_prop = np.genfromtxt('inserted_sn_props.txt', 
    dtype=None, names=True, encoding='ascii')

assign_spectra(img_savefile, sn_prop, visit=2)

# ---------------












