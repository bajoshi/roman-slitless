import numpy as np
from astropy.io import fits

import yaml
import os
import sys
import socket
import subprocess
from pprint import pprint

# -----------------
if 'plffsn2' in socket.gethostname():
    extdir = '/astro/ffsn/Joshi/'
    modeldir = extdir + 'bc03_output_dir/'
    
    roman_sims_seds = extdir + 'roman_slitless_sims_seds/'
    pylinear_lst_dir = extdir + 'pylinear_lst_files/'
    roman_direct_dir = extdir + 'roman_direct_sims/sims2021/'

    roman_slitless_dir = extdir + "GitHub/roman-slitless/"
    fitting_utils = roman_slitless_dir + "fitting_pipeline/utils/"

else:
    extdir = '/Volumes/Joshi_external_HDD/Roman/'
    modeldir = extdir + 'bc03_output_dir/m62/'
    
    roman_sims_seds = extdir + "roman_slitless_sims_seds/"
    pylinear_lst_dir = extdir + "pylinear_lst_files/"
    roman_direct_dir = extdir + 'roman_direct_sims/sims2021/'

    home = os.getenv("HOME")
    roman_slitless_dir = home + "/Documents/GitHub/roman-slitless/"
    fitting_utils = roman_slitless_dir + "fitting_pipeline/utils/"

#assert os.path.isdir(modeldir)
#assert os.path.isdir(roman_sims_seds)
#assert os.path.isdir(pylinear_lst_dir)
#assert os.path.isdir(roman_direct_dir)
# -----------------

######## CUSTOM IMPORTS

sys.path.append(fitting_utils)
import proper_and_lum_dist as cosmo
import dust_utils as du
from make_model_dirimg import gen_model_img
from ref_cutout import gen_reference_cutout
from get_insertion_coords import get_insertion_coords
from lc_plot import read_lc
from get_sn_mag import get_sn_magF106


################## END IMPORTS

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

###############################################################################
###############################################################################
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
img_sim_dir = roman_direct_dir + 'K_5degimages_part1/'

dir_img_name = (img_sim_dir + img_basename + img_filt + 
               str(pointing) + str(detector) + '.fits')
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
for i in range(insert_num):

    # Pick a phase with equal probability from
    # [-7, -6, -5, -4, -3]
    phase = np.random.randint(low=-7, high=-2)

    # Pick a redshift 
    # Equal probability for any z within [0.5, 3.0]
    redshift = np.random.choice(z_arr)

    # Now figure out what F106 magnitude this SN will be
    snmag = get_sn_magF106(phase, redshift)

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

# Save all needed quantities to a numpy array
dat = np.c_[x_ins, y_ins, phase_chosen, redshift_chosen, sn_magnitudes]

print(dat.shape)

np.savetxt('inserted_sn_props.txt', dat, 
    fmt=, 
    header='xc  yc  phase  redshift  magF106')

assign_spectra()

check_simprep()
run_sim(visit=1)

###############################################################################
###############################################################################
# Visit 2 and future visits
# ---------------
# Function to update the inserted SNe mags according to LC evolution
def update_sn_visit_mag(visit, sn_prop):

    for i in range(insert_num):
    
        sn_z = sn_prop['redshift'][i]
        starting_phase = sn_prop['phase'][i]
    
        cosmic_time_dilation_rest_frame = 5 / (1 + sn_z)
    
        # Get current phase in rest frame
        current_phase = starting_phase + cosmic_time_dilation_rest_frame * (visit - 1)
    
        # Get new apparent magnitude at this phase
        snmag = get_sn_magF106(current_phase, sn_z)
    
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

def update_sn_visit_spectra():

    return None

# ---------------
# Read in light curves to add in SN luminosity evolution
lc_b_phase, lc_b_absmag = read_lc('B')

# Also read in properties of inserted SNe
sn_prop = np.genfromtxt('inserted_sn_props.txt', dtype=None, 
    names=True, encoding='ascii')












