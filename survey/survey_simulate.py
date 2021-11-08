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
    
    roman_sims_seds = "/Volumes/Joshi_external_HDD/Roman/roman_slitless_sims_seds/"
    pylinear_lst_dir = "/Volumes/Joshi_external_HDD/Roman/pylinear_lst_files/"
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

sys.path.append(roman_slitless_dir)
import insert_sne as ins

sys.path.append(fitting_utils)
import proper_and_lum_dist as cosmo
import dust_utils as du
from make_model_dirimg import gen_model_img
from ref_cutout import gen_reference_cutout

################## END IMPORTS

############################## 
# Get configuration
config_flname = 'survey_config.yaml'
with open(config_flname, 'r') as fh:
    cfg = yaml.safe_load(fh)

print('Received the following configuration for the simulation:')
pprint(cfg)

############################## 
# Insert SNe in visit 1
# ---------------
# Read in the reference image of the star from 
# Y106_0_6 at 72.1175280, -53.5739388 (~16th mag)
ref_cutout_path = img_sim_dir + 'ref_cutout_psf.fits'
if not os.path.isfile(ref_cutout_path):
    gen_reference_cutout()

ref_data = fits.getdata(ref_cutout_path)

# For now we're simulating all SNE on a single Roman detector
# so no need to loop over all 18 or multiple pointings.
# Arrays to loop over
#pointings = np.arange(0, 1)
#detectors = np.arange(1, 19, 1)
detector = cfg['']

############################## 
# Visit 2 and future visits







