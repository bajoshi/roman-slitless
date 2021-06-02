import pylinear

import os
import sys

# ---------------------------
home = os.getenv('HOME')
tablespath = home + '/Documents/roman_slitless_sims_results/tables/'
pylinear_lst_dir = home + '/Documents/GitHub/roman-slitless/pylinear_lst_files/'
roman_direct_dir = '/Volumes/Joshi_external_HDD/Roman/roman_direct_sims/sims2021/'

# ---------------------------
# Info for sim
segid_list = [228, 260, 322, 487, 493]  # segids to test region for
exptime = 3600
obsstr = ''
img_basename = '5deg_'
img_suffix = 'Y106_0_6'

maglim = 99.0

dir_img_part = 'part1'
img_sim_dir = roman_direct_dir + 'K_5degimages_' + dir_img_part + '/'

# ---------------------------
# LST and other files needed
# Must already exist from run_pylinear
segfile = img_sim_dir + img_basename + img_suffix + '_segmap.fits'
obslst = pylinear_lst_dir + 'obs_' + img_suffix + obsstr + '.lst'
fltlst = pylinear_lst_dir + 'flt_' + img_suffix + '_' + \
                     str(exptime) + 's' + obsstr + '.lst'

assert os.path.isfile(segfile)
assert os.path.isfile(obslst)
assert os.path.isfile(fltlst)

# ---------------------------
# Load in sources
sources = pylinear.source.SourceCollection(segfile, obslst, 
            detindex=0, maglim=maglim)

# Load in grisms for the sim to test
grisms = pylinear.grism.GrismCollection(fltlst, observed=True)

# ---------------------------
# Loop over grisms and gen regions for all segid    
    
for grism in grisms:
        
    print("Working on:", grism.dataset)
    reg_filename = img_sim_dir + grism.dataset + '_2dext.reg'
    with open(reg_filename, 'w') as fh:
        
        with pylinear.h5table.H5Table(grism.dataset, path=tablespath, mode='r') as h5:
            
            device = grism['SCA09']
            
            h5.open_table('SCA09', '+1', 'pdt')

            for segid in segid_list:
            
                odt = h5.load_from_file(sources[segid], '+1', 'odt')
                ddt = odt.decimate(device.naxis1, device.naxis2)
            
                region_text = ddt.region()
                region_text = region_text.replace('helvetica 12 bold', 'helvetica 10 bold')
        
                fh.write(region_text + '\n')

sys.exit(0)