import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import socket

# ------------------------------------
if 'plffsn2' in socket.gethostname():
    extdir = '/astro/ffsn/Joshi/'
    modeldir = extdir + 'bc03_output_dir/'
    
    roman_sims_seds = extdir + 'roman_slitless_sims_seds/'
    pylinear_lst_dir = extdir + 'pylinear_lst_files/'
    roman_direct_dir = extdir + 'roman_direct_sims/sims2021/'

else:
    extdir = '/Volumes/Joshi_external_HDD/Roman/'
    modeldir = extdir + 'bc03_output_dir/m62/'
    
    roman_sims_seds = "/Volumes/Joshi_external_HDD/Roman/roman_slitless_sims_seds/"
    pylinear_lst_dir = "/Volumes/Joshi_external_HDD/Roman/pylinear_lst_files/"
    roman_direct_dir = extdir + 'roman_direct_sims/sims2021/'

assert os.path.isdir(modeldir)
assert os.path.isdir(roman_sims_seds)
assert os.path.isdir(pylinear_lst_dir)
assert os.path.isdir(roman_direct_dir)
# ------------------------------------
# TEST 1:
# Make sure that there are "enough" SNe inserted in each detector.
# There should be greater than 10 SNe detected in each, typically
# although sometimes you might get unlucky and have less than 10.
# This is because most of the SNe inserted are quite faint and 
# although there are quite a few inserted, SExtractor won't detect all of them.

pt = '0'  # Enter the pointing you want to test

def read_numsn(sedlst):

    with open(sedlst, 'r') as sed_fh:
        all_sed_lines = sed_fh.readlines()
        num_sn = 0
        for l in all_sed_lines:
            if 'salt' in l:
                num_sn += 1
    
    return num_sn

total_sne1 = 0
for i in range(18):
    s = pylinear_lst_dir + 'sed_Y106_' + pt + '_' + str(i+1) + '.lst'
    n = read_numsn(s)
    print(s, ' has ', n, 'SNe.')
    total_sne1 += n

print('Total SNe in pointing: ', total_sne1)
print('-------'*5)

# ------------------------------------
# TEST 2:
# Make sure that the inserted SNe follow cosmological dimming
# as expected. This test simply plots SN magnitude vs redshift.

all_sn_mags = []
all_sn_z = []

total_sne2 = 0

for i in range(18):

    # ------ Read the SED lst and the corresponding SExtractor catalog
    # Set filenames
    sed_filename = pylinear_lst_dir + 'sed_Y106_' + pt + '_' + str(i+1) + '.lst'
    cat_filename = roman_direct_dir + 'K_5degimages_part1/' + '5deg_Y106_' + pt + '_' + str(i+1) + '_SNadded.cat'

    # SEt cat header
    cat_header = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'ALPHA_J2000', 'DELTA_J2000', 
        'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_AUTO', 'MAGERR_AUTO', 'FLUX_RADIUS', 'FWHM_IMAGE']

    # Read in the files
    sed = np.genfromtxt(sed_filename, dtype=None, 
        names=['SegID', 'sed_path'], encoding='ascii', skip_header=2)
    cat = np.genfromtxt(cat_filename, dtype=None, names=cat_header, encoding='ascii')

    assert len(sed) == len(cat)

    # ----- Now get each SN mag and z
    for j in range(len(sed)):
        pth = sed['sed_path'][j]
        if 'salt' in pth:
            salt_pth = pth.split('/')[-1].split('_')
            z = salt_pth[3][1:]
            z = float(z.replace('p', '.'))

            segid = sed['SegID'][j]
            segid_idx = np.where(cat['NUMBER'] == segid)[0]

            mag = float(cat['MAG_AUTO'][segid_idx])

            # append
            all_sn_mags.append(mag)
            all_sn_z.append(z)

            #print(segid, '  ', z, '  ', mag)

            total_sne2 += 1

print('Total SNe in pointing: ', total_sne2)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xlabel('Inserted SN z', fontsize=14)
ax.set_ylabel('Inserted SN AB mag in F106', fontsize=14)

ax.scatter(all_sn_z, all_sn_mags, s=3, color='k')

fig.savefig(extdir + 'test_sn_insert_mag_z.pdf', dpi=200, bbox_inches='tight')

print('Testing finished.')


