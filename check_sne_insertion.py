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
num_sn_list = []
for i in range(3):
    s = pylinear_lst_dir + 'sed_Y106_' + pt + '_' + str(i+1) + '.lst'
    n = read_numsn(s)
    print(s, ' has ', n, 'SNe.')
    total_sne1 += n
    num_sn_list.append(n)

print('Total SNe in pointing: ', total_sne1)
print('-------'*5)

# ------------------------------------
# TEST 2:
# For the inserted SNe esure that SExtractor magnitude
# is same (or almost the same) as the inserted magnitude.
# SEt cat header
cat_header = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'ALPHA_J2000', 'DELTA_J2000', 
    'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_AUTO', 'MAGERR_AUTO', 'FLUX_RADIUS', 'FWHM_IMAGE']

# For this test you need to read in the SExtractor 
# catalog and the numpy file where the inserted mags
# are stored and compare the two.

sextractor_mags = []
inserted_mags = []

for i in range(3):

    # Read catalog
    cat_filename = roman_direct_dir + 'K_5degimages_part1/' + '5deg_Y106_' + pt + '_' + str(i+1) + '_SNadded.cat'
    cat = np.genfromtxt(cat_filename, dtype=None, names=cat_header, encoding='ascii')

    # Read in npy file
    ins_npy = np.load(cat_filename.replace('.cat', '.npy'))

    for j in range(len(ins_npy)):

        current_x = ins_npy[j][0]
        current_y = ins_npy[j][1]

        # Look for center within +- 4 pix
        cat_idx = np.where( (cat['X_IMAGE'] >= current_x - 4) & (cat['X_IMAGE'] <= current_x + 3) \
                          & (cat['Y_IMAGE'] >= current_y - 3) & (cat['Y_IMAGE'] <= current_y + 3))[0]

        #print(current_x, current_y, ' | ', cat['X_IMAGE'][cat_idx], cat['Y_IMAGE'][cat_idx])

        if cat_idx.size:
            sextractor_mags.append(float(cat['MAG_AUTO'][cat_idx]))
            inserted_mags.append(float(ins_npy[j][-1]))

# convert to numpy arrays
sextractor_mags = np.asarray(sextractor_mags)
inserted_mags = np.asarray(inserted_mags)

magdiff = sextractor_mags - inserted_mags

# plot
fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_ylabel('Mag. diff.', fontsize=14)
ax.set_xlabel('Inserted SN AB mag in F106', fontsize=14)

ax.scatter(inserted_mags, magdiff, s=7, color='k')
ax.axhline(y=0.0, ls='--', color='gray', lw=2.5)

fig.savefig(extdir + 'check_inserted_sn_mag.pdf', dpi=200, bbox_inches='tight')

fig.clear()
plt.close(fig)

# ------------------------------------
# TEST 3:
# Make sure that the inserted SNe follow cosmological dimming
# as expected. This test simply plots SN magnitude vs redshift.

all_sn_mags = []
all_sn_z = []

total_sne2 = 0

for i in range(3):

    # ------ Read the SED lst and the corresponding SExtractor catalog
    # Set filenames
    sed_filename = pylinear_lst_dir + 'sed_Y106_' + pt + '_' + str(i+1) + '.lst'
    cat_filename = roman_direct_dir + 'K_5degimages_part1/' + '5deg_Y106_' + pt + '_' + str(i+1) + '_SNadded.cat'

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
ax.set_ylabel('Distance modulus', fontsize=14)

# Plot dist mod vs z
absmag = -18.4  # in Y band # from Dhawan et al 2015
dist_mod = np.asarray(all_sn_mags) - absmag

ax.scatter(all_sn_z, dist_mod, s=7, color='k')

# Also plot apparent mag
axt = ax.twinx()
axt.scatter(all_sn_z, all_sn_mags, s=7, color='k')
axt.set_ylabel('Inserted SN AB mag in F106', fontsize=14)

fig.savefig(extdir + 'test_sn_insert_mag_z.pdf', dpi=200, bbox_inches='tight')

print('Testing finished.')


