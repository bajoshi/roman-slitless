import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import os
import sys
import socket

home = os.getenv('HOME')

# ------------------------------------
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
for i in range(18):
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

for i in range(18):

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
"""
# TEST 3:
# This test is for ALL objects (galaxies included).
# This will test if the spectrum passed to pylinear when
# convolved with the filter curve gives the expected mag
# which we have from SExtractor.
def filter_conv(filter_wav, filter_thru, spec_wav, spec_flam):

    # First grid the spectrum wavelengths to the filter wavelengths
    spec_on_filt_grid = griddata(points=spec_wav, values=spec_flam, xi=filter_wav)

    # Remove NaNs
    valid_idx = np.where(~np.isnan(spec_on_filt_grid))

    filter_wav = filter_wav[valid_idx]
    filter_thru = filter_thru[valid_idx]
    spec_on_filt_grid = spec_on_filt_grid[valid_idx]

    # Now do the two integrals
    num = np.trapz(y=spec_on_filt_grid * filter_thru, x=filter_wav)
    den = np.trapz(y=filter_thru, x=filter_wav)

    filter_flux = num / den

    return filter_flux

# Read in the F105W filter throughput
filt = np.genfromtxt(fitting_utils + 'F105W_IR_throughput.csv', \
                     delimiter=',', dtype=None, names=True, encoding='ascii', usecols=(1,2))
plot_filt = False
plot_magdiff = False
if plot_filt:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Wavelength [Angstroms]')
    ax.set_ylabel('Throughput')
    ax.plot(filt['Wave_Angstroms'], filt['Throughput'])
    fig.savefig(extdir + 'f105w_filt_curve.pdf', dpi=200, bbox_inches='tight')
    fig.clear()
    plt.close(fig)

# Loop through all objects and
# get the diff in mag between sextrator and 
# spec thru filt
mdiff = []

for d in range(18):

    # Read sedlst
    s = pylinear_lst_dir + 'sed_Y106_' + pt + '_' + str(d+1) + '.lst'
    sedlst = np.genfromtxt(s, dtype=None, names=['segid', 'sed_path'], encoding='ascii', skip_header=2)

    # Read catalog
    cat_filename = roman_direct_dir + 'K_5degimages_part1/' + '5deg_Y106_' + pt + '_' + str(d+1) + '_SNadded.cat'
    cat = np.genfromtxt(cat_filename, dtype=None, names=cat_header, encoding='ascii')

    print('--------- Working on detector:', d+1)

    for i in range(len(cat)):
        
        current_segid = cat['NUMBER'][i]
        mag = cat['MAG_AUTO'][i]

        sed_idx = np.where(sedlst['segid'] == current_segid)[0]
        if not sed_idx.size:
            print('Matched idx:', sed_idx)
            print('current_segid:', current_segid, type(current_segid))
            print(s)
            print(sedlst['segid'])
            print('Could not match segid:', current_segid, 'on detector ', d+1)
            print('Stopping. Check sedlst and catalog.')
            sys.exit(0)
        else:
            sed_idx = int(sed_idx)

        pth = sedlst['sed_path'][sed_idx]
        spec = np.genfromtxt(pth, dtype=None, names=True, encoding='ascii')
        
        flux = filter_conv(filt['Wave_Angstroms'], filt['Throughput'], spec['lam'], spec['flux'])
        fnu = (10552**2 / 3e18) * flux
        implied_mag_pylinear = -2.5 * np.log10(fnu) - 48.6
        
        md = mag - implied_mag_pylinear
        mdiff.append(md)

        # Only scale spectrum if the mag diff is larger than 0.1
        if np.abs(md) > 0.1:    
            # Now scale the spectrum to be supplied to pylinear
            # to be consistent with the broadband mag through the F105W filt
            flux_scale_fac = 10**(-0.4 * md)
            # above equation is flux in sextractor divided by the 
            # flux implied from the spectrum passed to pylinear
            new_sed_flux = spec['flux'] * flux_scale_fac

            with open(pth, 'w') as fh:
                fh.write('#  lam  flux' + '\n')
                for j in range(len(spec['lam'])):
                    fh.write('{:.2f}'.format(spec['lam'][j]) + ' ' + '{:.5e}'.format(new_sed_flux[j]) + '\n')

            print('Scaled and saved:', os.path.basename(pth), ' Scaled by:', '{:.2f}'.format(flux_scale_fac))

        else:
            print('Skipped:', os.path.basename(pth))
            continue

if plot_magdiff:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Sextractor mag - flam thru filter')
    ax.set_ylabel('Number')
    ax.hist(mdiff, 40, range=(-10,10))
    fig.savefig(extdir + 'sextractor_and_pylinear_magdiff.pdf', dpi=200, bbox_inches='tight')
    fig.clear()
    plt.close(fig)
"""
# ------------------------------------
# TEST 4:
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
absmag = -19.5  # in B band
dist_mod = np.asarray(all_sn_mags) - absmag# + Kcor

ax.scatter(all_sn_z, dist_mod, s=15, color='k')

# Also plot apparent mag
axt = ax.twinx()
axt.scatter(all_sn_z, all_sn_mags, s=3, color='r')
axt.set_ylabel('Inserted SN AB mag in F106', fontsize=14)

fig.savefig(extdir + 'test_sn_insert_mag_z.pdf', dpi=200, bbox_inches='tight')

fig.clear()
plt.close(fig)

# ------------------------------------
# TEST 5:
all_sn_av = []

for i in range(18):

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
            av = salt_pth[4][2:].replace('.txt','')
            av = float(av.replace('p', '.'))

            all_sn_av.append(av)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(all_sn_av, 30, color='k', histtype='step')
fig.savefig(extdir + 'test_sn_insert_av.pdf', dpi=200, bbox_inches='tight')

fig.clear()
plt.close(fig)

print('Testing finished.')


