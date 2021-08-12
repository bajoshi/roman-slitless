import numpy as np
from astropy.io import fits
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter1d

import os
import sys

import matplotlib.pyplot as plt

home = os.getenv('HOME')
roman_slitless_dir = home + '/Documents/GitHub/roman-slitless/'
fitting_utils = roman_slitless_dir + 'fitting_pipeline/utils/'

# Custom imports
sys.path.append(fitting_utils)
import dust_utils as du
from get_snr import get_snr

# Set pylinear f_lambda scaling factor
pylinear_flam_scale_fac = 1e-17

# Header for SExtractor catalog
cat_header = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'ALPHA_J2000', 'DELTA_J2000', 
'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_AUTO', 'MAGERR_AUTO', 'FLUX_RADIUS', 'FWHM_IMAGE']

basic_testdir = '/Volumes/Joshi_external_HDD/Roman/roman_direct_sims/pylinear_basic_test/'
tablespath = basic_testdir + 'tables/'

ext_spec_filename = basic_testdir + 'romansim_prism_basic_test_x1d.fits'

# -------------------------
"""
# Read in SALT2 SN IA file from Lou
salt2_spec = np.genfromtxt(fitting_utils + "salt2_template_0.txt", \
    dtype=None, names=['day', 'lam', 'flam'], encoding='ascii')

sn_day_arr = np.arange(-19,51,1)
sn_scalefac = 2.0842526537870818e+48  # see sn_scaling.py 
"""
# -------------------------
create_reg = False
def create_2d_ext_regions(segid_list, grisms, sources):

    # ---------------------------
    # Loop over grisms and gen regions for all segid    
        
    for grism in grisms:
            
        print("Working on:", grism.dataset)
        reg_filename = basic_testdir + grism.dataset + '_2dext.reg'
        with open(reg_filename, 'w') as fh:
            
            with pylinear.h5table.H5Table(grism.dataset, path=tablespath, mode='r') as h5:
                
                device = grism['SCA09']
                
                h5.open_table('SCA09', '+1', 'pdt')
    
                for segid in segid_list:
                
                    odt = h5.load_from_file(sources[segid], '+1', 'odt')
                    ddt = odt.decimate(device.naxis1, device.naxis2)
                
                    try:
                        region_text = ddt.region()
                    except ValueError:
                        print('ValueError raised for SegID:', segid)
                        print('Not sure why but I suspect this spectrum')
                        print('is not on the detector for this PA.')
                        continue

                    region_text = region_text.replace('helvetica 12 bold', 'helvetica 10 bold')
            
                    fh.write(region_text + '\n')

        print("Saved region for:", grism.dataset)

    return None

# -------------------------
# First create 2d extraction regions for selected sources
segids_for_2dreg = []
if create_reg:
    import pylinear

    # some of these might have the x1d spectrum steeper than the model

    segfile = basic_testdir + 'small_num_sources_test/5deg_Y106_0_1_cps_segmap_small.fits'
    obslst = basic_testdir + 'small_num_sources_test/obs.lst'
    fltlst = basic_testdir + 'small_num_sources_test/flt.lst'

    maglim = 99.0

    # Load in sources
    sources = pylinear.source.SourceCollection(segfile, obslst, 
                detindex=0, maglim=maglim)

    # Load in grisms for the sim to test
    grisms = pylinear.grism.GrismCollection(fltlst, observed=True)

    create_2d_ext_regions(segids_for_2dreg, grisms, sources)

    print('Regions created. Turn flag off and rerun.')
    sys.exit(0)

# ------------------------- Read in extracted spectra
ext_hdu = fits.open(ext_spec_filename)

# ------------------------- Read in sextractor catalog
catfile = basic_testdir + '5deg_Y106_0_1_SNadded.cat'
cat = np.genfromtxt(catfile, dtype=None, names=cat_header, encoding='ascii')

# ------------------------- Read in sed lst
sedlst_header = ['segid', 'sed_path']
sedlst_path = basic_testdir + 'sed.lst'
sedlst = np.genfromtxt(sedlst_path, dtype=None, names=sedlst_header, encoding='ascii')

# ------------------------- Make SNR vs mag plot
all_sn_segids = []
for i in range(len(sedlst)):
    if 'salt' in sedlst['sed_path'][i]:
        all_sn_segids.append(sedlst['segid'][i])

print('ALL SN segids in this file:', all_sn_segids)
print('Total SNe:', len(all_sn_segids))

# -----------
# Manual entries from running HST/WFC3 spectroscopic ETC
# For G102 and G141
etc_mags = np.arange(18.0, 25.5, 0.5)
etc_g102_snr = np.array([558.0, 414.0, 300.1, 211.89, 145.79, 
                         98.03, 64.68, 42.07, 27.09, 17.32, 
                         11.02, 6.99, 4.43, 2.80, 1.77])

all_sn_mags = []
all_sn_snr  = []

all_galaxy_mags = []
all_galaxy_snr  = []

for i in range(len(sedlst)):

    # First match with catalog
    current_segid = sedlst['segid'][i]
    cat_idx = np.where(cat['NUMBER'] == current_segid)[0]

    # now get magnitude
    mag = cat['MAG_AUTO'][cat_idx]

    # Get spectrum from extracted file adn SNR
    wav = ext_hdu[('SOURCE', current_segid)].data['wavelength']
    flam = ext_hdu[('SOURCE', current_segid)].data['flam'] * pylinear_flam_scale_fac
    snr = get_snr(wav, flam)

    # Append to appropriate lists depending on object type
    if 'salt' in sedlst['sed_path'][i]:
        all_sn_mags.append(mag)
        all_sn_snr.append(snr)
    else:
        all_galaxy_mags.append(mag)
        all_galaxy_snr.append(snr)

# ------------
fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_ylabel('SNR of extracted 1d spec', fontsize=14)
ax.set_xlabel('F106 mag', fontsize=14)

ax.scatter(all_galaxy_mags, all_galaxy_snr, marker='o', s=10, 
    color='k', label='pyLINEAR sim result, galaxies', zorder=1)
ax.scatter(all_sn_mags, all_sn_snr,         marker='o', s=10, 
    color='k', facecolors='None', label='pyLINEAR sim result, SNe', zorder=2)

ax.scatter(etc_mags, etc_g102_snr, s=8, color='royalblue', label='WFC3 G102 ETC prediction' + '\n' + 'Exptime: 18000s')

ax.legend(loc=0, fontsize=11)
ax.set_yscale('log')

fig.savefig(basic_testdir + 'pylinear_sim_snr_vs_mag.pdf', 
    dpi=200, bbox_inches='tight')

fig.clear()
plt.close(fig)
sys.exit(0)

# ------------------------- Plotting extracted spectra

# loop over all sources to plot
for i in range(1,len(sedlst)):

    segid = sedlst['segid'][i]
    #if segid not in segids_for_2dreg:
    #    continue

    print('\nPlotting SegID:', segid)

    wav = ext_hdu[('SOURCE', segid)].data['wavelength']
    flam = ext_hdu[('SOURCE', segid)].data['flam'] * pylinear_flam_scale_fac

    noise_lvl = 0.03

    # ---------- plotting
    fig = plt.figure(figsize=(9,5))
    ax = fig.add_subplot(111)

    ax.set_xlabel(r'$\mathrm{\lambda\ [\AA]}$', fontsize=15)
    ax.set_ylabel(r'$\mathrm{f_\lambda\ [erg\, s^{-1}\, cm^{-2}\, \AA^{-1}]}$', 
        fontsize=15)

    # plot x1d spec
    ax.plot(wav, flam, color='k', label='Extracted spectrum', zorder=2)
    sf = gaussian_filter1d(flam, sigma=2.5)
    ax.plot(wav, sf, color='gray', lw=2.5, zorder=5)

    # plot input spectrum
    input_sed = np.genfromtxt(sedlst['path'][i], dtype=None, names=True, encoding='ascii')
    sedflux_grid = griddata(points=input_sed['lam'], values=input_sed['flux'], xi=wav)
    sed_a, sed_chi2 = get_chi2(sedflux_grid, sf, noise_lvl*sf)
    print(sed_a, sed_chi2)
    ax.plot(input_sed['lam'], input_sed['flux']*sed_a, 
        color='crimson', label='Input SED, scaled', zorder=1)
    #ax.plot(wav, sedflux_grid*sed_a, color='purple')

    # Plot input model derived from model functions
    # Get template inputs
    template_name = os.path.basename(sedlst['path'][i])
    template_inputs = get_template_inputs(template_name)
    print(template_inputs)

    # models
    if 'salt' in template_name:
        m = model_sn(wav, template_inputs[0], template_inputs[1], template_inputs[2])
    else:
        m = model_galaxy(wav, template_inputs[0], template_inputs[1], template_inputs[2], 
                         template_inputs[3], template_inputs[4])
    
    a, chi2 = get_chi2(m, sf, noise_lvl*sf)
    print(a, chi2)

    ax.plot(wav, m*a, color='teal', label='Downgraded model, scaled')

    # Also manually get model
    """
    snz = template_inputs[0]
    day = template_inputs[1]
    snav = template_inputs[2]

    print('day:', day)
    day_idx = np.argmin(abs(sn_day_arr - day))
    print('matched day:', sn_day_arr[day_idx])
    sn_spec_idx = np.where(salt2_spec['day'] == sn_day_arr[day_idx])[0]
    print('all sn spec idx:', sn_spec_idx)

    snw = salt2_spec['lam'][sn_spec_idx]
    snf = salt2_spec['flam'][sn_spec_idx] * sn_scalefac

    # Apply dust and redshift
    snf = du.get_dust_atten_model(snw, snf, snav)

    dl = get_dl_at_z(snz)
    print('Luminosity distance for z [cm]:', dl)

    snw = snw * (1 + snz)
    snf = snf / (4 * np.pi * dl * dl * (1 + snz))

    ax.plot(snw, snf*a, color='goldenrod')
    """

    #ax.set_xlim(9800, 19500)
    ax.set_ylim(np.nanmin(sf) * 0.5, np.nanmax(sf) * 1.4)
    ax.set_xlim(7300, 18200)
    ax.legend(frameon=False)

    plt.show()

    plt.clf()
    plt.cla()
    plt.close()

    #if i > 10: sys.exit()






