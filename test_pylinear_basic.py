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

sys.path.append(roman_slitless_dir)
from test_pylinear_extractions import model_galaxy, model_sn, get_template_inputs, get_chi2, get_dl_at_z
import dust_utils as du

basic_testdir = '/Volumes/Joshi_external_HDD/Roman/roman_direct_sims/pylinear_basic_test/'
tablespath = basic_testdir + 'tables/'

ext_spec_filename = basic_testdir + 'romansim_prism_basic_test_x1d.fits'

# -------------------------
pylinear_flam_scale_fac = 1e-17

# Read in SALT2 SN IA file from Lou
salt2_spec = np.genfromtxt(fitting_utils + "salt2_template_0.txt", \
    dtype=None, names=['day', 'lam', 'flam'], encoding='ascii')

sn_day_arr = np.arange(-20,51,1)
sn_scalefac = 2.0842526537870818e+48  # see sn_scaling.py 

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

# -------------------------
# Read in extracted spectra
ext_hdu = fits.open(ext_spec_filename)

# Read in sed lst
#sedlst = np.genfromtxt(basic_testdir + 'small_num_sources_test/sed_small.lst', 
#    dtype=None, names=['segid','path'], skip_header=2, encoding='ascii')
sedlst = np.genfromtxt(basic_testdir + 'sed.lst', 
    dtype=None, names=['segid','path'], skip_header=2, encoding='ascii')

# loop over all sources to plot
for i in range(110,len(sedlst)):

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
    ax.set_xlim(7300, 18200)
    ax.legend(frameon=False)

    plt.show()

    plt.clf()
    plt.cla()
    plt.close()

    #if i > 10: sys.exit()



