import numpy as np
from astropy.io import fits
from scipy.interpolate import griddata

import os
import sys

import matplotlib.pyplot as plt

home = os.getenv('HOME')

sys.path.append(home + '/Documents/GitHub/roman-slitless/')
from test_pylinear_extractions import model_galaxy, model_sn, get_template_inputs, get_chi2

basic_testdir = '/Volumes/Joshi_external_HDD/Roman/roman_direct_sims/pylinear_basic_test/'
tablespath = basic_testdir + 'tables/'

ext_spec_filename = basic_testdir + 'romansim_grism_basic_test_x1d.fits'

pylinear_flam_scale_fac = 1e-17

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
if create_reg:
    import pylinear

    segids_for_2dreg = [53, 78, 209, 356, 464, 466, 525, 710, 771, 822, 993]
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
sedlst = np.genfromtxt(basic_testdir + 'small_num_sources_test/sed_small.lst', 
    dtype=None, names=['segid','path'], skip_header=2, encoding='ascii')

# loop over all sources to plot
for i in range(len(sedlst)):

    segid = sedlst['segid'][i]
    print('\nPlotting SegID:', segid)

    wav = ext_hdu[('SOURCE', segid)].data['wavelength']
    flam = ext_hdu[('SOURCE', segid)].data['flam'] * pylinear_flam_scale_fac

    noise_lvl = 0.03

    # ---------- plotting
    fig = plt.figure(figsize=(9,5))
    ax = fig.add_subplot(111)

    # plot x1d spec
    ax.plot(wav, flam, color='k', label='Extracted spectrum', zorder=2)

    # plot input spectrum
    input_sed = np.genfromtxt(sedlst['path'][i], dtype=None, names=True, encoding='ascii')
    sedflux_grid = griddata(points=input_sed['lam'], values=input_sed['flux'], xi=wav)
    sed_a, sed_chi2 = get_chi2(sedflux_grid, flam, noise_lvl*flam)
    ax.plot(input_sed['lam'], input_sed['flux']*sed_a, 
        color='crimson', label='Input SED, scaled', zorder=1)

    # Plot input model derived from model functions
    # Get template inputs
    template_name = os.path.basename(sedlst['path'][i])
    template_inputs = get_template_inputs(template_name)

    # models
    if 'salt' in template_name:
        m = model_sn(wav, template_inputs[0], template_inputs[1], template_inputs[2])
    else:
        m = model_galaxy(wav, template_inputs[0], template_inputs[1], template_inputs[2], 
                         template_inputs[3], template_inputs[4])

    
    a, chi2 = get_chi2(m, flam, noise_lvl*flam)

    ax.plot(wav, m*a, color='teal', label='Downgraded model, scaled')

    ax.set_xlim(9800, 19500)
    ax.legend(frameon=False)

    plt.show()

    plt.clf()
    plt.cla()
    plt.close()




