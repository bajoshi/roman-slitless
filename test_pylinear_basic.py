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

ext_spec_filename = basic_testdir + 'romansim_grism_basic_test_x1d.fits'

pylinear_flam_scale_fac = 1e-17

# Read in extracted spectra
ext_hdu = fits.open(ext_spec_filename)

# Read in sed lst
sedlst = np.genfromtxt(basic_testdir + 'small_num_sources_test/sed_small.lst', 
    dtype=None, names=['segid','path'], skip_header=2, encoding='ascii')

# loop over all sources to plot
for i in range(len(sedlst)):

    segid = sedlst['segid'][i]
    print('Plotting SegID:', segid)

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




