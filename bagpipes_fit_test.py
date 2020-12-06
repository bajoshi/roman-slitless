import numpy as np 
import bagpipes as pipes

from astropy.io import fits

import os
import sys

def load_data(hostid):
    
    hostid = int(hostid)
    
    ext_spec_filename = '/Users/baj/Documents/roman_slitless_sims_results/plffsn2_run_nov30/romansim1_ext_x1d.fits'
    ext_hdu = fits.open(ext_spec_filename)
    
    pylinear_flam_scale_fac = 1e-17
    
    host_wav = ext_hdu[('SOURCE', hostid)].data['wavelength']
    host_flam = ext_hdu[('SOURCE', hostid)].data['flam'] * pylinear_flam_scale_fac
    noise_level = 0.03
    host_ferr = noise_level * host_flam
    
    spectrum = np.c_[host_wav, host_flam, host_ferr]

    return spectrum

def main():

    galaxy = pipes.galaxy('755', load_data, photometry_exists=False)

    dblplaw = {}                        
    dblplaw["tau"] = (0., 15.)            
    dblplaw["alpha"] = (0.01, 1000.)
    dblplaw["beta"] = (0.01, 1000.)
    dblplaw["alpha_prior"] = "log_10"
    dblplaw["beta_prior"] = "log_10"
    dblplaw["massformed"] = (1., 15.)
    dblplaw["metallicity"] = (0.1, 2.)
    dblplaw["metallicity_prior"] = "log_10"
    
    nebular = {}
    nebular["logU"] = -3.
    
    dust = {}
    dust["type"] = "CF00"
    dust["eta"] = 2.
    dust["Av"] = (0., 2.0)
    dust["n"] = (0.3, 2.5)
    dust["n_prior"] = "Gaussian"
    dust["n_prior_mu"] = 0.7
    dust["n_prior_sigma"] = 0.3
    
    fit_instructions = {}
    fit_instructions["redshift"] = (0.7, 1.15)

    # Redshift ranges for the test galaxies
    # also make sure to change the redshift mu prior below 
    # (1.5, 2.5)  # for 207
    # (0.2, 0.7)  # for 475
    # (1.4, 1.8)  # for 548
    # (0.7, 1.15)  # for 755
    
    fit_instructions["t_bc"] = 0.01
    fit_instructions["redshift_prior"] = "Gaussian"
    fit_instructions["redshift_prior_mu"] = 0.92
    fit_instructions["redshift_prior_sigma"] = 0.05
    fit_instructions["dblplaw"] = dblplaw 
    fit_instructions["nebular"] = nebular
    fit_instructions["dust"] = dust

    fit_instructions["veldisp"] = (1., 1000.)   #km/s
    fit_instructions["veldisp_prior"] = "log_10"

    calib = {}
    calib["type"] = "polynomial_bayesian"
    
    calib["0"] = (0.5, 1.5) # Zero order is centred on 1, at which point there is no change to the spectrum.
    calib["0_prior"] = "Gaussian"
    calib["0_prior_mu"] = 1.0
    calib["0_prior_sigma"] = 0.25
    
    calib["1"] = (-0.5, 0.5) # Subsequent orders are centred on zero.
    calib["1_prior"] = "Gaussian"
    calib["1_prior_mu"] = 0.
    calib["1_prior_sigma"] = 0.25
    
    calib["2"] = (-0.5, 0.5)
    calib["2_prior"] = "Gaussian"
    calib["2_prior_mu"] = 0.
    calib["2_prior_sigma"] = 0.25
    
    fit_instructions["calib"] = calib

    noise = {}
    noise["type"] = "white_scaled"
    noise["scaling"] = (1., 10.)
    noise["scaling_prior"] = "log_10"
    fit_instructions["noise"] = noise

    fit = pipes.fit(galaxy, fit_instructions, run="spectroscopy")

    fit.fit(verbose=False)

    fig = fit.plot_spectrum_posterior(save=True, show=True)
    fig = fit.plot_calibration(save=True, show=True)
    fig = fit.plot_sfh_posterior(save=True, show=True)
    fig = fit.plot_corner(save=True, show=True)

    print("\a    \a    \a")

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)

