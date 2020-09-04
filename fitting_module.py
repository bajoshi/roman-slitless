import numpy as np
import scipy.interpolate as interpolate

import os
import sys

import matplotlib.pyplot as plt

home = os.getenv("HOME")
pears_figs_data_dir = home + "/Documents/pears_figs_data/"
stacking_util_codes = home + "/Documents/GitHub/stacking-analysis-pears/util_codes/"

sys.path.append(stacking_util_codes)
from proper_and_lum_dist import luminosity_distance

def do_fitting(obs_wav, obs_flux, obs_flux_err, object_type='galaxy'):

    # Load models
    if object_type == 'galaxy':
        models_llam = np.load(pears_figs_data_dir + 'model_comp_spec_llam_withlines_chabrier.npy', mmap_mode='r')
        models_grid = np.load(pears_figs_data_dir + 'model_lam_grid_withlines_chabrier.npy', mmap_mode='r')

        total_models = len(models_llam)

    # ----------------- Downgrading Resolution ----------------- #
    """
    # Modify models to observed wavelength binning
    # Not using griddata here because that involves interpolation
    # digitize is a better choice it seems

    # Get bin width for observed wavelength array
    wav_binwidth = (max(obs_wav) - min(obs_wav)) / (len(obs_wav) - 1)  # len-1 is the total number of "bins"

    # Now generate the bins
    wav_bin_low = 95.0
    wav_bin_high = 20005.0
    # Since we'll only be dealing with redshifted galaxies 
    # or SNe and this is slightly redder than the reddest wavelength possible
    wav_bin = np.arange(wav_bin_low, wav_bin_high, wav_binwidth)

    # Generating the rebinning grid
    # This array and the eventual rebinned model will have a one-to-one correspondence.
    # Simply taking the midpoints of all the bins in the wav_bin array above.
    rebin_grid = np.zeros(len(wav_bin)-1)
    for j in range(1, len(wav_bin)):
        rebin_grid[j-1] = (wav_bin[j-1] + wav_bin[j])/2.0

    if not os.path.isfile(pears_figs_data_dir + 'models_rebinned.npy'):

        # Find binning
        digitize_result = np.digitize(x=models_grid, bins=wav_bin, right=False)

        # Empty array to store new rebinned models
        models_rebinned = np.zeros((total_models, len(wav_bin)-1))

        # Loop over all models and downgrade resolution
        print("Downgrading resolution...", end='\n')
        for i in range(total_models):
            print("Percent done:", "{:.2f}".format(i * 100/total_models), end='\r')  # Kind of like a progress bar
            # Now take the mean within each bin
            for j in range(1, len(wav_bin)):
                bin_mean = np.mean(models_llam[i][digitize_result == j])
                models_rebinned[i][j-1] = bin_mean

        # IF pylinear always uses the same wavelength grid (at least for a given grism/prism)
        # then these downgraded spectra can be saved and do not need to be recomputed at every run.
        np.save(pears_figs_data_dir + 'models_rebinned.npy', models_rebinned)
        print("Rebinned models saved.")
        sys.exit(0)

    # ----------------- END Downgrading Resolution ----------------- #

    # Read in saved models downgraded in resolution
    models_rebinned = np.load(pears_figs_data_dir + 'models_rebinned.npy')
    """

    # ----------------- Redshift fitting ----------------- #
    redshift_search_grid = np.linspace(0.001, 1.0, 100)
    print("Will search within the following redshift grid:", redshift_search_grid)

    fit_idx_z = np.zeros(len(redshift_search_grid))
    chi2_z = np.zeros(len(redshift_search_grid))

    for z in range(len(redshift_search_grid)):

        redshift = redshift_search_grid[z]
 
        dl = luminosity_distance(redshift)  # returns dl in Mpc
        dl = dl * 3.09e24  # convert to cm

        redshifted_model_grid = models_grid * (1 + redshift)
        models_redshifted = models_llam / (4 * np.pi * dl * dl * (1 + redshift))
        # NOTE: the stellar mass for this model still remains at one solar.
        # However, I haven't yet multiplied by 1 solar luminosity here.
        # So the units are a bit wacky.... the solar lum factor gets absorbed into alpha below.
        # The stellar mass will be solved for later.

        print("\nRedshift:", redshift)
        print("Luminosity distance [cm]:", dl)

        # ----------------- Downgrading Resolution ----------------- #
        resampling_grid = obs_wav
        models_mod = np.zeros((total_models, len(resampling_grid)))

        ### Zeroth element
        lam_step = resampling_grid[1] - resampling_grid[0]
        idx = np.where((redshifted_model_grid >= resampling_grid[0] - lam_step) & \
                       (redshifted_model_grid < resampling_grid[0] + lam_step))[0]
        models_mod[:, 0] = np.mean(models_redshifted[:, idx], axis=1)

        ### all elements in between
        for u in range(1, len(resampling_grid) - 1):
            idx = np.where((redshifted_model_grid >= resampling_grid[u-1]) & \
                           (redshifted_model_grid < resampling_grid[u+1]))[0]
            models_mod[:, u] = np.mean(models_redshifted[:, idx], axis=1)
        
        ### Last element
        lam_step = resampling_grid[-1] - resampling_grid[-2]
        idx = np.where((redshifted_model_grid >= resampling_grid[-1] - lam_step) & \
                       (redshifted_model_grid < resampling_grid[-1] + lam_step))[0]
        models_mod[:, -1] = np.mean(models_redshifted[:, idx], axis=1)

        # ----------------- Chi2 ----------------- #
        num = np.nansum(2 * models_mod * obs_flux / obs_flux_err**2)
        den = 2 * np.nansum(models_mod**2 / obs_flux_err**2)
        alpha = num / den  # vertical scaling factor
        print("Alpha:", "{:.2e}".format(alpha))
        chi2 = (alpha * models_mod - obs_flux)**2 / obs_flux_err**2
        chi2 = np.nansum(chi2, axis=1)

        min_chi2 = min(chi2)
        print("Min chi2:", min_chi2)
        fit_idx_z[z] = np.argmin(chi2)
        chi2_z[z] = min_chi2

    # ----------------- plot p(z) ----------------- #
    pz = get_pz(chi2_z, redshift_search_grid)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(redshift_search_grid, pz)

    plt.show()

    sys.exit(0)

    return fit_dict

def get_pz(chi2_map, z_arr_to_check):

    # Convert chi2 to likelihood
    likelihood = np.exp(-1 * chi2_map / 2)

    # Normalize likelihood function
    norm_likelihood = likelihood / np.sum(likelihood)

    # Get p(z)
    pz = np.zeros(len(z_arr_to_check))

    for i in range(len(z_arr_to_check)):
        pz[i] = np.sum(norm_likelihood[i])

    return pz
    