import numpy as np
from astropy.io import fits
import emcee
import corner
import scipy
from scipy.interpolate import griddata
from astropy.cosmology import Planck15
from multiprocessing import Pool

import os
import sys
from functools import reduce
import time
import datetime

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec

home = os.getenv('HOME')
stacking_utils = home + '/Documents/GitHub/stacking-analysis-pears/util_codes/'
pears_figs_dir = home + '/Documents/pears_figs_data/'

roman_slitless_dir = home + "/Documents/GitHub/roman-slitless/"
ext_spectra_dir = home + "/Documents/roman_slitless_sims_results/"
template_dir = home + "/Documents/roman_slitless_sims_seds/"
roman_sims_seds = home + "/Documents/roman_slitless_sims_seds/"

sys.path.append(stacking_utils)
import proper_and_lum_dist as cosmo

# Read in all models and parameters
model_lam_grid = np.load(pears_figs_dir + 'model_lam_grid_withlines_chabrier.npy', mmap_mode='r')
model_grid = np.load(pears_figs_dir + 'model_comp_spec_llam_withlines_chabrier.npy', mmap_mode='r')

log_age_arr = np.load(pears_figs_dir + 'log_age_arr_chab.npy', mmap_mode='r')
metal_arr = np.load(pears_figs_dir + 'metal_arr_chab.npy', mmap_mode='r')
tau_gyr_arr = np.load(pears_figs_dir + 'tau_gyr_arr_chab.npy', mmap_mode='r')
tauv_arr = np.load(pears_figs_dir + 'tauv_arr_chab.npy', mmap_mode='r')

"""
Array ranges are:
1. Age: 7.02 to 10.114 (this is log of the age in years)
2. Metals: 0.0001 to 0.05 (absolute fraction of metals. All CSP models although are fixed at solar = 0.02)
3. Tau: 0.01 to 63.095 (this is in Gyr. SSP models get -99.0)
4. TauV: 0.0 to 2.8 (Visual dust extinction in magnitudes. SSP models get -99.0)
"""

# Read in SALT2 SN IA file from Lou
salt2_spec = np.genfromtxt(roman_sims_seds + "salt2_template_0.txt", \
    dtype=None, names=['day', 'lam', 'flam'], encoding='ascii')


def get_template(age, tau, tauv, metallicity, \
    log_age_arr, metal_arr, tau_gyr_arr, tauv_arr, \
    model_lam_grid_withlines_mmap, model_comp_spec_withlines_mmap):

    """
    print("\nFinding closest model to --")
    print("Age [Gyr]:", 10**age / 1e9)
    print("Tau [Gyr]:", tau)
    print("Tau_v:", tauv)
    print("Metallicity [abs. frac.]:", metallicity)
    """

    # First find closest values and then indices corresponding to them
    # It has to be done this way because you typically wont find an exact match
    closest_age_idx = np.argmin(abs(log_age_arr - age))
    closest_tau_idx = np.argmin(abs(tau_gyr_arr - tau))
    closest_tauv_idx = np.argmin(abs(tauv_arr - tauv))

    # Now get indices
    age_idx = np.where(log_age_arr == log_age_arr[closest_age_idx])[0]
    tau_idx = np.where(tau_gyr_arr == tau_gyr_arr[closest_tau_idx])[0]
    tauv_idx = np.where(tauv_arr   ==    tauv_arr[closest_tauv_idx])[0]
    metal_idx = np.where(metal_arr == metallicity)[0]

    model_idx = int(reduce(np.intersect1d, (age_idx, tau_idx, tauv_idx, metal_idx)))

    model_llam = model_comp_spec_withlines_mmap[model_idx]

    chosen_age = 10**log_age_arr[model_idx] / 1e9
    chosen_tau = tau_gyr_arr[model_idx]
    chosen_av = 1.086 * tauv_arr[model_idx]
    chosen_metallicity = metal_arr[model_idx]

    """
    print("\nChosen model index:", model_idx)
    print("Chosen model parameters -- ")
    print("Age [Gyr]:", chosen_age)
    print("Tau [Gyr]:", chosen_tau)
    print("A_v:", chosen_av)
    print("Metallicity [abs. frac.]:", chosen_metallicity)
    """

    return model_llam

def apply_redshift(restframe_wav, restframe_lum, redshift):

    dl = cosmo.luminosity_distance(redshift)  # returns dl in Mpc
    dl = dl * 3.09e24  # convert to cm

    redshifted_wav = restframe_wav * (1 + redshift)
    redshifted_flux = restframe_lum / (4 * np.pi * dl * dl * (1 + redshift))

    return redshifted_wav, redshifted_flux

def loglike_sn(theta, x, data, err):
    
    z, day = theta

    y = model_sn(x, z, day)
    #print("Model SN func result:", y)

    # ------- Vertical scaling factor
    alpha = np.nansum(data * y / err**2) / np.nansum(y**2 / err**2)
    #print("Alpha SN:", "{:.2e}".format(alpha))

    y = y * alpha

    lnLike = -0.5 * np.nansum( (y-data)**2/err**2  +  np.log(2 * np.pi * err**2))
    #print("ln(likelihood) SN", lnLike)
    
    return lnLike

def loglike_host(theta, x, data, err):
    
    z, age, tau, av = theta

    y = model_host(x, z, age, tau, av)
    #print("Model func result:", y)

    # ------- Vertical scaling factor
    alpha = np.nansum(data * y / err**2) / np.nansum(y**2 / err**2)
    #print("Alpha HOST:", "{:.2e}".format(alpha))

    y = y * alpha

    lnLike = -0.5 * np.nansum( (y-data)**2/err**2  +  np.log(2 * np.pi * err**2))
    return lnLike

def logprior_sn(theta):

    z, day = theta

    if ( 0.01 <= z <= 6.0  and  -19 <= day <= 50 ):
        return 0.0
    
    return -np.inf

def logprior_host(theta):

    z, age, tau, av = theta
    
    # Make sure model is not older than the Universe
    # Allowing at least 100 Myr for the first galaxies to form after Big Bang
    age_at_z = Planck15.age(z).value  # in Gyr
    age_lim = age_at_z - 0.1  # in Gyr

    #if ( 0.01 <= z <= 6.0  and  0.01 <= age <= age_lim  and  0.01 <= tau <= 100.0  and  0.0 <= av <= 3.0  and  0.0 <= lsf_sigma <= 300.0  ):
    if ( 0.01 <= z <= 6.0  and  0.01 <= age <= age_lim  and  0.01 <= tau <= 100.0  and  0.0 <= av <= 3.0 ):
        return 0.0
    
    return -np.inf

def logpost_sn(theta, x, data, err):

    lp = logprior_sn(theta)
    
    if not np.isfinite(lp):
        return -np.inf
    
    lnL = loglike_sn(theta, x, data, err)

    #print("Likelihood:", lnL)
    
    return lp + lnL

def logpost_host(theta, x, data, err):

    lp = logprior_host(theta)
    #print("Prior HOST:", lp)
    
    if not np.isfinite(lp):
        return -np.inf
    
    lnL = loglike_host(theta, x, data, err)

    #print("Likelihood HOST:", lnL)
    
    return lp + lnL

def model_sn(x, z, day):

    # pull out spectrum for the chosen day
    day_idx = np.where(salt2_spec['day'] == day)[0]

    sn_spec_flam = salt2_spec['flam'][day_idx]

    # ------ Apply redshift
    sn_lam_z, sn_flam_z = apply_redshift(salt2_spec['lam'][day_idx], sn_spec_flam, z)

    # ------ Downgrade to grism resolution
    sn_mod = np.zeros(len(x))

    ### Zeroth element
    lam_step = x[1] - x[0]
    idx = np.where((sn_lam_z >= x[0] - lam_step) & (sn_lam_z < x[0] + lam_step))[0]
    sn_mod[0] = np.mean(sn_flam_z[idx])

    ### all elements in between
    for j in range(1, len(x) - 1):
        idx = np.where((sn_lam_z >= x[j-1]) & (sn_lam_z < x[j+1]))[0]
        sn_mod[j] = np.mean(sn_flam_z[idx])
    
    ### Last element
    lam_step = x[-1] - x[-2]
    idx = np.where((sn_lam_z >= x[-1] - lam_step) & (sn_lam_z < x[-1] + lam_step))[0]
    sn_mod[-1] = np.mean(sn_flam_z[idx])

    return sn_mod

def model_host(x, z, age_gyr, tau_gyr, av):
    """
    This function will return the closest BC03 template 
    from a large grid of pre-generated templates.

    Expects to get the following arguments
    x: observed wavelength grid
    z: redshift to apply to template
    age: age of SED in Gyr
    tau: exponential SFH timescale in Gyr
    av: visual dust extinction
    lsf_sigma: in angstroms
    """

    current_age = np.log10(age_gyr * 1e9)  # because the saved age parameter is the log(age[yr])
    current_tau = tau_gyr  # because the saved tau is in Gyr
    tauv = av / 1.086
    current_tauv = tauv
    current_metallicity = 0.02  # Force it to only choose from the solar metallicity CSP models

    model_llam = get_template(current_age, current_tau, current_tauv, current_metallicity, \
        log_age_arr, metal_arr, tau_gyr_arr, tauv_arr, model_lam_grid, model_grid)

    # ------ Apply redshift
    model_lam_z, model_flam_z = apply_redshift(model_lam_grid, model_llam, z)

    # ------ Apply LSF
    #model_lsfconv = scipy.ndimage.gaussian_filter1d(input=model_flam_z, sigma=lsf_sigma)

    # ------ Downgrade to grism resolution
    model_mod = np.zeros(len(x))

    ### Zeroth element
    lam_step = x[1] - x[0]
    idx = np.where((model_lam_z >= x[0] - lam_step) & (model_lam_z < x[0] + lam_step))[0]
    model_mod[0] = np.mean(model_flam_z[idx])

    ### all elements in between
    for j in range(1, len(x) - 1):
        idx = np.where((model_lam_z >= x[j-1]) & (model_lam_z < x[j+1]))[0]
        model_mod[j] = np.mean(model_flam_z[idx])
    
    ### Last element
    lam_step = x[-1] - x[-2]
    idx = np.where((model_lam_z >= x[-1] - lam_step) & (model_lam_z < x[-1] + lam_step))[0]
    model_mod[-1] = np.mean(model_flam_z[idx])

    return model_mod

def add_noise(sig_arr, noise_level):
    """
    This function will vary the flux randomly within 
    the noise level specified. It assumes the statistical
    noise is Gaussian.
    The final error array returned is Poissonian noise.
    """
    # Poisson noise: does the signal have to be in 
    # units of photons or electrons for sqrt(N) to 
    # work? like I cannot use sqrt(signal) in physical 
    # units and call it Poisson noise?

    sigma_arr = noise_level * sig_arr

    spec_noise = np.zeros(len(sig_arr))
    err_arr = np.zeros(len(sig_arr))

    for k in range(len(sig_arr)):

        mu = sig_arr[k]
        sigma = sigma_arr[k]

        # Now vary flux using numpy random.normal
        # mu and the resulting new flux value HAVE TO BE POSITIVE!
        if mu >= 0:
            spec_noise[k] = np.random.normal(mu, sigma, 1)
        elif mu < 0:
            spec_noise[k] = np.nan

        if spec_noise[k] < 0:
            max_iters = 10
            iter_count = 0
            while iter_count < max_iters:
                spec_noise[k] = np.random.normal(mu, sigma, 1)
                iter_count += 1
                if spec_noise[k] > 0:
                    break
            # if it still hasn't given a positive number after max_iters
            # then revert it back to whatever the signal was before randomly varying
            if (iter_count >= max_iters) and (spec_noise[k] < 0):
                spec_noise[k] = sig_arr[k]

        # err_arr[k] = np.sqrt(spec_noise[k])
        err_arr[k] = 4 * noise_level * spec_noise[k]

    return spec_noise, err_arr

def get_autocorr_time(sampler):

    try:
        tau = sampler.get_autocorr_time()
    except emcee.autocorr.AutocorrError as errmsg:
        print(errmsg)
        print("\n")
        print("Emcee AutocorrError occured.")
        print("The chain is shorter than 50 times the integrated autocorrelation time for 5 parameter(s).")
        print("Use this estimate with caution and run a longer chain!")
        print("\n")

        tau_list_str = str(errmsg).split('tau:')[-1]
        tau_list = tau_list_str.split()
        print("Tau list:", tau_list)

        tau = []
        for j in range(len(tau_list)):
            curr_elem = tau_list[j]
            if ('[' in curr_elem) and (len(curr_elem) > 1):
                tau.append(float(curr_elem.lstrip('[')))
            elif (']' in curr_elem) and (len(curr_elem) > 1):
                tau.append(float(curr_elem.rstrip(']')))
            elif len(curr_elem) > 1:
                tau.append(float(tau_list[j]))

    print("Tau:", tau)

    return tau

def main():

    print("\n * * * *    [WARNING]: model has worse resolution than data in NIR. np.mean() will result in nan. Needs fixing.    * * * * \n")
    print("\n * * * *    [WARNING]: check vertical scaling.    * * * * \n")

    ext_root = "romansim1"
    img_suffix = 'Y106_11_1'

    # Read in sed.lst
    sedlst_header = ['segid', 'sed_path']
    sedlst = np.genfromtxt(roman_slitless_dir + 'sed_' + img_suffix + '.lst', \
        dtype=None, names=sedlst_header, encoding='ascii')

    # Read in the extracted spectra
    ext_spec_filename = ext_spectra_dir + ext_root + '_ext_x1d.fits'
    ext_hdu = fits.open(ext_spec_filename)
    print("Read in extracted spectra from:", ext_spec_filename)

    # Set pylinear f_lambda scaling factor
    pylinear_flam_scale_fac = 1e-17

    # This will come from detection on the direct image
    # For now this comes from the sedlst generation code
    # For Y106_11_1
    host_segids = np.array([207])  # ([475, 755, 548, 207])
    sn_segids = np.array([241])  # ([481, 753, 547, 241])

    for i in range(len(sedlst)):

        # Get info
        segid = sedlst['segid'][i]

        # Read in the dummy template passed to pyLINEAR
        template_name = os.path.basename(sedlst['sed_path'][i])

        if 'salt' not in template_name:
            continue
        else:
            print("Segmentation ID:", segid, "is a SN. Will begin fitting.")

            # Get corresponding host ID
            hostid = int(host_segids[np.where(sn_segids == segid)[0]])
            print("I have the following SN and HOST IDs:", segid, hostid)

            # Read in template
            template = np.genfromtxt(template_dir + template_name, dtype=None, names=True, encoding='ascii')

            # ---------------------------- Set up input params dict ---------------------------- #
            input_dict = {}

            print("INPUTS:")
            # ---- SN
            t = template_name.split('.txt')[0].split('_')

            sn_z = float(t[-1].replace('p', '.').replace('z',''))
            sn_day = int(t[-2].replace('day',''))
            print("Supernova input z:", sn_z)
            print("Supernova day:", sn_day, "\n")

            # ---- HOST
            h_idx = int(np.where(sedlst['segid'] == hostid)[0])
            h_path = sedlst['sed_path'][h_idx]
            th = os.path.basename(h_path)

            th = th.split('.txt')[0].split('_')

            host_z = float(th[-1].replace('p', '.').replace('z',''))
            host_ms = float(th[-2].replace('p', '.').replace('ms',''))

            host_age_u = th[-3]
            if host_age_u == 'gyr':
                host_age = float(th[2])
            elif host_age_u == 'myr':
                host_age = float(th[2])/1e3

            print("Host input z:", host_z)
            print("Host input stellar mass [log(Ms/Msol)]:", host_ms)
            print("Host input age [Gyr]:", host_age)

            input_dict['host_z'] = host_z
            input_dict['host_ms'] = host_ms
            input_dict['host_age'] = host_age

            input_dict['sn_z'] = sn_z
            input_dict['sn_day'] = sn_day

            # ---------------------------- FITTING ---------------------------- #
            # ---------- Get spectrum for host and sn
            host_wav = ext_hdu[hostid].data['wavelength']
            host_flam = ext_hdu[hostid].data['flam'] * pylinear_flam_scale_fac
        
            sn_wav = ext_hdu[segid].data['wavelength']
            sn_flam = ext_hdu[segid].data['flam'] * pylinear_flam_scale_fac

            # ---- Fit template to HOST and SN
            noise_level = 0.05  # relative to signal
            # First assign noise to each point
            #host_flam_noisy, host_ferr = add_noise(host_flam, noise_level)
            #sn_flam_noisy, sn_ferr = add_noise(sn_flam, noise_level)

            host_ferr = noise_level * host_flam
            sn_ferr = noise_level * host_flam

            # Test figure
            """
            snr_host = host_flam_noisy / host_ferr
            print("Signal to noise array:", snr_host)
            print("Mean of signal to noise array:", np.mean(snr_host))

            fig = plt.figure()
            ax = fig.add_subplot()
            ax.plot(host_wav, host_flam_noisy, color='tab:brown', lw=1.0)
            ax.fill_between(host_wav, host_flam_noisy - host_ferr, host_flam_noisy + host_ferr, \
                color='grey', alpha=0.5)

            # overplot rebinned spectrum
            rebin_grid = np.arange(9500, 20000, 30)
            host_flam_binned = scipy.interpolate.griddata(points=host_wav, values=host_flam_noisy, xi=rebin_grid)
            ax.plot(rebin_grid, host_flam_binned, color='tab:blue', lw=1.5)

            m = model_host(host_wav, host_z, host_age, 1.0, 0.0)
            a = np.nansum(host_flam_noisy * m / host_ferr**2) / np.nansum(m**2 / host_ferr**2)
            #ax.plot(host_wav, m * a, color='tab:red')

            plt.show()
            sys.exit(0)
            """
            
            """
            # pull out spectrum for the chosen day
            day_idx = np.where(salt2_spec['day'] == sn_day)[0]
            sn_flam_true = salt2_spec['flam'][day_idx]
            sn_wav_true = salt2_spec['lam'][day_idx]

            sn_wav_true_z, sn_flam_true_z = apply_redshift(sn_wav_true, sn_flam_true, sn_z)

            scalefac = 2e42
            sn_flam_true_z *= scalefac

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(sn_wav, sn_flam_noisy, color='k')
            ax.fill_between(sn_wav, sn_flam_noisy - sn_ferr, sn_flam_noisy + sn_ferr, \
                color='grey', alpha=0.5)

            ax.plot(sn_wav_true_z, sn_flam_true_z, color='tab:blue')
            sn_flam_true_z = scipy.ndimage.gaussian_filter1d(input=sn_flam_true_z, sigma=15.0)
            ax.plot(sn_wav_true_z, sn_flam_true_z, color='tab:red')

            ax.set_xlim(9000, 20000)

            plt.show()

            sys.exit(0)
            """

            # ----------------------- Using MCMC to fit ----------------------- #
            print("\nRunning emcee...")

            # Set jump sizes # ONLY FOR INITIAL POSITION SETUP
            jump_size_z = 0.02  
            jump_size_age = 0.1  # in gyr
            jump_size_tau = 0.1  # in gyr
            jump_size_av = 0.5  # magnitudes

            jump_size_day = 2  # days
            jump_size_lsf = 1.0  # angstroms

            # Labels for corner and trace plots
            label_list_host = [r'$z$', r'$Age [Gyr]$', r'$\tau [Gyr]$', r'$A_V [mag]$']
            label_list_sn = [r'$z$', r'$Day$']

            # Define initial position
            # The parameter vector is (redshift, age, tau, av, lsf_sigma)
            # age in gyr and tau in gyr
            # dust parameter is av not tauv
            # lsf_sigma in angstroms
            rhost_init = np.array([0.02, 0.4, 2.0, 0.0])
            rsn_init = np.array([0.01, 1])  # redshift and day relative to peak

            # Setup dims and walkers
            ndim_host, nwalkers = 4, 100  # setting up emcee params--number of params and number of walkers
            ndim_sn, nwalkers   = 2, 100  # setting up emcee params--number of params and number of walkers

            # generating ball of walkers about initial position defined above
            pos_host = np.zeros(shape=(nwalkers, ndim_host))
            pos_sn = np.zeros(shape=(nwalkers, ndim_sn))

            for i in range(nwalkers):

                # ---------- For HOST
                rh0 = float(rhost_init[0] + jump_size_z * np.random.normal(size=1))
                rh1 = float(rhost_init[1] + jump_size_age * np.random.normal(size=1))
                rh2 = float(rhost_init[2] + jump_size_tau * np.random.normal(size=1))
                rh3 = float(rhost_init[3] + jump_size_av * np.random.normal(size=1))

                rh = np.array([rh0, rh1, rh2, rh3])

                pos_host[i] = rh

                # ---------- For SN
                rsn0 = float(rsn_init[0] + jump_size_z * np.random.normal(size=1))
                rsn1 = int(rsn_init[1] + jump_size_day * np.random.choice([-1, 1]))

                rsn = np.array([rsn0, rsn1])

                pos_sn[i] = rsn

            # ----------- Emcee 
            with Pool() as pool:
                sampler_sn = emcee.EnsembleSampler(nwalkers, ndim_sn, logpost_sn, args=[sn_wav, sn_flam, sn_ferr], pool=pool)
                sampler_sn.run_mcmc(pos_sn, 5000, progress=True)

            print("Done with SN fitting.")

            with Pool() as pool:            
                sampler_host = emcee.EnsembleSampler(nwalkers, ndim_host, logpost_host, args=[host_wav, host_flam, host_ferr], pool=pool)
                sampler_host.run_mcmc(pos_host, 5000, progress=True)

            print("Done with host galaxy fitting.")
            print("Finished running emcee.")

            samples_host = sampler_host.get_chain()
            print("Samples HOST shape:", samples_host.shape)
            samples_sn = sampler_sn.get_chain()
            print("Samples SN shape:", samples_sn.shape)

            # plot trace HOST
            fig1, axes1 = plt.subplots(ndim_host, figsize=(10, 6), sharex=True)

            for i in range(ndim_host):
                ax1 = axes1[i]
                ax1.plot(samples_host[:, :, i], "k", alpha=0.1)
                ax1.set_xlim(0, len(samples_host))
                ax1.set_ylabel(label_list_host[i])
                ax1.yaxis.set_label_coords(-0.1, 0.5)

            axes1[-1].set_xlabel("Step number")

            fig1.savefig(roman_slitless_dir + 'mcmc_trace_host_' + str(hostid) + '_' + img_suffix + '.pdf', \
                dpi=200, bbox_inches='tight')

            # plot trace SN
            fig2, axes2 = plt.subplots(ndim_sn, figsize=(10, 6), sharex=True)

            for i in range(ndim_sn):
                ax2 = axes2[i]
                ax2.plot(samples_sn[:, :, i], "k", alpha=0.1)
                ax2.set_xlim(0, len(samples_sn))
                ax2.set_ylabel(label_list_sn[i])
                ax2.yaxis.set_label_coords(-0.1, 0.5)

            axes2[-1].set_xlabel("Step number")

            fig2.savefig(roman_slitless_dir + 'mcmc_trace_sn_' + str(segid) + '_' + img_suffix + '.pdf', \
                dpi=200, bbox_inches='tight')

            # Get autocorrelation time
            tau_host = get_autocorr_time(sampler_host)
            tau_sn = get_autocorr_time(sampler_sn)

            # Discard burn-in. You do not want to consider the burn in the corner plots/estimation.
            if not np.isnan(tau_host[0]):
                burn_in_host = int(3 * tau_host[0])
                thinning_steps_host = int(0.5 * tau_host[0])
            else:
                burn_in_host = 400
                thinning_steps_host = 67
            print("Burn-in HOST:", burn_in_host)
            print("Thinning steps HOST:", thinning_steps_host)

            flat_samples_host = sampler_host.get_chain(discard=burn_in_host, thin=thinning_steps_host, flat=True)
            print("Flat samples HOST shape:", flat_samples_host.shape)

            if not np.isnan(tau_sn[0]):
                burn_in_sn = int(3 * tau_sn[0])
                thinning_steps_sn = int(0.5 * tau_sn[0])
            else:
                burn_in_sn = 400
                thinning_steps_sn = 67
            print("Burn-in SN:", burn_in_sn)
            print("Thinning steps SN:", thinning_steps_sn)

            flat_samples_sn = sampler_sn.get_chain(discard=burn_in_sn, thin=thinning_steps_sn, flat=True)
            print("Flat samples SN shape:", flat_samples_sn.shape)

            # plot corner plot
            fig_host = corner.corner(flat_samples_host, plot_contours='True', labels=label_list_host, label_kwargs={"fontsize": 12}, \
                show_titles='True', title_kwargs={"fontsize": 12}, truths=np.array([host_z, host_age, 1.0, 0.0]))
            fig_host.savefig(roman_slitless_dir + 'mcmc_fitres_host_' + str(hostid) + '_' + img_suffix + '.pdf', \
                dpi=200, bbox_inches='tight')

            fig_sn = corner.corner(flat_samples_sn, plot_contours='True', labels=label_list_sn, label_kwargs={"fontsize": 12}, \
                show_titles='True', title_kwargs={"fontsize": 12})
            fig_sn.savefig(roman_slitless_dir + 'mcmc_fitres_sn_' + str(segid) + '_' + img_suffix + '.pdf', \
                dpi=200, bbox_inches='tight')

            # ------------ Plot 100 random models from the parameter space
            inds_host = np.random.randint(len(flat_samples_host), size=100)

            fig3 = plt.figure()
            ax3 = fig3.add_subplot(111)

            ax3.plot(host_wav, host_flam, color='k')
            ax3.fill_between(host_wav, host_flam - host_ferr, host_flam + host_ferr, color='gray', alpha=0.5, zorder=3)

            for ind in inds_host:
                sample = flat_samples_host[ind]
                m = model_host(host_wav, sample[0], sample[1], sample[2], sample[3])
                a = np.nansum(host_flam * m / host_ferr**2) / np.nansum(m**2 / host_ferr**2)
                ax3.plot(host_wav, a * m, color='tab:red', alpha=0.2, zorder=2)

            fig3.savefig(roman_slitless_dir + 'mcmc_fitres_host_overplot_' + str(hostid) + '_' + img_suffix + '.pdf', \
                dpi=200, bbox_inches='tight')

            # ----------- Plot 100 random models from the parameter space
            inds_sn = np.random.randint(len(flat_samples_sn), size=100)

            fig4 = plt.figure()
            ax4 = fig4.add_subplot(111)

            ax4.plot(sn_wav, sn_flam, color='k')
            ax4.fill_between(sn_wav, sn_flam - sn_ferr, sn_flam + sn_ferr, color='gray', alpha=0.5, zorder=3)

            for ind in inds_sn:
                sample = flat_samples_sn[ind]
                m = model_sn(sn_wav, sample[0], sample[1])
                a = np.nansum(sn_flam * m / sn_ferr**2) / np.nansum(m**2 / sn_ferr**2)
                ax4.plot(sn_wav, a * m, color='tab:red', alpha=0.2, zorder=2)

            fig4.savefig(roman_slitless_dir + 'mcmc_fitres_sn_overplot_' + str(segid) + '_' + img_suffix + '.pdf', \
                dpi=200, bbox_inches='tight')

            sys.exit(0)

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)

