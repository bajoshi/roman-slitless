import numpy as np
from astropy.io import fits
import emcee
import corner
import scipy
from scipy.interpolate import griddata

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
astropy_cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)
from specutils.analysis import snr_derived
from specutils import Spectrum1D

import os
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import datetime as dt
import time 
import socket
from multiprocessing import Pool
from numba import jit

import matplotlib.pyplot as plt

start = time.time()
print("Starting at:", dt.datetime.now())

print("Emcee version:", emcee.__version__)
print("Corner version:", corner.__version__)

# Assign directories and custom imports
cwd = os.path.dirname(os.path.abspath(__file__))
fitting_utils = cwd + '/utils/'

sys.path.append(fitting_utils)
import dust_utils as du

# Define any required constants/arrays
Lsol = 3.826e33
sn_day_arr = np.arange(-19,50,1)

# Load in all models
# ------ THIS HAS TO BE GLOBAL!

# Read in SALT2 SN IA file from Lou
salt2_spec = np.genfromtxt(fitting_utils + "salt2_template_0.txt", \
    dtype=None, names=['day', 'lam', 'flam'], encoding='ascii')

# Also load in lookup table for luminosity distance
dl_cat = np.genfromtxt(fitting_utils + 'dl_lookup_table.txt', dtype=None, names=True)
# Get arrays 
dl_z_arr = np.asarray(dl_cat['z'], dtype=np.float64)
dl_cm_arr = np.asarray(dl_cat['dl_cm'], dtype=np.float64)

del dl_cat

print("Done loading all models. Time taken:", "{:.3f}".format(time.time()-start), "seconds.")

# This class came from stackoverflow
# SEE: https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-python
class bcolors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

@jit(nopython=True)
def apply_redshift(restframe_wav, restframe_lum, redshift):

    adiff = np.abs(dl_z_arr - redshift)
    z_idx = np.argmin(adiff)
    dl = dl_cm_arr[z_idx]
    
    #dl = luminosity_distance(redshift)  # returns dl in Mpc
    #dl = dl * 3.086e24  # convert to cm

    redshifted_wav = restframe_wav * (1 + redshift)
    redshifted_flux = restframe_lum / (4 * np.pi * dl * dl * (1 + redshift))

    return redshifted_wav, redshifted_flux

def get_snr(wav, flux):

    spectrum1d_wav = wav * u.AA
    spectrum1d_flux = flux * u.erg / (u.cm * u.cm * u.s * u.AA)
    spec1d = Spectrum1D(spectral_axis=spectrum1d_wav, flux=spectrum1d_flux)

    return snr_derived(spec1d)

def loglike_sn(theta, x, data, err):
    
    z, day, av = theta

    y = model_sn(x, z, day, av)
    #print("Model SN func result:", y)

    # ------- Clip all arrays to where grism sensitivity is >= 25%
    # then get the log likelihood
    #x0 = np.where( (x >= grism_sens_wav[grism_wav_idx][0]  ) &
    #               (x <= grism_sens_wav[grism_wav_idx][-1] ) )[0]

    #y = y[x0]
    #data = data[x0]
    #err = err[x0]
    #x = x[x0]

    # ------- Vertical scaling factor
    #alpha = np.nansum(data * y / err**2) / np.nansum(y**2 / err**2)
    #print("Alpha SN:", "{:.2e}".format(alpha))
    #y = y * alpha
    y = get_y_alpha(y, data, err)

    #lnLike = -0.5 * np.nansum( (y-data)**2/err**2 ) #  +  np.log(2 * np.pi * err**2))

    lnLike = get_lnLike(y, data, err)

    #print("Chi2 term:", np.sum((y-data)**2/err**2))
    #print("Second loglikelihood term:", np.nansum( np.log(2 * np.pi * err**2)) )
    #print("ln(likelihood) SN", lnLike)

    """
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'$\lambda\, [\mathrm{\AA}]$', fontsize=14)
    ax.set_ylabel(r'$f_\lambda\, [\mathrm{cgs}]$', fontsize=14)

    ax.plot(x, data, color='k')
    ax.fill_between(x, data - err, data + err, color='gray', alpha=0.5)

    ax.plot(x, y, color='firebrick')

    plt.show()
    #sys.exit(0)
    """

    return lnLike

def logprior_sn(theta):

    z, day, av = theta
    #print("\nParameter vector given:", theta)

    if ( 0.0001 <= z <= 3.0  and  -19 <= day <= 50  and  0.0 <= av <= 5.0):
        return 0.0
    
    return -np.inf

def logpost_sn(theta, x, data, err):

    lp = logprior_sn(theta)

    #print("SN prior:", lp)
    
    if not np.isfinite(lp):
        return -np.inf
    
    lnL = loglike_sn(theta, x, data, err)

    #print("SN log(likelihood):", lnL)
    
    return lp + lnL

def model_sn(x, z, day, sn_av):

    # pull out spectrum for the chosen day
    day_idx_ = np.argmin(abs(sn_day_arr - day))
    day_idx = np.where(salt2_spec['day'] == sn_day_arr[day_idx_])[0]

    sn_spec_llam = salt2_spec['flam'][day_idx]
    sn_spec_lam = salt2_spec['lam'][day_idx]

    # ------ Apply dust extinction
    sn_dusty_llam = du.get_dust_atten_model(sn_spec_lam, sn_spec_llam, sn_av)

    # ------ Apply redshift
    sn_lam_z, sn_flam_z = apply_redshift(sn_spec_lam, sn_dusty_llam, z)

    # ------ Apply some LSF. 
    # This is a NUISANCE FACTOR ONLY FOR NOW
    # When we use actual SNe they will be point sources.
    #lsf_sigma = 0.5
    #sn_flam_z = scipy.ndimage.gaussian_filter1d(input=sn_flam_z, sigma=lsf_sigma)

    sn_mod = griddata(points=sn_lam_z, values=sn_flam_z, xi=x)

    # ------ combine host light
    # some fraction to account for host contamination
    # This fraction is a free parameter
    #sn_flam_hostcomb = sn_mod  +  host_frac * host_flam

    return sn_mod

@jit(nopython=True)
def get_y_alpha(y, data, err):

    alpha = np.nansum(data * y / err**2) / np.nansum(y**2 / err**2)

    ya = y * alpha

    return ya

@jit(nopython=True)
def get_lnLike(y, data, err):

    lnLike = -0.5 * np.nansum( (y-data)**2/ err**2 ) 

    return lnLike

def read_sn_data(sn_filename):

    truth_dict = {}

    with open(sn_filename, 'r') as dat:

        all_lines = dat.readlines()

        # First initialize arrays to save spectra
        # Also get truths
        for line in all_lines:
            if 'SIM_REDSHIFT_CMB:' in line:
                true_z = float(line.split()[1])

            if 'SIM_PEAKMJD:' in line:
                true_peakmjd = float(line.split()[1])

            if 'NSPECTRA:' in line:
                nspectra = int(line.split()[1])

            if 'SPECTRUM_NLAM:' in line:
                spec_nlam = int(line.split()[1])

            if 'SPECTRUM_MJD:' in line:
                spec_mjd = float(line.split()[1])
                if spec_mjd != -9.0:
                    break

        true_phase = spec_mjd - true_peakmjd

        # Set up truth dict
        truth_dict['z'] = true_z
        truth_dict['peak_mjd'] = true_peakmjd
        truth_dict['spec_mjd'] = spec_mjd
        truth_dict['phase'] = true_phase

        # SEt up empty arrays
        l0 = np.empty((nspectra, spec_nlam))
        l1 = np.empty((nspectra, spec_nlam))
        sn_flam = np.empty((nspectra, spec_nlam))
        sn_simflam = np.empty((nspectra, spec_nlam))
        sn_ferr = np.empty((nspectra, spec_nlam))

        # NOw get spectra
        i = 0
        j = 0

        for line in all_lines:
        
            if line[:5] == 'SPEC:':
                lsp = line.split()

                l0[i,j] = float(lsp[1])
                l1[i,j] = float(lsp[2])

                sn_flam[i,j] = float(lsp[3])
                sn_ferr[i,j] = float(lsp[4])
                sn_simflam[i,j] = float(lsp[5])

                j += 1

            if 'SPECTRUM_END:' in line:
                i += 1
                j = 0

    # get the midpoints of the wav ranges
    sn_wav = (l0 + l1) / 2

    return nspectra, sn_wav, sn_flam, sn_ferr, sn_simflam, truth_dict

def read_pickle_make_plots_sn(object_type, ndim, args_obj, label_list, truth_dict):

    h5_path = 'emcee_sampler_' + object_type + '.h5'
    sampler = emcee.backends.HDFBackend(h5_path)

    samples = sampler.get_chain()
    print(f"{bcolors.CYAN}\nRead in sampler:", h5_path, f"{bcolors.ENDC}")
    print("Samples shape:", samples.shape)

    #reader = emcee.backends.HDFBackend(pkl_path.replace('.pkl', '.h5'))
    #samples = reader.get_chain()
    #tau = reader.get_autocorr_time(tol=0)

    # Get autocorrelation time
    # Discard burn-in. You do not want to consider the burn in the corner plots/estimation.
    tau = sampler.get_autocorr_time(tol=0)
    if not np.any(np.isnan(tau)):
        burn_in = int(2 * np.max(tau))
        thinning_steps = int(0.5 * np.min(tau))
    else:
        burn_in = 200
        thinning_steps = 30

    print(f"{bcolors.CYAN}")
    print("Average Tau:", np.mean(tau))
    print("Burn-in:", burn_in)
    print("Thinning steps:", thinning_steps)
    print(f"{bcolors.ENDC}")

    # construct truth arr and plot
    truth_arr = np.array([truth_dict['z'], truth_dict['phase'], 0.0])

    # plot trace
    fig1, axes1 = plt.subplots(ndim, figsize=(10, 6), sharex=True)

    for i in range(ndim):
        ax1 = axes1[i]
        ax1.plot(samples[:, :, i], "k", alpha=0.05)
        ax1.axhline(y=truth_arr[i], color='tab:red', lw=2.0)
        ax1.set_xlim(0, len(samples))
        ax1.set_ylabel(label_list[i], fontsize=15)
        ax1.yaxis.set_label_coords(-0.1, 0.5)

    axes1[-1].set_xlabel("Step number")

    fig1.savefig('emcee_trace_' + object_type + '.pdf', dpi=200, bbox_inches='tight')

    # Create flat samples
    flat_samples = sampler.get_chain(discard=burn_in, thin=thinning_steps, flat=True)
    print("\nFlat samples shape:", flat_samples.shape)

    # plot corner plot
    cq_z = corner.quantile(x=flat_samples[:, 0], q=[0.16, 0.5, 0.84])
    cq_day = corner.quantile(x=flat_samples[:, 1], q=[0.16, 0.5, 0.84])
    cq_av = corner.quantile(x=flat_samples[:, 2], q=[0.16, 0.5, 0.84])

    # print parameter estimates
    print(f"{bcolors.CYAN}")
    print("Parameter estimates:")
    print("Redshift: ", cq_z)
    print("Supernova phase [day]:", cq_day)
    print("Visual extinction [mag]:", cq_av)
    print(f"{bcolors.ENDC}")

    fig = corner.corner(flat_samples, quantiles=[0.16, 0.5, 0.84], labels=label_list, 
        label_kwargs={"fontsize": 14}, show_titles='True', title_kwargs={"fontsize": 14},
        truth_color='tab:red', truths=truth_arr, verbose=True, smooth=0.8, smooth1d=0.8,
        range=[(0.51, 0.53), (2, 8), (0.2, 0.8)])

    # Extract the axes
    axes = np.array(fig.axes).reshape((ndim, ndim))

    # Get the redshift axis
    # and edit how the errors are displayed
    ax_z = axes[0, 0]

    z_err_high = cq_z[2] - cq_z[1]
    z_err_low = cq_z[1] - cq_z[0]

    ax_z.set_title(r"$z \, =\,$" + r"${:.3f}$".format(cq_z[1]) + \
        r"$\substack{+$" + r"${:.3f}$".format(z_err_high) + r"$\\ -$" + \
        r"${:.3f}$".format(z_err_low) + r"$}$", 
        fontsize=11)

    fig.savefig('corner_' + object_type + '.pdf', dpi=200, bbox_inches='tight')

    # ------------ Plot 100 random models from the parameter space within +-1sigma of corner estimates
    # first pull out required stuff from args
    wav = args_obj[0]
    flam = args_obj[1]
    ferr = args_obj[2]

    fig3 = plt.figure(figsize=(9,4))
    ax3 = fig3.add_subplot(111)

    ax3.set_xlabel(r'$\mathrm{\lambda\ [\AA]}$', fontsize=15)
    ax3.set_ylabel(r'$\mathrm{f_\lambda\ [erg\, s^{-1}\, cm^{-2}\, \AA^{-1}]}$', fontsize=15)

    model_count = 0
    ind_list = []

    while model_count <= 100:

        ind = int(np.random.randint(len(flat_samples), size=1))
        ind_list.append(ind)

        # make sure sample has correct shape
        sample = flat_samples[ind]
        
        model_okay = 0

        sample = sample.reshape(3)

        # Get the parameters of the sample
        model_z = sample[0]
        model_day = sample[1]
        model_av = sample[2]

        # Check that the model is within +-1 sigma
        # of value inferred by corner contours
        if (model_z >= cq_z[0]) and (model_z <= cq_z[2]) and \
           (model_day >= cq_day[0]) and (model_day <= cq_day[2]) and \
           (model_av >= cq_av[0]) and (model_av <= cq_av[2]):

           model_okay = 1

        # Now plot if the model is okay
        if model_okay:

            m = model_sn(wav, sample[0], sample[1], sample[2])

            a = np.nansum(flam * m / ferr**2) / np.nansum(m**2 / ferr**2)
            m = m * a

            ax3.plot(wav, m, color='royalblue', lw=0.5, alpha=0.05, zorder=2)

            model_count += 1

    print("\nList of randomly chosen indices:", ind_list)

    ax3.plot(wav, flam, color='k', lw=1.0, zorder=1)
    ax3.fill_between(wav, flam - ferr, flam + ferr, color='gray', alpha=0.5, zorder=1)

    # ADD LEGEND
    ax3.text(x=0.65, y=0.92, s='--- Simulated data', 
        verticalalignment='top', horizontalalignment='left', 
        transform=ax3.transAxes, color='k', size=12)
    ax3.text(x=0.65, y=0.85, s='--- 100 randomly chosen samples', 
        verticalalignment='top', horizontalalignment='left', 
        transform=ax3.transAxes, color='royalblue', size=12)

    fig3.savefig('emcee_overplot_' + object_type + '.pdf', dpi=200, bbox_inches='tight')

    # close figs
    plt.clf()
    plt.cla()
    plt.close()

    return None

def main():

    # -----------------------------------------------
    snnum_arr = np.arange(1, 1001)
    snana_sn_spec_dir = home + '/Documents/sn_sit_hackday/20210325_BMR_PRISM/'

    # ---------------------------------- Set up
    # Labels for corner and trace plots
    label_list_sn = [r'$z$', r'$Day$', r'$A_V [mag]$']

    # ----------------------- Using emcee ----------------------- #
    # Set jump sizes # ONLY FOR INITIAL POSITION SETUP
    jump_size_z = 0.01
    jump_size_av = 0.1  # magnitudes
    jump_size_day = 2  # days

    zprior = 0.5
    zprior_sigma = 0.02

    #get_optimal_position()
    rsn_init = np.array([zprior, 0, 0.0])  # redshift, day relative to peak, and dust extinction

    # Setup dims and walkers
    nwalkers = 300
    ndim_sn  = 3

    # generating ball of walkers about initial position defined above
    pos_sn = np.zeros(shape=(nwalkers, ndim_sn))

    for i in range(nwalkers):

        # ---------- For SN
        rsn0 = float(rsn_init[0] + jump_size_z * np.random.normal(size=1))
        rsn1 = int(rsn_init[1] + jump_size_day * np.random.normal(size=1))
        rsn2 = float(rsn_init[2] + jump_size_av * np.random.normal(size=1))

        rsn = np.array([rsn0, rsn1, rsn2])

        pos_sn[i] = rsn

    print(f"{bcolors.GREEN}")
    print("Starting position for SN from where ball of walkers will be generated:\n")
    print(rsn_init, f"{bcolors.ENDC}")

    # Loop over all spectra files
    for j in range(len(snnum_arr)):

        snnum = 10000 + snnum_arr[j]

        nspectra_sn, sn_wav_arr, sn_flam_arr, sn_ferr_arr, sn_simflam_arr, truth_dict = \
        read_sn_data(snana_sn_spec_dir + 'BMR_PRISM_2_TEST_SN0' + str(snnum) + '.DAT')

        print("\nRead in SN spectrum", snnum)
        print("Truth values:", truth_dict)

        # confirm with Ben but it seems like index 1 is always the SN spectrum
        sn_wav = sn_wav_arr[1]
        sn_flam = sn_flam_arr[1]
        sn_ferr = sn_ferr_arr[1]
        sn_simflam = sn_simflam_arr[1]

        # Clip data at the ends
        wav_idx = np.where((sn_wav > 8000) & (sn_wav < 18000))[0]
        sn_wav = sn_wav[wav_idx]
        sn_flam = sn_flam[wav_idx]
        sn_ferr = sn_ferr[wav_idx]
        sn_simflam = sn_simflam[wav_idx]

        snr = get_snr(sn_wav, sn_flam)

        print("Signal to noise for noised spectrum:", snr)
        #print("Signal to noise for sim spectrum:", get_snr(sn_wav, sn_simflam))

        if snr < 3.0:
            print("SNR too low. Skipping this spectrum.")
            continue

        # DO NOT DELETE
        # Code block to check fig
        """
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(111)
        ax.set_xlabel(r'$\mathrm{\lambda\ [\AA]}$', fontsize=15)
        ax.set_ylabel(r'$\mathrm{f_\lambda\ [erg\, s^{-1}\, cm^{-2}\, \AA^{-1}]}$', fontsize=15)
        ax.plot(sn_wav, sn_flam, lw=2.0, color='k')
        ax.fill_between(sn_wav, sn_flam - sn_ferr, sn_flam + sn_ferr, color='gray', alpha=0.5)
        ax.plot(sn_wav, sn_simflam, lw=2.0, color='firebrick')

        # now get truth model
        t = model_sn(sn_wav, truth_dict['z'], truth_dict['phase'], 0.0)
        t = get_y_alpha(t, sn_flam, sn_ferr)
        ax.plot(sn_wav, t, lw=2.5, color='forestgreen')

        # also check what corner thinks the truth is
        #t1 = model_sn(sn_wav, 0.57, -11.0, 0.0)
        #t1 = get_y_alpha(t1, sn_flam, sn_ferr)
        #ax.plot(sn_wav, t1, lw=2.5, color='dodgerblue')

        plt.show()

        if j > 20: sys.exit(0)
        else: continue
        """

        # Set up args
        args_sn = [sn_wav, sn_flam, sn_ferr]

        print("logpost at starting position for SN:", logpost_sn(rsn_init, sn_wav, sn_flam, sn_ferr))

        # --------------------------------------------------
        # Now run on SN
        emcee_savefile = 'emcee_sampler_sn' + str(snnum) + '.h5'
        if not os.path.isfile(emcee_savefile):
            backend = emcee.backends.HDFBackend(emcee_savefile)
            backend.reset(nwalkers, ndim_sn)

            with Pool() as pool:
                
                sampler = emcee.EnsembleSampler(nwalkers, ndim_sn, logpost_sn, 
                    args=args_sn, pool=pool, backend=backend, \
                    moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2),])
                sampler.run_mcmc(pos_sn, 2000, progress=True)

            print("Finished running emcee.")
            print("Mean acceptance Fraction:", np.mean(sampler.acceptance_fraction), "\n")

        read_pickle_make_plots_sn('sn' + str(snnum), ndim_sn, args_sn, label_list_sn, truth_dict)

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)










