import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import griddata
from scipy.interpolate import splev, splrep
import emcee 
import corner

from multiprocessing import Pool
import pickle

import os
import sys
import socket
import time
import datetime as dt

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
astropy_cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

start = time.time()
print("Starting at:", dt.datetime.now())

# Assign directories and custom imports
home = os.getenv('HOME')
stacking_utils = home + '/Documents/GitHub/stacking-analysis-pears/util_codes/'

roman_slitless_dir = home + "/Documents/GitHub/roman-slitless/"
ext_spectra_dir = home + "/Documents/roman_slitless_sims_results/"
roman_sims_seds = home + "/Documents/roman_slitless_sims_seds/"
emcee_diagnostics_dir = home + "/Documents/emcee_runs/emcee_diagnostics_roman/"

sys.path.append(stacking_utils)
import proper_and_lum_dist as cosmo
from dust_utils import get_dust_atten_model
from bc03_utils import get_age_spec

grism_sens_cat = np.genfromtxt(home + '/Documents/pylinear_ref_files/pylinear_config/Roman/roman_throughput_20190325.txt', \
    dtype=None, names=True, skip_header=3)

grism_sens_wav = grism_sens_cat['Wave'] * 1e4  # the text file has wavelengths in microns # needed in angstroms
grism_sens = grism_sens_cat['BAOGrism_1st']
grism_wav_idx = np.where(grism_sens > 0.25)

# Load in all models
# ------ THIS HAS TO BE GLOBAL!

# Read in SALT2 SN IA file from Lou
salt2_spec = np.genfromtxt(roman_sims_seds + "salt2_template_0.txt", \
    dtype=None, names=['day', 'lam', 'flam'], encoding='ascii')

if 'plffsn2' in socket.gethostname():
    extdir = '/astro/ffsn/Joshi/'
    modeldir = extdir + 'bc03_output_dir/'
else:
    extdir = '/Volumes/Joshi_external_HDD/Roman/'
    modeldir = extdir + 'bc03_output_dir/m62/'

assert os.path.isdir(modeldir)

model_lam = np.load(extdir + "bc03_output_dir/bc03_models_wavelengths.npy", mmap_mode='r')
model_ages = np.load(extdir + "bc03_output_dir/bc03_models_ages.npy", mmap_mode='r')

all_m62_models = []
tau_low = 0
tau_high = 20
for t in range(tau_low, tau_high, 1):
    tau_str = "{:.3f}".format(t).replace('.', 'p')
    a = np.load(modeldir + 'bc03_all_tau' + tau_str + '_m62_chab.npy', mmap_mode='r')
    all_m62_models.append(a)
    del a

# load models with large tau separately
all_m62_models.append(np.load(modeldir + 'bc03_all_tau20p000_m62_chab.npy', mmap_mode='r'))

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

def logpost_host(theta, x, data, err):

    lp = logprior_host(theta)
    #print("Prior HOST:", lp)
    
    if not np.isfinite(lp):
        return -np.inf
    
    lnL = loglike_host(theta, x, data, err)

    #print("Likelihood HOST:", lnL)
    
    return lp + lnL

"""
def logprior_host(theta, *priorargs, zprior_flag=False):

    z, age, logtau, av = theta
    #print("\nParameter vector given:", theta)

    if zprior_flag:

        zprior, zprior_sigma = args[0], args[1]

        if (0.0001 <= z <= 6.0):
    
            # Make sure model is not older than the Universe
            # Allowing at least 100 Myr for the first galaxies to form after Big Bang
            age_at_z = astropy_cosmo.age(z).value  # in Gyr
            age_lim = age_at_z - 0.1  # in Gyr

            if ((0.01 <= age <= age_lim) and \
                (-3.0 <= logtau <= 2.0) and \
                (0.0 <= av <= 5.0)):

                # Gaussian prior on redshift
                ln_pz = np.log( 1.0 / (np.sqrt(2*np.pi)*zprior_sigma) ) - 0.5*(z - zprior)**2/zprior_sigma**2

                return ln_pz
    
    else:

        if (0.0001 <= z <= 6.0):
    
            # Make sure model is not older than the Universe
            # Allowing at least 100 Myr for the first galaxies to form after Big Bang
            age_at_z = astropy_cosmo.age(z).value  # in Gyr
            age_lim = age_at_z - 0.1  # in Gyr

            if ((0.01 <= age <= age_lim) and \
                (-3.0 <= logtau <= 2.0) and \
                (0.0 <= av <= 5.0)):

                return 0.0

    return -np.inf
"""

def logprior_host(theta):

    z, age, logtau = theta
    #print("\nParameter vector given:", theta)

    if (0.0001 <= z <= 6.0):
    
        # Make sure model is not older than the Universe
        # Allowing at least 100 Myr for the first galaxies to form after Big Bang
        age_at_z = astropy_cosmo.age(z).value  # in Gyr
        age_lim = age_at_z - 0.1  # in Gyr

        if ((0.01 <= age <= age_lim) and \
            (-3.0 <= logtau <= 2.0)):

            return 0.0

    return -np.inf

def loglike_host(theta, x, data, err):
    
    z, age, logtau = theta

    y = model_host(x, z, age, logtau)
    #print("Model func result:", y)

    # ------- Clip all arrays to where grism sensitivity is >= 25%
    # then get the log likelihood
    x0 = np.where( (x >= grism_sens_wav[grism_wav_idx][0]  ) &
                   (x <= grism_sens_wav[grism_wav_idx][-1] ) )[0]

    y = y[x0]
    data = data[x0]
    err = err[x0]
    x = x[x0]

    # ------- Vertical scaling factor
    #try:
    #alpha = np.nansum(data * y / err**2) / np.nansum(y**2 / err**2)
    #print("Alpha HOST:", "{:.2e}".format(alpha))
    #except RuntimeWarning:
    #    print("RuntimeWarning encountered.")
    #    print("Parameter vector given:", theta)
    #    sys.exit(0)

    #y = y * alpha

    # ------- log likelihood
    #chi2 = np.nansum( (y-data)**2/err**2 ) / len(y)
    lnLike = -0.5 * np.nansum( (y-data)**2/err**2 ) - 0.5 * np.nansum( np.log(2 * np.pi * err**2) )
    #stretch_fac = 10.0
    #lnLike = -0.5 * (1 + stretch_fac) * chi2

    #print("Pure chi2 term:", np.nansum( (y-data)**2/err**2 ))
    #print("Second error term:", np.nansum( np.log(2 * np.pi * err**2) ))
    #print("log likelihood HOST:", lnLike)

    """
    fig = plt.figure(figsize=(9,5))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'$\lambda\, [\mathrm{\AA}]$', fontsize=14)
    ax.set_ylabel(r'$f_\lambda\, [\mathrm{cgs}]$', fontsize=14)

    ax.plot(x, data, color='k')
    ax.fill_between(x, data - err, data + err, color='gray', alpha=0.5)

    ax.plot(x, y, color='firebrick')

    ax.set_xscale('log')
    plt.show()
    sys.exit(0)
    """

    return lnLike

def model_host(x, z, age, logtau):
    """
    Expects to get the following arguments
    
    x: observed wavelength grid
    
    z: redshift to apply to template
    ms: log of the stellar mass
    age: age of SED in Gyr
    tau: exponential SFH timescale in Gyr
    metallicity: absolute fraction of metals
    av: visual dust extinction
    """

    metals = 0.02

    tau = 10**logtau  # logtau is log of tau in gyr

    #print("log(tau [Gyr]):", logtau)
    #print("Tau [Gyr]:", tau)
    #print("Age [Gyr]:", age)

    if tau < 20.0:

        tau_int_idx = int((tau - int(np.floor(tau))) * 1e3)
        age_idx = np.argmin(abs(model_ages - age*1e9))
        model_idx = tau_int_idx * len(model_ages)  +  age_idx

        models_taurange_idx = np.argmin(abs(np.arange(tau_low, tau_high, 1) - int(np.floor(tau))))
        models_arr = all_m62_models[models_taurange_idx]

        #print("Tau int and age index:", tau_int_idx, age_idx)
        #print("Tau and age from index:", models_taurange_idx+tau_int_idx/1e3, model_ages[age_idx]/1e9)
        #print("Model tau range index:", models_taurange_idx)

    elif tau >= 20.0:
        
        logtau_arr = np.arange(1.30, 2.01, 0.01)
        logtau_idx = np.argmin(abs(logtau_arr - logtau))

        age_idx = np.argmin(abs(model_ages - age*1e9))
        model_idx = logtau_idx * len(model_ages) + age_idx

        models_arr = all_m62_models[-1]

        #print("logtau and age index:", logtau_idx, age_idx)
        #print("Tau and age from index:", 10**(logtau_arr[logtau_idx]), model_ages[age_idx]/1e9)

    #print("Model index:", model_idx)

    model_llam = models_arr[model_idx]

    # ------ Apply dust extinction
    #model_dusty_llam = get_dust_atten_model(model_lam, model_llam, av)

    # ------ Multiply luminosity by stellar mass
    #model_dusty_llam = model_dusty_llam * 10**ms

    # ------ Apply redshift
    model_lam_z, model_flam_z = cosmo.apply_redshift(model_lam, model_llam, z)
    Lsol = 3.826e33
    model_flam_z = Lsol * model_flam_z

    # ------ Apply LSF
    model_lsfconv = gaussian_filter1d(input=model_flam_z, sigma=1.0)

    # ------ Downgrade to grism resolution
    model_mod = griddata(points=model_lam_z, values=model_lsfconv, xi=x)

    model_err = np.zeros(len(x))
    model_cont_norm, model_err_cont_norm = divcont(x, model_mod, model_err, showplot=False)

    return model_cont_norm

def divcont(wav, flux, ferr, showplot=False):

    # Normalize flux levels to approx 1.0
    flux_norm = flux / np.mean(flux)
    ferr_norm = ferr / np.mean(flux)

    # Mask lines
    #mask_indices = get_mask_indices(wav, zprior)

    # Make sure masking indices are consistent with array to be masked
    #remove_mask_idx = np.where(mask_indices >= len(wav))[0]
    #mask_indices = np.delete(arr=mask_indices, obj=remove_mask_idx)

    #weights = np.ones(len(wav))
    #mask_indices = np.array([483, 484, 485, 486, 487, 488, 489])
    # the above indices are manually done as a test for masking H-beta
    #weights[mask_indices] = 0

    # SciPy smoothing spline fit
    spl = splrep(x=wav, y=flux_norm, k=3, s=5.0)
    wav_plt = np.arange(wav[0], wav[-1], 1.0)
    spl_eval = splev(wav_plt, spl)

    # Divide the given flux by the smooth spline fit and return
    cont_div_flux = flux_norm / splev(wav, spl)
    cont_div_err  = ferr_norm / splev(wav, spl)

    # Test figure showing fits
    if showplot:
        fig = plt.figure(figsize=(10,6))
        gs = gridspec.GridSpec(5,1)
        gs.update(left=0.06, right=0.95, bottom=0.1, top=0.9, wspace=0.00, hspace=0.5)

        ax1 = fig.add_subplot(gs[:3,:])
        ax2 = fig.add_subplot(gs[3:,:])

        ax1.set_ylabel(r'$\mathrm{Flux\ [normalized]}$', fontsize=15)
        ax2.set_ylabel(r'$\mathrm{Continuum\ divided\ flux}$', fontsize=15)
        ax2.set_xlabel(r'$\mathrm{Wavelength\ [\AA]}$', fontsize=15)

        ax1.plot(wav, flux_norm, color='k')
        ax1.fill_between(wav, flux_norm - ferr_norm, flux_norm + ferr_norm, color='gray', alpha=0.5)
        ax1.plot(wav_plt, spl_eval, color='crimson', lw=3.0, label='SciPy smooth spline fit')

        ax2.plot(wav, cont_div_flux, color='teal', lw=2.0, label='Continuum divided flux')
        ax2.axhline(y=1.0, ls='--', color='k', lw=1.8)

        # Tick label sizes
        ax1.tick_params(which='both', labelsize=14)
        ax2.tick_params(which='both', labelsize=14)

        plt.show()

    return cont_div_flux, cont_div_err

def air2vac():

    # Conversion between air and vacuum wavelengths
    # from here: http://classic.sdss.org/dr3/products/spectra/vacwavelength.html
    # they got it from Morton 1991, ApJS, 77, 119
    # Vacuum wavelengths in angstroms
    # air = vac / (1.0 + 2.735182e-4 + 131.4182 / vac**2 + 2.76249e8 / vac**4)
    pass

def gen_balmer_lines():

    # Check the latest Rydberg constant data here
    # https://physics.nist.gov/cgi-bin/cuu/Value?ryd|search_for=Rydberg
    # short list here for quick reference: https://physics.nist.gov/cuu/pdf/wall_2018.pdf
    rydberg_const = 10973731.568  # in m^-1

    balmer_line_wav_list = []

    for lvl in range(3, 15):

        energy_levels_term = (1/4) - (1/lvl**2)
        lam_vac = (1/rydberg_const) * (1/energy_levels_term)

        lam_vac_ang = lam_vac*1e10  # meters to angstroms # since the rydberg const above is in (1/m)

        #print("Transition:", lvl, "--> 2,       wavelength in vacuum [Angstroms]:", "{:.3f}".format(lam_vac_ang))

        balmer_line_wav_list.append(lam_vac_ang)

    return balmer_line_wav_list

def get_mask_indices(obs_wav, redshift):

    # Define rest-frame wavelengths in vacuum
    # Emission or absorption doesn't matter
    gband = 4300
    #hbeta = 4862.72
    oiii4959 = 4960.295
    oiii5007 = 5008.239
    mg2_mgb = 5175
    fe5270 = 5270
    fe5335 = 5335
    fe5406 = 5406
    nad = 5890
    #halpha = 6564.614

    all_balmer_lines = gen_balmer_lines()

    all_lines = [gband, oiii4959, oiii5007]
    all_lines = all_lines + all_balmer_lines

    # Set up empty array for masking indices
    mask_indices = []

    # Loop over all lines and get masking indices
    for line in all_lines:

        obs_line_wav = (1 + redshift) * line
        #print(obs_line_wav)
        if (obs_line_wav >= obs_wav[0]) and (obs_line_wav <= obs_wav[-1]):
            closest_obs_wav_idx = np.argmin(abs(obs_wav - obs_line_wav))

            #print(line, "  ", redshift, "  ", obs_line_wav, "  ", closest_obs_wav_idx)

            mask_indices.append(np.arange(closest_obs_wav_idx-3, closest_obs_wav_idx+4))

    # Convert to numpy array
    mask_indices = np.asarray(mask_indices)
    mask_indices = mask_indices.ravel()

    # Make sure the returned indices are unique and sorted
    mask_indices = np.unique(mask_indices)

    return mask_indices

def main():

    print(f"{bcolors.WARNING}")
    print("* * * *   [WARNING]: the supplied vacuum wavelengths are a bit off from those ")
    print("          typically quoted for the Balmer lines. Needs to be checked.  * * * *")
    print("* * * *   [WARNING]: model has worse resolution than data in NIR. np.mean() will result in nan. Needs fixing.   * * * *")
    print(f"{bcolors.ENDC}")

    ext_root = "romansim1"
    img_suffix = 'Y106_11_1'

    # Read in sed.lst
    sedlst_header = ['segid', 'sed_path']
    sedlst_path = roman_slitless_dir + 'pylinear_lst_files/' + 'sed_plffsn2_' + img_suffix + '.lst'
    sedlst = np.genfromtxt(sedlst_path, dtype=None, names=sedlst_header, encoding='ascii')
    print("Read in sed.lst from:", sedlst_path)

    # Read in the extracted spectra
    ext_spec_filename = ext_spectra_dir + 'plffsn2_run_dec7/' + ext_root + '_ext_x1d.fits'
    ext_hdu = fits.open(ext_spec_filename)
    print("Read in extracted spectra from:", ext_spec_filename)

    # Set pylinear f_lambda scaling factor
    pylinear_flam_scale_fac = 1e-17

    # ---- Get data and test fitting
    hostid = 207

    host_wav = ext_hdu[('SOURCE', hostid)].data['wavelength']
    host_flam = ext_hdu[('SOURCE', hostid)].data['flam'] * pylinear_flam_scale_fac

    # ---- Apply noise and get dummy noisy spectra
    noise_level = 0.03  # relative to signal

    host_ferr = noise_level * host_flam

    # ---- fitting
    #zprior = 1.95
    #zprior_sigma = 0.02
    rhost_init = np.array([1.96, 1.0, 1.1])

    # Divide by continuum
    # In this call you want to see the plot showing the fit
    # because you need the assess how good your redshift prior is
    host_flam_cont_norm, host_ferr_cont_norm = divcont(host_wav, host_flam, host_ferr, showplot=False)

    """
    # Test figure showing pdf for redshift prior
    fig = plt.figure()
    ax = fig.add_subplot(111)

    z_arr = np.arange(0.001, 6.001, 0.001)
    for z in z_arr:
        pdf_z = ( 1.0 / (np.sqrt(2*np.pi)*zprior_sigma) ) * np.exp(-0.5*(z - zprior)**2/zprior_sigma**2)
        print("{:.3f}".format(z), "      ", "{:.3e}".format(pdf_z))

    #ln_pz = np.log( 1.0 / (np.sqrt(2*np.pi)*zprior_sigma) ) - 0.5*(z_arr - zprior)**2/zprior_sigma**2
    pdf_z = ( 1.0 / (np.sqrt(2*np.pi)*zprior_sigma) ) * np.exp(-0.5*(z_arr - zprior)**2/zprior_sigma**2)

    ax.plot(z_arr, pdf_z)
    #ax.set_yscale('log')
    plt.show()
    sys.exit(0)
    """

    # now call posterior func to test
    #logpost_host(rhost_init, host_wav, host_flam_cont_norm, host_ferr_cont_norm, zprior, zprior_sigma)

    # --------------------------------- Emcee run
    nwalkers, ndim = 300, 3

    # Set jump sizes # ONLY FOR INITIAL POSITION SETUP
    pos_host = np.zeros(shape=(nwalkers, ndim))

    jump_size_z = 0.01
    jump_size_age = 0.1  # in gyr
    jump_size_logtau = 0.01  # tau in gyr
    #jump_size_av = 0.1  # magnitudes

    for i in range(nwalkers):

        # ---------- For HOST
        rh0 = float(rhost_init[0] + jump_size_z * np.random.normal(size=1))
        rh1 = float(rhost_init[1] + jump_size_age * np.random.normal(size=1))
        rh2 = float(rhost_init[2] + jump_size_logtau * np.random.normal(size=1))
        #rh3 = float(rhost_init[3] + jump_size_av * np.random.normal(size=1))

        rh = np.array([rh0, rh1, rh2])#, rh3])

        pos_host[i] = rh

    # Setup arguments for posterior function
    args_host = [host_wav, host_flam_cont_norm, host_ferr_cont_norm]

    print("Running on:", hostid)

    # ----------- Set up the HDF5 file to incrementally save progress to
    emcee_savefile = emcee_diagnostics_dir +'emcee_sampler_' + str(hostid) + '_contdivtest.h5'

    backend = emcee.backends.HDFBackend(emcee_savefile)
    backend.reset(nwalkers, ndim)

    # ----------- Emcee 
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logpost_host, args=args_host, pool=pool, backend=backend)
        #moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2),],)
        sampler.run_mcmc(pos_host, 1000, progress=True)

    # ----------- Also save the final result as a pickle dump
    pickle.dump(sampler, open(emcee_savefile.replace('.h5','.pkl'), 'wb'))

    print("Done with fitting.")

    # ---------------------------------------- Plot results
    #  r'$\mathrm{log(M_s/M_\odot)}$',
    label_list = [r'$z$', r'$\mathrm{Age\, [Gyr]}$', r'$\mathrm{\log(\tau\, [Gyr])}$', r'$A_V [mag]$']

    sampler = emcee.backends.HDFBackend(emcee_savefile)

    samples = sampler.get_chain()
    print(f"{bcolors.CYAN}\nRead in sampler:", emcee_savefile, f"{bcolors.ENDC}")
    print("Samples shape:", samples.shape)

    # Get autocorrelation time
    # Discard burn-in. You do not want to consider the burn in the corner plots/estimation.
    tau = sampler.get_autocorr_time(tol=0)
    burn_in = int(2 * np.max(tau))
    thinning_steps = int(0.5 * np.min(tau))

    print(f"{bcolors.WARNING}")
    #print("Acceptance Fraction:", sampler.acceptance_fraction, "\n")
    print("Average Tau:", np.mean(tau))
    print("Burn-in:", burn_in)
    print("Thinning steps:", thinning_steps)
    print(f"{bcolors.ENDC}")

    # plot trace
    fig1, axes1 = plt.subplots(ndim, figsize=(10, 6), sharex=True)

    for i in range(ndim):
        ax1 = axes1[i]
        ax1.plot(samples[:, :, i], "k", alpha=0.05)
        ax1.set_xlim(0, len(samples))
        ax1.set_ylabel(label_list[i])
        ax1.yaxis.set_label_coords(-0.1, 0.5)

    axes1[-1].set_xlabel("Step number")

    fig1.savefig(emcee_diagnostics_dir + 'emcee_trace_' + str(hostid) + '_' + img_suffix + '_contdivtest.pdf', \
        dpi=200, bbox_inches='tight')

    # Create flat samples
    flat_samples = sampler.get_chain(discard=burn_in, thin=thinning_steps, flat=True)
    print("\nFlat samples shape:", flat_samples.shape)

    # Get truths
    # ---- HOST
    h_idx = int(np.where(sedlst['segid'] == hostid)[0])
    h_path = sedlst['sed_path'][h_idx]
    th = os.path.basename(h_path)
    print("Template name HOST:", th)

    th = th.split('.txt')[0].split('_')

    host_av = float(th[-1].replace('p', '.').replace('av',''))
    host_met = float(th[-2].replace('p', '.').replace('met',''))
    host_tau = float(th[-3].replace('p', '.').replace('tau',''))
    host_age = float(th[-4].replace('p', '.').replace('age',''))
    host_ms = float(th[-5].replace('p', '.').replace('ms',''))
    host_z = float(th[-6].replace('p', '.').replace('z',''))

    truth_arr = np.array([host_z, host_age, np.log10(host_tau), host_av])

    #print(f"{bcolors.WARNING}\nUsing hardcoded ranges in corner plot.{bcolors.ENDC}")
    fig = corner.corner(flat_samples, quantiles=[0.16, 0.5, 0.84], labels=label_list, \
        label_kwargs={"fontsize": 14}, show_titles='True', title_kwargs={"fontsize": 14}, truths=truth_arr, \
        verbose=True, truth_color='tab:red', smooth=0.7, smooth1d=0.7)#, \
    #range=[(1.952, 1.954), (12.5, 13.2), (0.5, 1.0), (-0.6, 0.6), (0.5, 0.9)] )
    fig.savefig(emcee_diagnostics_dir + 'corner_' + str(hostid) + '_' + img_suffix + '_contdivtest.pdf', \
        dpi=200, bbox_inches='tight')

    # ------------ Plot 100 random models from the parameter space
    inds = np.random.randint(len(flat_samples), size=100)

    fig3 = plt.figure(figsize=(9,4))
    ax3 = fig3.add_subplot(111)

    ax3.set_xlabel(r'$\mathrm{\lambda\ [\AA]}$', fontsize=15)
    ax3.set_ylabel(r'$\mathrm{f_\lambda\ [normalized]}$', fontsize=15)

    for ind in inds:
        sample = flat_samples[ind]

        m = model_host(host_wav, sample[0], sample[1], sample[2])

        # ------- Clip all arrays to where grism sensitivity is >= 25%
        x0 = np.where( (host_wav >= grism_sens_wav[grism_wav_idx][0]  ) &
                       (host_wav <= grism_sens_wav[grism_wav_idx][-1] ) )[0]

        m = m[x0]
        host_wav = host_wav[x0]
        host_flam_cont_norm = host_flam_cont_norm[x0]
        host_ferr_cont_norm = host_ferr_cont_norm[x0]

        ax3.plot(host_wav, m, color='tab:red', alpha=0.2, zorder=2)

    ax3.plot(host_wav, host_flam_cont_norm, color='k', zorder=3)
    ax3.fill_between(host_wav, host_flam_cont_norm - host_ferr_cont_norm, host_flam_cont_norm + host_ferr_cont_norm, \
        color='gray', alpha=0.5, zorder=3)

    fig3.savefig(emcee_diagnostics_dir + 'emcee_overplot_' + str(hostid) + '_' + img_suffix + '_contdivtest.pdf', \
        dpi=200, bbox_inches='tight')

    return None



if __name__ == '__main__':
    main()
    sys.exit(0)










