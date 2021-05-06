import numpy as np

from astropy.io import fits
from astropy import units as u
from specutils import Spectrum1D
from specutils.analysis import snr, snr_derived
from astropy.stats import mad_std

from scipy.interpolate import griddata
import emcee
import corner

import os
import sys
import glob

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

home = os.getenv('HOME')
sn_fit_results_dir = home + '/Documents/sn_sit_hackday/'
snana_sn_spec_dir = home + '/Documents/sn_sit_hackday/20210325_BMR_PRISM/'

gal_fit_results_dir = home + '/Desktop/Prism_shallow_hostIa/results/'
snana_gal_spec_dir = home + '/Desktop/Prism_shallow_hostIa/'

roman_sims_seds = home + "/Documents/roman_slitless_sims_seds/"
stacking_utils = home + '/Documents/GitHub/stacking-analysis-pears/util_codes/'
roman_slitless_dir = home + "/Documents/GitHub/roman-slitless/"

sys.path.append(stacking_utils)
import dust_utils as du
from fit_galaxy import read_galaxy_data
from fit_sn import read_sn_data

# Define any required constants/arrays
Lsol = 3.826e33
sn_day_arr = np.arange(-19,50,1)

# Load in all models
# ------ THIS HAS TO BE GLOBAL!

# Read in SALT2 SN IA file from Lou
salt2_spec = np.genfromtxt(roman_sims_seds + "salt2_template_0.txt", \
    dtype=None, names=['day', 'lam', 'flam'], encoding='ascii')

# Also load in lookup table for luminosity distance
dl_cat = np.genfromtxt(stacking_utils + 'dl_lookup_table.txt', dtype=None, names=True)
# Get arrays 
dl_z_arr = np.asarray(dl_cat['z'], dtype=np.float64)
dl_cm_arr = np.asarray(dl_cat['dl_cm'], dtype=np.float64)

del dl_cat

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

def get_y_alpha(y, data, err):

    alpha = np.nansum(data * y / err**2) / np.nansum(y**2 / err**2)

    ya = y * alpha

    return ya

def get_lnLike(y, data, err):

    chi2 = np.nansum( (y-data)**2/err**2 )
    lnLike = -0.5 * chi2

    return lnLike, chi2

def main():

    runtype = 'galaxy'
    # -------------

    # Set up empty lists
    zerr_low_list = []
    zerr_up_list = []
    z_corner = []
    z_truth = []

    snr_arr = []
    z_acc = []
    z_err = []

    phase_acc = []
    phase_err = []
    phase_corner = []
    phase_truth = []

    fail_id_list = []

    # Set up paths correctly 
    if runtype == 'galaxy':
        fit_results_dir = gal_fit_results_dir
    else:
        fit_results_dir = sn_fit_results_dir

    # Loop over all results
    for fl in glob.glob(fit_results_dir + '*.h5'):

        # Because the fitting program is running
        # Dont accept incomplete sampler.h5 files
        flsize = os.stat(fl).st_size / 1e6  # MB
        if flsize < 30:  # full sampler size is 33.6 MB for SNe and 32.9 for Galaxies
            print("Incomplete sampler:", fl)
            print("Skipping for now.")
            continue

        # Get data and truths
        flbasename = os.path.basename(fl)
        if runtype == 'galaxy':
            dat_file = fl.replace('.h5','.DAT')
            dat_file = dat_file.replace('results/emcee_sampler_', 'Prism_shallow_hostIa_SN0')
            
            nspectra, gal_wav, gal_flam, gal_ferr, gal_simflam, truth_dict = read_galaxy_data(dat_file)

            obj_wav = gal_wav
            obj_flam = gal_flam
            obj_ferr = gal_ferr
            obj_simflam = gal_simflam

            dat_name_base = os.path.basename(dat_file).split('.DAT')[0]
            galid = int(dat_name_base.split('_')[-1].lstrip('SN'))

            objid = galid

        else:
            snnum = int(flbasename.split('.')[0].split('_')[-1].split('sn')[1])
            nspectra_sn, sn_wav_arr, sn_flam_arr, sn_ferr_arr, sn_simflam_arr, truth_dict = \
            read_sn_data(snana_sn_spec_dir + 'BMR_PRISM_2_TEST_SN0' + str(snnum) + '.DAT')

            # confirm with Ben but it seems like index 1 is always the SN spectrum
            sn_wav = sn_wav_arr[1]
            sn_flam = sn_flam_arr[1]
            sn_ferr = sn_ferr_arr[1]
            sn_simflam = sn_simflam_arr[1]

            obj_wav = sn_wav
            obj_flam = sn_flam
            obj_ferr = sn_ferr
            obj_simflam = sn_simflam

            objid = snnum

        # Clip data at the ends
        wav_idx = np.where((obj_wav > 7600) & (obj_wav < 17800))[0]

        obj_wav = obj_wav[wav_idx]
        obj_flam = obj_flam[wav_idx]
        obj_ferr = obj_ferr[wav_idx]
        obj_simflam = obj_simflam[wav_idx]

        snr = get_snr(obj_wav, obj_flam)

        snr_arr.append(snr)

        # Now get the redshift recovery stats
        sampler = emcee.backends.HDFBackend(fl)

        samples = sampler.get_chain()
        print("Working on sampler:", fl)

        # Get autocorrelation time
        # Discard burn-in. You do not want to consider the burn in the corner plots/estimation.
        tau = sampler.get_autocorr_time(tol=0)
        if not np.any(np.isnan(tau)):
            burn_in = int(2 * np.max(tau))
            thinning_steps = int(0.5 * np.min(tau))
        else:
            burn_in = 50
            thinning_steps = 5

        # Create flat samples
        flat_samples = sampler.get_chain(discard=burn_in, thin=thinning_steps, flat=True)

        # plot corner plot
        cq_z = corner.quantile(x=flat_samples[:, 0], q=[0.16, 0.5, 0.84])

        if runtype == 'galaxy':
            cq_mass = corner.quantile(x=flat_samples[:, 1], q=[0.16, 0.5, 0.84])
        else:
            cq_day = corner.quantile(x=flat_samples[:, 1], q=[0.16, 0.5, 0.84])
            cq_av = corner.quantile(x=flat_samples[:, 2], q=[0.16, 0.5, 0.84])

        # Now get redshift and accuracy
        zacc = (cq_z[1] - truth_dict['z']) / (1 + truth_dict['z'])
        z_acc.append(zacc)

        zerr_up = cq_z[2] - cq_z[1]
        zerr_low = cq_z[1] - cq_z[0]

        z_err.append([zerr_low, zerr_up])

        zerr_low_list.append(zerr_low)
        zerr_up_list.append(zerr_up)
        z_corner.append(cq_z[1])
        z_truth.append(truth_dict['z'])

        # phase stuff
        if runtype == 'sn':
            phase_acc.append(cq_day[1] - truth_dict['phase'])

            phase_err_up = cq_day[2] - cq_day[1]
            phase_err_low = cq_day[1] - cq_day[0]

            phase_err.append([phase_err_low, phase_err_up])

            phase_corner.append(cq_day[1])
            phase_truth.append(truth_dict['phase'])

        # if catastrophic failure then identify
        if np.abs(zacc) >= 0.1:
            fail_id_list.append(objid)

        # # 
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
        t1 = model_sn(sn_wav, cq_z[1], cq_day[1], cq_av[1])
        t1 = get_y_alpha(t1, sn_flam, sn_ferr)
        ax.plot(sn_wav, t1, lw=2.5, color='dodgerblue')


        print("lnLike for truth [in my template]:", get_lnLike(t, sn_flam, sn_ferr))
        print("lnLike for bestfit model from corner:", get_lnLike(t1, sn_flam, sn_ferr))

        plt.show()

        sys.exit(0)
        """

    fails = np.array(fail_id_list)
    print("\nCatastrophic failures:", np.sort(fails))
    print("Total catastrophic fails:", len(fails))

    # Convert to numpy arrays
    zerr_low = np.array(zerr_low_list)
    zerr_up  = np.array(zerr_up_list)
    z_corner = np.array(z_corner)
    z_acc = np.array(z_acc)
    z_err = np.array(z_err)
    z_truth = np.array(z_truth)

    snr_arr = np.array(snr_arr)
    
    if runtype == 'sn':
        phase_acc = np.array(phase_acc)
        phase_err = np.array(phase_err)
        phase_corner = np.array(phase_corner)
        phase_truth = np.array(phase_truth)

        phase_err = phase_err.reshape(2, len(phase_truth))

    # Reshape errorbars to correct shape
    z_err = z_err.reshape(2, len(z_truth))

    num_plot = len(z_acc)
    print("Number of points on plot:", num_plot)

    nmad = mad_std(z_acc)
    print("NMAD:", "{:.5f}".format(nmad))

    # ------- plot
    fig = plt.figure(figsize=(10,5))

    gs = fig.add_gridspec(nrows=1, ncols=10, left=0.05, right=0.95, wspace=0.1)

    ax1 = fig.add_subplot(gs[:7])
    ax2 = fig.add_subplot(gs[7:])

    ax1.set_xlabel(r'$\mathrm{SNR}$', fontsize=16)
    ax1.set_ylabel(r'$\frac{z_\mathrm{corner} - z_\mathrm{truth}}{1 + z_\mathrm{truth}}$', fontsize=16)

    ax1.errorbar(snr_arr, z_acc, yerr=z_err, fmt='o', \
        markersize=2.0, markerfacecolor='k', markeredgecolor='k', ecolor='k', elinewidth=0.3)
    ax1.axhline(y=0.0, ls='--', color='steelblue', lw=1.5)

    ax1.text(x=0.8, y=0.95, s=r'$\mathrm{N_{total}\,=\,}$' + str(num_plot), color='k', \
        verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, size=14)
    #ax1.text(x=0.78, y=0.1, s=r'$\mathrm{\sigma_{NMAD}\,=\,}$' + '{:.4f}'.format(nmad), color='k', \
    #    verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, size=14)

    # -------- Side histogram
    if runtype == 'galaxy':
        ax1.set_ylim(-0.4, 0.4)
        ax2.set_ylim(-0.4, 0.4)
        hist_rng = (-0.4, 0.4)
    else:
        ax1.set_ylim(-0.015, 0.015)
        ax2.set_ylim(-0.015, 0.015)
        hist_rng = (-0.015, 0.015)

    ax2.hist(z_acc, range=hist_rng, bins=30, 
        orientation='horizontal', histtype='step', color='k', lw=2.0)
    ax2.set_yticks([])

    # -------- Inset fig showing zoomed in area within +-0.01 in z_acc
    if runtype == 'galaxy':
        rect = [0.35, 0.17, 0.3, 0.2]  # [left, bottom, width, height]
        ax_inset = fig.add_axes(rect)

        zoom_idx = np.where(np.abs(z_acc) <= 0.01)[0]
        print("Num points within 1 percent accuracy:", len(zoom_idx))

        ax_inset.errorbar(snr_arr[zoom_idx], z_acc[zoom_idx], yerr=z_err[:, zoom_idx], fmt='o', \
            markersize=2.0, markerfacecolor='k', markeredgecolor='k', ecolor='k', elinewidth=0.3)
        ax_inset.axhline(y=0.0, ls='--', color='steelblue', lw=1.5)

        ax_inset.text(x=0.7, y=0.25, s=r'$\mathrm{N_{\leq1\%}\,=\,}$' + str(len(zoom_idx)), color='k', \
            verticalalignment='top', horizontalalignment='left', transform=ax_inset.transAxes, size=11)

        ax_inset.set_ylim(-0.015, 0.015)

    if runtype == 'galaxy':
        fig.savefig(fit_results_dir + 'snr_vs_z_accuracy_snanaGALsim.pdf', dpi=300, bbox_inches='tight')
    else:
        fig.savefig(fit_results_dir + 'snr_vs_z_accuracy_snanaSNsim.pdf', dpi=300, bbox_inches='tight')

    del fig, ax1, ax2

    # ---------------------- 
    # Make z vs z plot
    fig1 = plt.figure(figsize=(8, 11))

    gs = fig1.add_gridspec(nrows=11, ncols=1, left=0.05, right=0.95, wspace=0.1)

    ax1 = fig1.add_subplot(gs[:8])
    ax2 = fig1.add_subplot(gs[8:])

    ax1.set_ylabel(r'$z_\mathrm{corner}$', fontsize=16)
    ax2.set_ylabel(r'$\frac{z_\mathrm{corner} - z_\mathrm{truth}}{1 + z_\mathrm{truth}}$', fontsize=16)
    ax2.set_xlabel(r'$z_\mathrm{truth}$', fontsize=16)

    ax1.errorbar(z_truth, z_corner, yerr=z_err, fmt='o', \
        markersize=2.0, markerfacecolor='k', markeredgecolor='k', \
        ecolor='k', elinewidth=0.3)
    ax1.plot(np.arange(0.0, 1.31, 0.01), np.arange(0.0, 1.31, 0.01), \
        ls='--', color='steelblue', lw=1.5)

    ax2.errorbar(z_truth, z_acc, yerr=z_err, fmt='o', \
        markersize=2.0, markerfacecolor='k', markeredgecolor='k', \
        ecolor='k', elinewidth=0.3)
    ax2.axhline(y=0.0, ls='--', color='steelblue', lw=1.5)

    # Set limits
    if runtype == 'galaxy':
        ax1.set_xlim(0.0, 1.05)
        ax1.set_ylim(0.0, 1.05)

        ax2.set_xlim(0.0, 1.05)
        ax2.set_ylim(-0.4, 0.4)
    else:
        ax1.set_xlim(0.0, 1.30)
        ax1.set_ylim(0.0, 1.30)

        ax2.set_xlim(0.0, 1.30)
        ax2.set_ylim(-0.015, 0.015)

    # Ticks
    ax1.set_xticklabels([])

    # Add zoomed in inset for galaxies
    if runtype == 'galaxy':
        rect = [0.71, 0.14, 0.2, 0.1]  # [left, bottom, width, height]
        ax_inset = fig1.add_axes(rect)

        ax_inset.errorbar(z_truth[zoom_idx], z_acc[zoom_idx], yerr=z_err[:, zoom_idx], fmt='o', \
            markersize=2.0, markerfacecolor='k', markeredgecolor='k', ecolor='k', elinewidth=0.3)
        ax_inset.axhline(y=0.0, ls='--', color='steelblue', lw=1.5)

        ax_inset.text(x=0.48, y=0.25, s=r'$\mathrm{N_{\leq1\%}\,=\,}$' + str(len(zoom_idx)), color='k', \
            verticalalignment='top', horizontalalignment='left', transform=ax_inset.transAxes, size=11)

        ax_inset.set_xlim(0.1, 0.8)
        ax_inset.set_ylim(-0.015, 0.015)

    if runtype == 'galaxy':
        fig1.savefig(fit_results_dir + 'redshift_acc_comparison_snanaGALsim.pdf', 
            dpi=300, bbox_inches='tight')
    else:
        fig1.savefig(fit_results_dir + 'redshift_acc_comparison_snanaSNsim.pdf', 
            dpi=300, bbox_inches='tight')

    # ---------------------- 
    # Catastrophic failures vs SNR
    fig2 = plt.figure()
    ax = fig2.add_subplot(111)

    fail_idx = np.where(np.abs(z_acc) >= 0.1)[0]

    print("Total catastrophic fails:", len(fail_idx))

    ax.set_xlabel(r'$\mathrm{SNR}$', fontsize=16)
    catfail_str = r'$\mathrm{Catastrophic\ failure\ fraction}\ \left(\frac{\Delta z}{1+z}\,\geq\,0.1\right)$'
    ax.set_ylabel(catfail_str, fontsize=16)

    counts, bins = np.histogram(snr_arr[fail_idx], 32, range=(3, 35))

    bar_cen = np.empty(len(counts))
    for k in range(len(counts)):
        bar_cen[k] = (bins[k] + bins[k+1]) / 2

    ax.bar(x=bar_cen, height=counts/num_plot, width=1.0, color='None', edgecolor='k')

    if runtype == 'galaxy':
        fig2.savefig(fit_results_dir + 'fails_hist_snr_snanaGALsim.pdf', dpi=300, bbox_inches='tight')
    else:
        fig2.savefig(fit_results_dir + 'fails_hist_snr_snanaSNsim.pdf', dpi=300, bbox_inches='tight')

    # ---------------------- 
    # Phase recovery for SNe
    if runtype == 'sn':
        fig3 = plt.figure(figsize=(8, 11))
        gs = fig3.add_gridspec(nrows=11, ncols=1, left=0.05, right=0.95, wspace=0.1)

        ax1 = fig3.add_subplot(gs[:8])
        ax2 = fig3.add_subplot(gs[8:])

        ax1.set_ylabel(r'$\mathrm{Phase_{corner}}$', fontsize=16)
        ax2.set_ylabel(r'$\mathrm{\Delta(Phase)}$', fontsize=16)
        ax2.set_xlabel(r'$\mathrm{Phase_{truth}}$', fontsize=16)

        ax1.errorbar(phase_truth, phase_corner, yerr=phase_err, fmt='o', \
            markersize=2.0, markerfacecolor='k', markeredgecolor='k', \
            ecolor='k', elinewidth=0.3)
        ax1.plot(np.arange(-19, 50, 0.01), np.arange(-19, 50, 0.01), \
            ls='--', color='steelblue', lw=1.5)

        ax2.errorbar(phase_truth, phase_acc, yerr=phase_err, fmt='o', \
            markersize=2.0, markerfacecolor='k', markeredgecolor='k', \
            ecolor='k', elinewidth=0.3)
        ax2.axhline(y=0.0, ls='--', color='steelblue', lw=1.5)

        # Set limits
        ax1.set_xlim(-12, 20)
        ax1.set_ylim(-12, 20)
        ax2.set_xlim(-12, 20)

        # Ticks
        ax1.set_xticklabels([])

        fig3.savefig(fit_results_dir + 'phase_recovery_snr.pdf', dpi=300, bbox_inches='tight')

    plt.clf()
    plt.cla()
    plt.close()

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)
