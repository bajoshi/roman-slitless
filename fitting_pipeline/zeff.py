import numpy as np

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

extdir = '/Volumes/Joshi_external_HDD/Roman/'
gal_fit_dir = extdir + 'sn_sit_hackday/testv3/'
results_dir = gal_fit_dir + 'Prism_deep_hostIav3/results/'

def old_main():

    # Empty arrays for plotting
    ztrue_list = []
    zinfer_list = []

    # Loop over all results
    for fl in glob.glob(savedir + '*.h5'):
        
        # Because the fitting program is running
        # Dont accept incomplete sampler.h5 files
        flsize = os.stat(fl).st_size / 1e6  # MB
        if flsize < 30:  # full sampler size is 32.9 MB 
            print("Incomplete sampler:", fl, "of size", flsize, "MB.")
            print("Skipping for now.")
            continue

        print("Working on:", fl)

        dat_file = fl.replace('.h5','.DAT')
        dat_file = dat_file.replace('results/emcee_sampler_', 'Prism_shallow_hostIa_SN0')

        print("Reading data file:", dat_file)

        nspectra, gal_wav, gal_flam, gal_ferr, gal_simflam, truth_dict = \
        read_galaxy_data(dat_file)

        # check data quality
        snr = get_snr(gal_wav, gal_flam)
        print("SNR:", snr)
        print("True values", truth_dict)

        # Read in sampler
        sampler = emcee.backends.HDFBackend(fl)

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

        cq_z = corner.quantile(x=flat_samples[:, 0], q=[0.16, 0.5, 0.84])
        cq_ms = corner.quantile(x=flat_samples[:, 1], q=[0.16, 0.5, 0.84])
        cq_age = corner.quantile(x=flat_samples[:, 2], q=[0.16, 0.5, 0.84])
        cq_tau = corner.quantile(x=flat_samples[:, 3], q=[0.16, 0.5, 0.84])
        cq_av = corner.quantile(x=flat_samples[:, 4], q=[0.16, 0.5, 0.84])

        # append to lists
        ztrue_list.append(truth_dict['z'])
        zinfer_list.append(cq_z[1])

    return None

def main():

    cat = np.genfromtxt(results_dir + 'zrecovery_results_deep.txt', 
                        dtype=None, names=True, encoding='ascii')

    zinfer = cat['z_corner']
    ztrue = cat['z_truth']

    # Empty arrays for plotting
    ztrue_list = []
    zinfer_list = []

    # Make plots 
    # ----------- z efficiency
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel(r'$z$', fontsize=16)
    ax.set_ylabel(r'$z_\mathrm{eff}$', fontsize=16)

    bins = np.arange(0.1, 3.1, 0.25)

    bin_cen = np.zeros(len(bins)-1)
    for i in range(len(bin_cen)):
        bin_cen[i] = (bins[i] + bins[i+1] ) / 2

    print(bins)
    print(bin_cen)

    for i in range(len(cat)):

        current_ztrue = ztrue[i]
        current_zinfer = zinfer[i]

        #print('\n', current_ztrue, current_zinfer)

        bin_idx = np.argmin(abs(bin_cen - current_ztrue))

        #print(bin_idx, bins[bin_idx], bins[bin_idx+1])

        ztrue_list.append(current_ztrue)

        if bins[bin_idx] <= current_zinfer < bins[bin_idx+1]:
            zinfer_list.append(current_zinfer)

    inferred_counts, bin_edges = np.histogram(zinfer_list, bins=bins)
    true_counts, bin_edges = np.histogram(ztrue_list, bins=bins)

    zeff = inferred_counts / true_counts

    print("\nz inferred counts:", inferred_counts)
    print("z true counts:    ", true_counts)

    print("Bin centers:      ", bin_cen)
    print("z efficiency:     ", zeff)

    ax.plot(bin_cen, zeff, 'o--', markersize=3.5, color='k', ls='--')

    fig.savefig(results_dir + 'zefficiency_snanaGALsim.pdf', dpi=200, bbox_inches='tight')

    # Save to ascii file
    with open(results_dir + 'zeff_vs_z.txt', 'w') as fh:
        fh.write('#  z  zeff' + '\n')
        for j in range(len(zeff)):
            fh.write('{:.3f}'.format(bin_cen[j]) + '  ')
            fh.write('{:.3f}'.format(zeff[j]) + '\n')

    # ----------- True vs recovered z distribution
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel(r'$z$', fontsize=16)
    ax.set_ylabel(r'$\# \mathrm{objects}$', fontsize=16)

    ax.hist(zinfer_list, 30, range=(0.0, 3.0), histtype='step', 
        lw=2.0, color='k', label='Inferred distribution')
    ax.hist(ztrue_list, 30, range=(0.0, 3.0), histtype='step', 
        lw=2.0, color='tab:red', label='True distribution')

    ax.legend(fontsize=11, frameon=False)

    fig.savefig(results_dir + 'zdist_recovery_snanaGALsim.pdf', dpi=200, bbox_inches='tight')

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)

