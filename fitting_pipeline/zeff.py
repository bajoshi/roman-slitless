import numpy as np

import emcee
import corner

import os
import sys
import glob

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

home = os.getenv('HOME')
# data dir
datadir = home + '/Desktop/Prism_shallow_hostIa/'
savedir = datadir + 'results/'

from fit_galaxy import read_galaxy_data, get_snr

def main():

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

    # Make plots 
    fig = plt.figure()
    ax = fig.add_subplot(111)

    bins = np.arange(0.0, 1.1, 0.1)

    inferred_counts, bin_edges = np.histogram(zinfer_list, bins=bins)
    true_counts, bin_edges = np.histogram(ztrue_list, bins=bins)

    zeff = inferred_counts / true_counts

    print(inferred_counts)
    print(true_counts)
    print(zeff)

    bin_cen = np.zeros(len(zeff))
    for i in range(len(bin_cen)):
        bin_cen[i] = (bin_edges[i] + bin_edges[i+1] ) / 2

    ax.scatter(bin_cen, zeff, color='k')

    plt.show()

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)