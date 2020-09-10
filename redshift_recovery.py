import numpy as np

import os
import sys
import glob

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

home = os.getenv("HOME")
ext_spectra_dir = home + "/Documents/roman_slitless_sims_results/"

def main():

    res_dir = ext_spectra_dir + 'fitting_results/'

    # Create empty lists
    input_z_list = [0.527, 0.033, 0.93, 0.583, 0.281, 0.274, 0.999, 0.327, 0.124]
    sn_z_fit_list = [0.527, 0.054, 0.001, 0.579, 0.527, 0.685, 0.685, 0.579, 0.474]

    """
    for fl in glob.glob(res_dir + 'fitres_sn*.npy'):

        res = np.load(fl, allow_pickle=True)
        sn_z_fit_list.append(r.item()['redshift'])

        inp = np.load(fl.replace('fitres', 'input'), allow_pickle=True)
        input_z_list.append(inp.item()['sn_z'])
    """

    # Convert to numpy arrays and plot
    input_z = np.asarray(input_z_list)
    sn_z_fit = np.asarray(sn_z_fit_list)

    # Plot
    # Define figure
    fig = plt.figure(figsize=(5, 6))
    gs = gridspec.GridSpec(12,16)
    gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.25, hspace=1.0)

    # Put axes on grid
    ax1 = fig.add_subplot(gs[:8, :])
    ax2 = fig.add_subplot(gs[8:, :])

    # Set labels
    ax1.set_ylabel(r'$z_\mathrm{fit}$', fontsize=15)

    ax2.set_ylabel(r'$\frac{\Delta z}{1 + z_\mathrm{input}}$', fontsize=15)
    ax2.set_xlabel(r'$z_\mathrm{input}$', fontsize=15)

    # Plot
    ax1.plot(input_z, sn_z_fit, 'o', markersize=5, color='k', markeredgecolor='k')

    xplot = np.arange(0.001, 1.001, 0.001)
    ax1.plot(xplot, xplot, '--', color='r')

    # plot residuals
    delta_z = input_z - sn_z_fit
    ax2.plot(input_z, delta_z/(1+input_z), 'o', markersize=5, color='k', markeredgecolor='k')

    ax2.axhline(y=0.0, ls='--', color='r')

    # Force limits
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.0)

    ax2.set_xlim(0.0, 1.0)

    # Save fig
    fig.savefig(res_dir + 'redshift_recovery.pdf', dpi=200, bbox_inches='tight')

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)