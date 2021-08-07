import numpy as np

import matplotlib.pyplot as plt

import os
import sys

if __name__ == '__main__':

    

    cat = np.genfromtxt(catfile, dtype=None, names=True, encoding='ascii')

    # ---------------------------- SNR vs mag plot
    # Manual entries from running HST/WFC3 spectroscopic ETC
    # For G102 and G141
    etc_mags = np.arange(18.0, 25.5, 0.5)
    etc_g102_snr = np.array([558.0, 414.0, 300.1, 211.89, 145.79, 
                             98.03, 64.68, 42.07, 27.09, 17.32, 
                             11.02, 6.99, 4.43, 2.80, 1.77])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_ylabel('SNR of extracted 1d spec', fontsize=14)
    ax.set_xlabel('F106 mag', fontsize=14)

    ax.scatter(cat['Y106mag'], cat['SNR3600'], s=8, color='k', label='pyLINEAR sim result')
    ax.scatter(etc_mags, etc_g102_snr, s=8, color='royalblue', label='WFC3 G102 ETC prediction')

    ax.legend(loc=0, fontsize=14)

    ax.set_yscale('log')

    fig.savefig(results_dir + 'pylinear_sim_snr_vs_mag.pdf', 
        dpi=200, bbox_inches='tight')
    fig.clear()
    plt.close(fig)