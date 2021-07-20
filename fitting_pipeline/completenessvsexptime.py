import numpy as np

import matplotlib.pyplot as plt

import os
import sys

def main():

    # Read in results file
    ext_spectra_dir = "/Volumes/Joshi_external_HDD/Roman/roman_slitless_sims_results/"
    results_dir = ext_spectra_dir + 'fitting_results/'
    
    resfile = results_dir + 'zrecovery_pylinear_sims_pt0.txt'
    cat = np.genfromtxt(resfile, dtype=None, names=True, encoding='ascii')

    # PLotting
    fig = plt.figure()
    ax = fig.add_subplot(111)

    

    plt.show()

	return None

if __name__ == '__main__':
	main()
	sys.exit(0)