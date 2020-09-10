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
    sn_z_fit_list = []
    input_z_list = []

    for fl in glob.glob(res_dir + 'fitres_sn*.npy'):

        res = np.load(fl, allow_pickle=True)
        sn_z_fit_list.append(r.item()['redshift'])

        inp = np.load(fl.replace('fitres', 'input'), allow_pickle=True)
        input_z_list.append(inp.item()['sn_z'])

    # Plot
    

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)