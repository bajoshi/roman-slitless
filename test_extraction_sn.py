import numpy as np

import os
import sys

import matplotlib.pyplot as plt

home = os.getenv("HOME")
roman_slitless_dir = home + "/Documents/GitHub/roman-slitless/"

sys.path.append(home + '/Documents/GitHub/pyLINEAR-fork/')

import pylinear_fork.modules.extraction.sedfile as sedfile

def main():

    # Read in the template spectra
    bc03_6gyr = np.genfromtxt('/Users/bhavinjoshi/Documents/roman_slitless_sims_seds/bc03_template_6_gyr.txt', \
        dtype=None, names=True, encoding='ascii')
    bc03_100myr = np.genfromtxt('/Users/bhavinjoshi/Documents/roman_slitless_sims_seds/bc03_template_100_myr.txt', \
        dtype=None, names=True, encoding='ascii')

    # Define redshifts and do redshifting
    # constant for now # deal with flux later
    redshift = 0.5

    bc03_6gyr['wav'] *= (1 + redshift)
    bc03_100myr['wav'] *= (1 + redshift)

    # Find valid wavelengths to show
    wav_idx = np.where((bc03_6gyr['wav'] >= 10000) & (bc03_6gyr['wav'] <= 20000))[0]

    # Read in the extracted spectra
    with sedfile.SEDFile(home + '/Documents/GitHub/pyLINEAR-fork/pylinear_fork/romansim.h5') as sd:

        for segid in sd.segIDs:
            
            if (segid >= 491) and (segid <= 495):

                print("Segmentation ID:", segid)

                data = sd.spectrum(segid)
                
                fig = plt.figure()
                ax = fig.add_subplot(111)

                ax.set_xlabel(r'$\mathrm{Wavelength\ [\AA]}$', fontsize=16)
                ax.set_ylabel(r'$\mathrm{Flux\ [arbitrary\ scale]}$', fontsize=16)

                # Scale up BC03 spectra according to broadband photometry
                scale_fac_galaxy = np.nanmax(data['flam']) / np.nanmax(bc03_6gyr['llam'][wav_idx])
                bc03_6gyr['llam'] *= scale_fac_galaxy

                scale_fac_sn = np.nanmax(data['flam']) / np.nanmax(bc03_100myr['llam'][wav_idx])
                bc03_100myr['llam'] *= scale_fac_sn

                # PLot extracted spectra
                ax.plot(data['lam'], data['flam'], 'o-', markersize=1.0, color='tab:blue', label='{}'.format(segid))

                # Plot template spectra
                ax.plot(bc03_6gyr['wav'][wav_idx], bc03_6gyr['llam'][wav_idx], 'o-', markersize=1.0, color='tab:red')
                ax.plot(bc03_100myr['wav'][wav_idx], bc03_100myr['llam'][wav_idx], 'o-', markersize=1.0, color='tab:olive')
                
                ax.legend(loc=0)
                plt.show()

                plt.clf()
                plt.cla()
                plt.close()

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)