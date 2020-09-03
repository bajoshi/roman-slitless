import numpy as np
from astropy.io import fits

import os
import sys

import matplotlib.pyplot as plt

home = os.getenv("HOME")
roman_slitless_dir = home + "/Documents/GitHub/roman-slitless/"
ext_spectra_dir = home + "/Documents/roman_slitless_sims_results/"
template_dir = home + "/Documents/roman_slitless_sims_seds/"
ext_root = "romansim"

import fitting_module as fm

def main():

    # Read in sed.lst
    sedlst_header = ['segid', 'sed_path']
    sedlst = np.genfromtxt(roman_slitless_dir + 'sed.lst', dtype=None, names=sedlst_header, encoding='ascii')

    # Read in the extracted spectra
    ext_spec_filename = ext_spectra_dir + ext_root + '_ext_x1d.fits'
    ext_hdu = fits.open(ext_spec_filename)

    # Set pylinear f_lambda scaling factor
    pylinear_flam_scale_fac = 1e-17

    # This will come from detection on the direct image
    # For now this comes from the sedlst generation code
    host_segids = np.array([475, 755, 548, 207])
    sn_segids = np.array([481, 753, 547, 241])

    for i in range(len(sedlst)):

        # Get info
        segid = sedlst['segid'][i]

        # Read in the dummy template passed to pyLINEAR
        template_name = os.path.basename(sedlst['sed_path'][i])

        if 'salt' not in template_name:
            continue
        else:
            print("Segmentation ID:", segid, "is a SN. Will begin fitting.")

            # Get corresponding host ID
            hostid = int(host_segids[np.where(sn_segids == segid)[0]])
            print("I have the following host and SN IDs:", segid, hostid)

            # Read in template
            template = np.genfromtxt(template_dir + template_name, dtype=None, names=True, encoding='ascii')

            # Get spectrum for host and sn
            host_wav = ext_hdu[hostid].data['wavelength']
            host_flam = ext_hdu[hostid].data['flam'] * pylinear_flam_scale_fac
        
            sn_wav = ext_hdu[segid].data['wavelength']
            sn_flam = ext_hdu[segid].data['flam'] * pylinear_flam_scale_fac

            # ---- Fit template to HOST
            # First assign a 33% (3-sigma) error to each point
            host_ferr = 0.33 * host_flam
            fit_dict_host = fm.do_fitting(host_wav, host_flam, host_ferr, object_type='galaxy')

            # ---- Fit template to SN
            # First assign a 33% (3-sigma) error to each point
            sn_ferr = 0.33 * sn_flam
            fit_dict_sn = fm.do_fitting(sn_wav, sn_flam, sn_ferr, object_type='supernova')

            fit_wav = fit_dict['wav']
            fit_flam = fit_dict['flam']
            fit_z = fit_dict['redshift']

            sys.exit(0)

            # ----------------- Plotting -------------------- #
            # Set up the figure
            fig = plt.figure()
            ax = fig.add_subplot(111)

            ax.set_xlabel(r'$\mathrm{Wavelength\ [\AA]}$', fontsize=16)
            ax.set_ylabel(r'$\mathrm{F_{\lambda}\ [erg\, s^{-1}\, cm^{-2}\, \AA^{-1}\, \times 10^{-17}]}$', fontsize=16)

            # Plot the extracted spectrum
            ax.plot(wav, flam, 'o-', markersize=1.0, color='tab:blue', label='{}'.format(segid))

            # Plot template spectra
            ax.plot(bestfit_wav, bestfit_flam, 'o-', markersize=1.0, color='tab:red')

            # Wavelength limits
            ax.set_xlim(9000, 20000)

            ax.legend(loc=2)
            plt.show()

            plt.clf()
            plt.cla()
            plt.close()

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)

