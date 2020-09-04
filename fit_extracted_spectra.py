import numpy as np
from astropy.io import fits

import os
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
            # sn_ferr = 0.33 * sn_flam
            # fit_dict_sn = fm.do_fitting(sn_wav, sn_flam, sn_ferr, object_type='supernova')

            fit_wav_host = fit_dict_host['wav']
            fit_flam_host = fit_dict_host['flam']
            fit_z_host = fit_dict_host['redshift']

            fit_alpha_host = fit_dict_host['alpha']

            fit_model_grid_host = fit_dict_host['model_lam']
            fit_fullres_host = fit_dict_host['fullres']

            fit_pz_host = fit_dict_host['pz']
            fit_zsearch_host = fit_dict_host['zsearch']

            # ----------------- Plotting -------------------- #
            # Set up the figure
            fig = plt.figure()
            ax = fig.add_subplot(111)

            ax.set_xlabel(r'$\mathrm{Wavelength\ [\AA]}$', fontsize=16)
            ax.set_ylabel(r'$\mathrm{F_{\lambda}\ [erg\, s^{-1}\, cm^{-2}\, \AA^{-1}\, \times 10^{-17}]}$', fontsize=16)

            # Plot the extracted spectrum
            ax.plot(host_wav, host_flam, 'o-', markersize=1.0, color='tab:blue', label='{}'.format(segid), zorder=2)

            # Plot template spectra
            #ax.plot(fit_model_grid_host, fit_fullres_host, 'o-', markersize=1.0, color='tab:gray', alpha=0.7, zorder=1)
            ax.plot(fit_wav_host, fit_alpha_host * fit_flam_host, 'o-', markersize=1.0, color='tab:red', zorder=3)

            # Add info of test and recovered parameters

            # Add p(z) as an inset plot
            ax_in = inset_axes(ax, width=1.5, height=1.0)  # width and height in inches
            ax_in.plot(fit_zsearch_host, fit_pz_host)

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

