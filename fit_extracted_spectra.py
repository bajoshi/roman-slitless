import numpy as np
from astropy.io import fits
from scipy.integrate import trapz

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

home = os.getenv("HOME")
roman_slitless_dir = home + "/Documents/GitHub/roman-slitless/"
ext_spectra_dir = home + "/Documents/roman_slitless_sims_results/"
template_dir = home + "/Documents/roman_slitless_sims_seds/"
ext_root = "romansim"

import fitting_module as fm

def add_noise(sig_arr, noise_level):
    """
    This function will vary the flux randomly within 
    the noise level specified. It assumes the statistical
    noise is Gaussian.
    The final error array returned is Poissonian noise.
    """
    # Poisson noise: does the signal have to be in 
    # units of photons or electrons for sqrt(N) to 
    # work? like I cannot use sqrt(signal) in physical 
    # units and call it Poisson noise?

    sigma_arr = noise_level * sig_arr

    spec_noise = np.zeros(len(sig_arr))
    err_arr = np.zeros(len(sig_arr))

    for k in range(len(sig_arr)):

        mu = sig_arr[k]
        sigma = sigma_arr[k]

        # Now vary flux using numpy random.normal
        # HAS TO BE POSITIVE!
        spec_noise[k] = np.random.normal(mu, sigma, 1)

        if spec_noise[k] < 0:
            max_iters = 10
            iter_count = 0
            while iter_count < max_iters:
                spec_noise[k] = np.random.normal(mu, sigma, 1)
                iter_count += 1
                if spec_noise[k] > 0:
                    break
            # if it still hasn't given a positive number after max_iters
            # then revert it back to whatever the signal was before randomly varying
            if (iter_count >= max_iters) and (spec_noise[k] < 0):
                spec_noise[k] = sig_arr[k]

        # err_arr[k] = np.sqrt(spec_noise[k])
        err_arr[k] = noise_level * spec_noise[k]

    return spec_noise, err_arr

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

            # ---------------------------- Set up input params dict ---------------------------- #
            input_dict = {}

            print("INPUTS:")
            # ---- SN
            t = template_name.split('.txt')[0].split('_')

            sn_z = float(t[-1].replace('p', '.').replace('z',''))
            sn_day = int(t[-2].replace('day',''))
            print("Supernova input z:", sn_z)
            print("Supernova day:", sn_day, "\n")

            # ---- HOST
            h_idx = int(np.where(sedlst['segid'] == hostid)[0])
            h_path = sedlst['sed_path'][h_idx]
            th = os.path.basename(h_path)

            th = th.split('.txt')[0].split('_')

            host_z = float(th[-1].replace('p', '.').replace('z',''))
            host_ms = float(th[-2].replace('p', '.').replace('ms',''))

            host_age_u = th[-3]
            if host_age_u == 'gyr':
                host_age = float(th[2])
            elif host_age_u == 'myr':
                host_age = float(th[2])/1e3

            print("Host input z:", host_z)
            print("Host input stellar mass [log(Ms/Msol)]:", host_ms)
            print("Host input age [Gyr]:", host_age)

            input_dict['host_z'] = host_z
            input_dict['host_ms'] = host_ms
            input_dict['host_age'] = host_age

            input_dict['sn_z'] = sn_z
            input_dict['sn_day'] = sn_day

            # ---------------------------- FITTING ---------------------------- #
            # ---------- Get spectrum for host and sn
            host_wav = ext_hdu[hostid].data['wavelength']
            host_flam = ext_hdu[hostid].data['flam'] * pylinear_flam_scale_fac
        
            sn_wav = ext_hdu[segid].data['wavelength']
            sn_flam = ext_hdu[segid].data['flam'] * pylinear_flam_scale_fac

            # ---- Fit template to HOST
            noise_level = 0.05  # relative to signal
            # First assign a 33% (3-sigma) error to each point
            host_flam_noisy, host_ferr = add_noise(host_flam, noise_level)
            fit_dict_host = fm.do_fitting(host_wav, host_flam_noisy, host_ferr, object_type='galaxy')

            # ---- Fit template to SN
            # First assign a 33% (3-sigma) error to each point
            sn_flam_noisy, sn_ferr = add_noise(sn_flam, noise_level)
            fit_dict_sn = fm.do_fitting(sn_wav, sn_flam_noisy, sn_ferr, object_type='supernova')

            # ---- Assign recovered params to variables
            # ---- HOST
            fit_wav_host = fit_dict_host['wav']
            fit_flam_host = fit_dict_host['flam']
            fit_z_host = fit_dict_host['redshift']

            fit_alpha_host = fit_dict_host['alpha']

            fit_model_grid_host = fit_dict_host['model_lam']
            fit_fullres_host = fit_dict_host['fullres']

            fit_pz_host = fit_dict_host['pz']
            fit_zsearch = fit_dict_host['zsearch']  # only defined once because the same search grid is used for HOST and SN

            fit_age_host = fit_dict_host['age']
            fit_av_host  = fit_dict_host['av']

            # ---- SN
            fit_wav_sn = fit_dict_sn['wav']
            fit_flam_sn = fit_dict_sn['flam']
            fit_z_sn = fit_dict_sn['redshift']

            fit_alpha_sn = fit_dict_sn['alpha']

            fit_model_grid_sn = fit_dict_sn['model_lam']
            fit_fullres_sn = fit_dict_sn['fullres']

            fit_pz_sn = fit_dict_sn['pz']
            fit_zsearch = fit_dict_sn['zsearch']

            fit_sn_day = fit_dict_sn['day']

            # ----------------- Plotting -------------------- #
            # Set up the figure
            fig = plt.figure(figsize=(12.5,5))

            gs = gridspec.GridSpec(6,9)
            gs.update(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.00, hspace=0.7)

            axh = fig.add_subplot(gs[:, :4])
            axs = fig.add_subplot(gs[:, 5:])

            axh.set_xlabel(r'$\mathrm{Wavelength\ [\AA]}$', fontsize=16)
            axh.set_ylabel(r'$\mathrm{F_{\lambda}\ [erg\, s^{-1}\, cm^{-2}\, \AA^{-1}]}$', fontsize=16)
            axh.xaxis.set_label_coords(1.1, -0.07)

            # Plot the extracted spectrum
            axh.plot(host_wav, host_flam_noisy, 'o-', markersize=1.0, color='tab:blue', zorder=2, \
                label=r'$\mathrm{Simulated\ host\ galaxy\ data}$')
            axh.fill_between(host_wav, host_flam_noisy - host_ferr, host_flam_noisy + host_ferr, color='gray', alpha=0.5)
            axs.plot(sn_wav, sn_flam_noisy, 'o-', markersize=1.0, color='tab:blue', zorder=2, \
                label=r'$\mathrm{Simulated\ SN\ data}$')
            axs.fill_between(sn_wav, sn_flam_noisy - sn_ferr, sn_flam_noisy + sn_ferr, color='gray', alpha=0.5)

            # Plot template spectra
            #axh.plot(fit_model_grid_host, fit_fullres_host, 'o-', markersize=1.0, color='tab:gray', alpha=0.7, zorder=1)
            axh.plot(fit_wav_host, fit_alpha_host * fit_flam_host, 'o-', markersize=1.0, color='tab:red', \
                zorder=3, label=r'$\mathrm{Template\ fit}$')
            #axs.plot(fit_model_grid_sn, fit_fullres_sn, 'o-', markersize=1.0, color='tab:gray', alpha=0.7, zorder=1)
            axs.plot(fit_wav_sn, fit_alpha_sn * fit_flam_sn, 'o-', markersize=1.0, color='tab:red', \
                zorder=3, label=r'$\mathrm{Template\ fit}$')

            # ---------- Add info of input and recovered parameters
            # Table bounding box for both HOST and SN tables
            table_bbox = np.array([0.77, 0.25, 0.2, 0.3])  # A numpy array of the form [left, bottom, width, height]... I think

            # ---- HOST
            rows_host = [r'$z_\mathrm{peak}$', r'$z_\mathrm{wt}$', \
            r'$\mathrm{Stellar\ mass\, [log(M/M_\odot)]}$', 'Age [Gyr]', r'$A_V$']
            columns_host = ['Input', 'Recovered']

            z_wt_host = trapz(y=fit_zsearch * fit_pz_host, x=fit_zsearch)
            ms = np.log10(fit_alpha_host)

            cell_text_host = [['{:.3f}'.format(host_z),   '{:.3f}'.format(fit_z_host)], 
                         ['{:.3f}'.format(host_z),   '{:.3f}'.format(z_wt_host)],
                         ['{:.3f}'.format(host_ms),  '{:.2f}'.format(ms)],
                         ['{:.3f}'.format(host_age), '{:.2f}'.format(fit_age_host)],
                         ['0',     '{:.2f}'.format(fit_av_host)]]

            table_host = axh.table(cellText=cell_text_host, rowLabels=rows_host, rowLoc='center', colWidths=[0.1, 0.1],
                                   cellLoc='center', colLabels=columns_host, bbox=table_bbox, 
                                   fontsize=12)

            # ---- SN
            rows_sn = [r'$z_\mathrm{peak}$', r'$z_\mathrm{wt}$', \
            'SN Day [rel to peak]', r'$A_V$']
            columns_sn = ['Input', 'Recovered']

            z_wt_sn = trapz(y=fit_zsearch * fit_pz_sn, x=fit_zsearch)
            av = 0.0

            cell_text_sn = [['{:.3f}'.format(sn_z),   '{:.3f}'.format(fit_z_sn)], 
                         ['{:.3f}'.format(sn_z),   '{:.3f}'.format(z_wt_sn)],
                         ['{:d}'.format(sn_day),   '{:d}'.format(fit_sn_day)],
                         ['0',     '{:.2f}'.format(av)]]

            table_sn = axs.table(cellText=cell_text_sn, rowLabels=rows_sn, rowLoc='center', colWidths=[0.1, 0.1],
                                 cellLoc='center', colLabels=columns_sn, bbox=table_bbox, 
                                 fontsize=12)
 
            axh.text(0.01, 0.2, s=r'$\mathrm{SegID}: $' + str(hostid), \
                verticalalignment='top', horizontalalignment='left', transform=axh.transAxes, color='k', size=12)
            axs.text(0.01, 0.2, s=r'$\mathrm{SegID}: $' + str(segid), \
                verticalalignment='top', horizontalalignment='left', transform=axs.transAxes, color='k', size=12)

            # ------------ Add p(z) as an inset plot
            # This has to come after the table apparently 
            # because otherwise it puts the table inside the inset axes.
            # For the rectangles below:
            # [left, bottom, width, height] and also width and height in 
            # relative figure units (i.e., between 0 to 1 as a fraction of
            # the dimension -- height or width -- that you are referring to).
            rect_h = [0.28, 0.67, 0.15, 0.2]
            rect_s = [0.78, 0.67, 0.15, 0.2]
            axh_in = fig.add_axes(rect_h)
            axs_in = fig.add_axes(rect_s)
            axh_in.plot(fit_zsearch, fit_pz_host)
            axs_in.plot(fit_zsearch, fit_pz_sn)

            axh_in.set_xlabel('z', fontsize=12)
            axh_in.set_ylabel('p(z)', fontsize=12)
            axs_in.set_xlabel('z', fontsize=12)
            axs_in.set_ylabel('p(z)', fontsize=12)

            # Wavelength limits
            axh.set_xlim(9000, 20000)
            axs.set_xlim(9000, 20000)

            axh.legend(loc=3)
            axs.legend(loc=3)

            plt.show()
            plt.clf()
            plt.cla()
            plt.close()

            # ------------ Save figure
            fig.savefig(roman_slitless_dir + 'fitres_sn' + str(segid) + '.pdf', dpi=200, bbox_inches='tight')

            # ------------ Save fit results
            fhost = ext_spectra_dir + 'fitting_results/fitres_host_' + str(hostid) + '.npy'
            fsn = ext_spectra_dir + 'fitting_results/fitres_sn_' + str(segid) + '.npy'
            np.save(fhost, fit_dict_host)
            np.save(fsn, fit_dict_sn)

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)

