import numpy as np
from astropy.io import fits

import os
import sys
import socket

import matplotlib.pyplot as plt

home = os.getenv('HOME')
ext_spectra_dir = home + "/Documents/roman_slitless_sims_results/"
roman_slitless_dir = home + "/Documents/GitHub/roman-slitless/"

def main():

    img_suffix = 'Y106_11_1'

    # ---------- Read in extracted 1D spectra
    ext_spec_filename = ext_spectra_dir + 'romansim_' + img_suffix + '_x1d.fits'
    ext_hdu = fits.open(ext_spec_filename)

    # Read in sed.lst
    sedlst_header = ['segid', 'sed_path']
    sedlst_path = roman_slitless_dir + 'pylinear_lst_files/' + 'sed_' + img_suffix + '_plffsn2' + '.lst'
    sedlst = np.genfromtxt(sedlst_path, dtype=None, names=sedlst_header, encoding='ascii')

    pylinear_flam_scale_fac = 1e-17

    hostid = 207
    segid = 241

    # Get input spectra for host and SN
    h_idx = int(np.where(sedlst['segid'] == hostid)[0])
    h_path = sedlst['sed_path'][h_idx]

    s_idx = int(np.where(sedlst['segid'] == segid)[0])
    s_path = sedlst['sed_path'][s_idx]

    # ---------- Get spectrum for host and sn
    host_wav = ext_hdu[('SOURCE', hostid)].data['wavelength']
    host_flam = ext_hdu[('SOURCE', hostid)].data['flam'] * pylinear_flam_scale_fac
        
    sn_wav = ext_hdu[('SOURCE', segid)].data['wavelength']
    sn_flam = ext_hdu[('SOURCE', segid)].data['flam'] * pylinear_flam_scale_fac

    # ---------- plot
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot()

    ax.set_xlabel(r'$\mathrm{\lambda\ [\AA]}$', fontsize=16)
    ax.set_ylabel(r'$\mathrm{f_\lambda\ [cgs]}$', fontsize=16)

    # Scale if needed
    # for 207
    host_flam /= 315.6

    # plot extracted spectrum
    ax.plot(host_wav, host_flam, color='tab:blue', lw=2, label='pyLINEAR extraction host galaxy', zorder=1)
    #ax.plot(sn_wav, sn_flam, color='salmon', lw=2, label='pyLINEAR extraction SN', zorder=1)

    #ax.fill_between(host_wav, host_flam - host_ferr, host_flam + host_ferr, \
    #    color='grey', alpha=0.5, zorder=1)

    #m = model_host(host_wav, host_z, host_ms, host_age, np.log10(host_tau), host_av)

    ## Only consider wavelengths where sensitivity is above 20%
    #host_x0 = np.where( (host_wav >= grism_sens_wav[grism_wav_idx][0]  ) &
    #                    (host_wav <= grism_sens_wav[grism_wav_idx][-1] ) )[0]
    #m = m[host_x0]

    #a = np.nansum(host_flam[host_x0] * m / host_ferr[host_x0]**2) / np.nansum(m**2 / host_ferr[host_x0]**2)
    #print("HOST a:", "{:.4e}".format(a))
    #m = a*m
    #chi2_good = np.nansum( (m - host_flam[host_x0])**2 / host_ferr[host_x0]**2 )# / len(m)
    #print("HOST base model chi2:", chi2_good)

    #ax.plot(host_wav[host_x0], m, lw=1.0, color='tab:red', zorder=2, label='Downgraded model from mcmc code')

    # plot actual template passed into pylinear
    if 'plffsn2' not in socket.gethostname():
        h_path = h_path.replace('/home/bajoshi/', '/Users/baj/')
        s_path = s_path.replace('/home/bajoshi/', '/Users/baj/')

    host_template = np.genfromtxt(h_path, dtype=None, names=True, encoding='ascii')
    ax.plot(host_template['lam'], host_template['flux'], lw=1.5, \
        color='lightsteelblue', zorder=2, label='model given to pyLINEAR for host galaxy')

    sn_template = np.genfromtxt(s_path, dtype=None, names=True, encoding='ascii')
    #ax.plot(sn_template['lam'], sn_template['flux'], lw=1.5, \
    #    color='tab:red', zorder=2, label='model given to pyLINEAR for SN')

    ax.set_xlim(9000, 20000)
    host_fig_ymin = np.min(host_flam)
    host_fig_ymax = np.max(host_flam)
    #ax.set_ylim(host_fig_ymin * 0.4, host_fig_ymax * 1.2)

    ax.legend(loc=0, fontsize=12, frameon=False)

    plt.show()

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)