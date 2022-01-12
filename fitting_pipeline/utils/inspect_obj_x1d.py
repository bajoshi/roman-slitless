from astropy.io import fits
import numpy as np
from scipy.interpolate import griddata

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import os
import sys

from get_template_inputs import get_template_inputs

home = os.getenv('HOME')
roman_slitless_dir = home + '/Documents/GitHub/roman-slitless/'

sys.path.append(roman_slitless_dir)
from gen_sed_lst import get_sn_spec_path, get_gal_spec_path  # noqa


def filter_conv(filter_wav, filter_thru, spec_wav, spec_flam):

    # First grid the spectrum wavelengths to the filter wavelengths
    spec_on_filt_grid = griddata(points=spec_wav, 
                                 values=spec_flam, xi=filter_wav)

    # Remove NaNs
    valid_idx = np.where(~np.isnan(spec_on_filt_grid))

    filter_wav = filter_wav[valid_idx]
    filter_thru = filter_thru[valid_idx]
    spec_on_filt_grid = spec_on_filt_grid[valid_idx]

    # Now do the two integrals
    num = np.trapz(y=spec_on_filt_grid * filter_thru, x=filter_wav)
    den = np.trapz(y=filter_thru, x=filter_wav)

    filter_flux = num / den

    return filter_flux


def read_sed_scale(sed_path, obj_mag):

    # First fix the path
    sed_path = sed_path.replace('/astro/ffsn/Joshi/', extdir)

    # Now check if it exists and if not then create it
    if not os.path.isfile(sed_path):
        template_inputs = get_template_inputs(sed_path, verbose=True)
        if 'salt' in sed_path:
            sn_z = template_inputs[0]
            template_day = template_inputs[1]
            template_av = template_inputs[2]
            get_sn_spec_path(sn_z, day_chosen=template_day, 
                             chosen_av=template_av)
        else:
            galaxy_z = template_inputs[0]
            template_logms = template_inputs[1]
            template_age = template_inputs[2]
            template_tau = 10**(template_inputs[3])
            template_av = template_inputs[4]
            get_gal_spec_path(galaxy_z, 
                              log_stellar_mass_chosen=template_logms,
                              chosen_age=template_age, 
                              chosen_tau=template_tau,
                              chosen_av=template_av)

    # Read in SED
    sed = np.genfromtxt(sed_path, dtype=None,
                        names=['wav', 'flam'], encoding='ascii')

    obj_wav = sed['wav']
    obj_flam = sed['flam']

    # Now convolve with the F106 filter and
    # ensure that the flam is scaled to give
    # the expected magnitude
    # First crop the arrays
    wav_idx = np.where((obj_wav > 7600) & (obj_wav < 18200))[0]
    obj_wav = obj_wav[wav_idx]
    obj_flam = obj_flam[wav_idx]

    # Read in F106 filter
    f106_filt_path = 'throughputs/F105W_IR_throughput.csv'
    filt = np.genfromtxt(f106_filt_path,
                         delimiter=',', dtype=None, names=True, 
                         encoding='ascii', usecols=(1, 2))

    sed_filt_flam = filter_conv(filt['Wave_Angstroms'], filt['Throughput'],
                                obj_wav, obj_flam)

    # This must be converted to fnu to work with AB mag
    # i.e., multiply by lam^2/c where lam is the pivot wav
    # for the F106 filter
    sed_filt_fnu = (10552**2 / 3e18) * sed_filt_flam
    implied_mag = -2.5 * np.log10(sed_filt_fnu) - 48.6

    required_fnu = np.power(10, ((obj_mag + 48.6)/-2.5))

    # Compute a scaling factor to scale the SED
    scalefac = required_fnu / sed_filt_fnu

    print('Flam thru filter:', sed_filt_flam)
    print('Fnu thru filter:', sed_filt_fnu)
    print('Required Fnu:', required_fnu)

    # Now update the SED related quantities
    sed_filt_fnu *= scalefac
    implied_mag = -2.5 * np.log10(sed_filt_fnu) - 48.6

    print('Updated SED Fnu:', sed_filt_fnu)
    print('Implied and expected object mags:', 
          implied_mag, obj_mag)

    assert np.allclose(implied_mag, obj_mag)

    # We need to return the scaled f-lambda
    # The scaling factor is the same in nu space
    # and in lambda space. You do not need to do
    # any other conversion here.
    obj_flam *= scalefac

    return obj_wav, obj_flam


def display_segmap_and_spec(ext_hdu, insert_cat, all_inserted_segids, 
                            sn_segid, imdat, segmap, sedlst):

    # ---------------
    # Get object info from inserted objects file
    sn_idx = int(np.where(all_inserted_segids == sn_segid)[0])
    matched_segid = int(insert_cat[sn_idx][-1])

    assert matched_segid == sn_segid

    snmag = float(insert_cat[sn_idx][2])
    hostmag = float(insert_cat[sn_idx][3])
    host_segid = int(float(insert_cat[sn_idx][4]))
    obj_type = insert_cat[sn_idx][-2]

    xsn = float(insert_cat[sn_idx][0])
    ysn = float(insert_cat[sn_idx][1])

    print('\nMatched/SN Seg ID:   ', sn_segid,
          '\nSN magnitude:        ', '{:.3f}'.format(snmag),
          '\nHost galaxy mag:     ', '{:.3f}'.format(hostmag),
          '\nHost galaxy Seg ID:  ', host_segid,
          '\nInserted object type:', obj_type,
          '\nRedshift:            ', 
          '\n---------------------\n')

    # ---------------
    # Get extracted 1d spec 
    sn_wav = ext_hdu[('SOURCE', sn_segid)].data['wavelength']
    sn_flam = ext_hdu[('SOURCE', sn_segid)].data['flam'] * 1e-17
    sn_ferr_lo = ext_hdu[('SOURCE', sn_segid)].data['flounc'] * 1e-17
    sn_ferr_hi = ext_hdu[('SOURCE', sn_segid)].data['fhiunc'] * 1e-17

    host_wav = ext_hdu[('SOURCE', host_segid)].data['wavelength']
    host_flam = ext_hdu[('SOURCE', host_segid)].data['flam'] * 1e-17
    host_ferr_lo = ext_hdu[('SOURCE', host_segid)].data['flounc'] * 1e-17
    host_ferr_hi = ext_hdu[('SOURCE', host_segid)].data['fhiunc'] * 1e-17

    # ---------------
    # Now get dir img and segmap cutouts to show
    host_rows, host_cols = np.where(segmap == host_segid)

    # Get a bounding box for the host-galaxy
    # and add some padding
    padding = 10  # in pixels
    top = np.max(host_rows) + padding
    bottom = np.min(host_rows) - padding

    right = np.max(host_cols) + padding
    left = np.min(host_cols) - padding

    # Construct cutouts
    dirimg_cutout = imdat[bottom:top, left:right]
    segmap_cutout = segmap[bottom:top, left:right]

    # Image extent
    ext = [left, right, bottom, top]

    # Ensure square extent
    x_extent = right - left
    y_extent = top - bottom

    if x_extent > y_extent:
        ext_diff = x_extent - y_extent
        ext = [left, right, 
               bottom-int(ext_diff/2), 
               top+int(ext_diff/2)]
    elif y_extent > x_extent:
        ext_diff = y_extent - x_extent
        ext = [left-int(ext_diff/2), 
               right+int(ext_diff/2), 
               bottom, top]

    # ---------------
    # Read in SED templates for SN and host
    sn_sed_idx = int(np.where(sedlst['segid'] == sn_segid)[0])
    host_sed_idx = int(np.where(sedlst['segid'] == host_segid)[0])

    sn_sed_path = sedlst['sed_path'][sn_sed_idx]
    host_sed_path = sedlst['sed_path'][host_sed_idx]

    print('SN template name:', os.path.basename(sn_sed_path))
    print('HOST template name:', os.path.basename(host_sed_path))
    print('\n')

    sn_template_wav, sn_template_flam = read_sed_scale(sn_sed_path, snmag)
    host_template_wav, host_template_flam = \
        read_sed_scale(host_sed_path, hostmag)

    # ---------------
    # figure
    fig = plt.figure(figsize=(13, 6.5))

    # Gridspec
    gs = GridSpec(18, 18, wspace=10, hspace=10)

    # Add axes
    ax1 = fig.add_subplot(gs[:9, :6])
    ax2 = fig.add_subplot(gs[9:, :6])
    ax3 = fig.add_subplot(gs[:9, 6:])
    ax4 = fig.add_subplot(gs[9:, 6:])

    # ---- Direct image centered on host galaxy
    # Add an X to SN location
    ax1.set_ylabel('Y [pixels]', fontsize=15)
    ax1.imshow(np.log10(dirimg_cutout), extent=ext, 
               origin='lower', cmap='Greys')
    ax1.scatter(xsn, ysn, marker='x', lw=5.0, 
                s=60, color='crimson')

    # ---- Segmentation map for both objects
    ax2.set_xlabel('X [pixels]', fontsize=15)
    ax2.set_ylabel('Y [pixels]', fontsize=15)

    ax2.imshow(segmap_cutout, extent=ext, origin='lower')

    # ---- SN extracted spectrum
    ax3.set_ylabel('F-lambda [cgs]', fontsize=15)
    
    ax3.plot(sn_wav, sn_flam, color='k', lw=1.5, label='pyLINEAR x1d spec SN')
    ax3.fill_between(sn_wav, sn_flam - sn_ferr_lo, sn_flam + sn_ferr_hi, 
                     color='gray', alpha=0.5)
    
    ax3.plot(sn_template_wav, sn_template_flam, 
             color='dodgerblue', lw=2.0, label='SN template')

    # ---- Host galaxy extracted spectrum
    ax4.set_ylabel('F-lambda [cgs]', fontsize=15)
    ax4.set_xlabel('Wavelength [Angstroms]', fontsize=15)

    ax4.plot(host_wav, host_flam, color='k', lw=1.5, 
             label='pyLINEAR x1d spec host-galaxy', zorder=3)
    ax4.fill_between(host_wav, host_flam - host_ferr_lo, 
                     host_flam + host_ferr_hi,
                     color='gray', alpha=0.5, zorder=3)

    ax4.plot(host_template_wav, host_template_flam, 
             color='dodgerblue', lw=1.2, label='HOST-galaxy template',
             alpha=0.9, zorder=2)

    # ---- Legend
    ax3.legend(loc=0, frameon=False, fontsize=14)
    ax4.legend(loc=0, frameon=False, fontsize=14)

    # ---- Limits
    # Decide Y limits based on flux limits
    sn_min_flux = np.nanmin(sn_flam)
    sn_max_flux = np.nanmax(sn_flam)

    ax3.set_ylim(sn_min_flux * 0.8, sn_max_flux * 1.2)
    
    host_min_flux = np.nanmin(host_flam)
    host_max_flux = np.nanmax(host_flam)

    ax4.set_ylim(host_min_flux * 0.8, host_max_flux * 1.2)

    # Force X limits to be the same for both axes
    ax3.set_xlim(7600.0, 18200.0)
    ax4.set_xlim(7600.0, 18200.0)

    # ---- Suppress some axes tick labels
    ax1.set_xticklabels([])
    ax3.set_xticklabels([])

    # ---- Axes title
    ax1.set_title('F106 image', fontsize=15)
    ax3.set_title('1-hour prism spectrum', fontsize=15)

    # ---- Save figure
    # plt.show()
    figdir = roman_slitless_dir + 'figures/sn_spectra_inspect/'
    figname = figdir + 'sn_' + str(sn_segid) + '_Y106_0_' + \
        str(detector) + '.pdf'
    fig.savefig(figname, dpi=300, bbox_inches='tight')

    return None


if __name__ == '__main__':

    # ---------------
    # Info for code to plot
    sn_segid = int(sys.argv[1])
    detector = '1'
    exptime = '1200s'

    # THIS CODE IS INTENDED TO ONLY BE RUN AFTER 
    # PLFFSN2 FINISHES AND NOT ON PLFFSN2.
    # You will also need to rsync the following from PLFFSN2:
    # 1. *SNadded.fits, *SNadded_segmap.fits, and *SNadded.npy
    # 2. SED LST files
    # 3. x1d results
    # 4. SED txt files
    # While the salt2*.txt SED files can all be transferred 
    # simultaneously, the bc03*.txt templates must be done
    # for each galaxy separately because there are WAY too 
    # many of them (and we're only going to need a handful
    # to check these plots).
    ######################
    
    # ---------------
    # Read in npy file with all inserted object info
    extdir = '/Volumes/Joshi_external_HDD/Roman/'
    img_sim_dir = extdir + 'roman_direct_sims/sims2021/K_5degimages_part1/'
    insert_npy_path = img_sim_dir + '5deg_Y106_0_' + detector + '_SNadded.npy'

    insert_cat = np.load(insert_npy_path)

    # Grab all inserted segids
    all_inserted_segids = insert_cat[:, -1].astype(np.int64)

    # ---------------
    # Read in extracted spectra
    resdir = extdir + 'roman_slitless_sims_results/'
    x1d = fits.open(resdir + 'romansim_prism_Y106_0_' +
                    detector + '_' + exptime + '_x1d.fits')

    # ---------------
    # Read in image and segmap
    img_path = img_sim_dir + '5deg_Y106_0_' + detector + '_SNadded.fits'
    segmap_path = img_sim_dir + '5deg_Y106_0_' + \
        detector + '_SNadded_segmap.fits'

    imdat = fits.getdata(img_path)
    segmap = fits.getdata(segmap_path)

    # ---------------
    # Read in sedlst
    sedlst_path = extdir + 'pylinear_lst_files/sed_Y106_0_' + detector + '.lst'
    sedlst = np.genfromtxt(sedlst_path, dtype=None, 
                           names=['segid', 'sed_path'], encoding='ascii')

    # ---------------
    # Call display func
    display_segmap_and_spec(x1d, insert_cat, all_inserted_segids, sn_segid,
                            imdat, segmap, sedlst)
