from astropy.io import fits
import numpy as np
from scipy.interpolate import griddata

import emcee
import corner

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import os
import sys

from get_template_inputs import get_template_inputs

home = os.getenv('HOME')
roman_slitless_dir = home + '/Documents/GitHub/roman-slitless/'
fitting_utils = roman_slitless_dir + 'fitting_pipeline/utils/'
throughput_dir = fitting_utils + 'throughputs/'
extdir = '/Volumes/Joshi_external_HDD/Roman/'

sys.path.append(roman_slitless_dir)
sys.path.append(roman_slitless_dir + 'fitting_pipeline/')
from gen_sed_lst import get_sn_spec_path, get_gal_spec_path  # noqa
from model_sn import model_sn  # noqa

# Read in F106 filter
f106_filt_path = throughput_dir + 'F105W_IR_throughput.csv'
filt = np.genfromtxt(f106_filt_path,
                     delimiter=',', dtype=None, names=True,
                     encoding='ascii', usecols=(1, 2))

# And load table for SN Ia mF106 to z conversion
sn_mag_z = np.genfromtxt(fitting_utils + 'sn_mag_z_lookup.txt',
                         dtype=None, names=True, encoding='ascii')


def get_sn_mag_from_z(redshift):

    z_idx = np.argmin(np.abs(sn_mag_z['Redshift'] - redshift))
    snmag = float(sn_mag_z['mF106'][z_idx])

    return snmag


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


def read_sed_scale(sed_path, obj_mag, verbose=False):

    # First fix the path
    sed_path = sed_path.replace('/astro/ffsn/Joshi/', extdir)

    # Now check if it exists and if not then create it
    if not os.path.isfile(sed_path):
        template_inputs = get_template_inputs(sed_path, verbose=True)
        if ('salt' in sed_path)\
           or ('contam' in sed_path):
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
    if ('contam' in sed_path):
        roman_seds = os.path.dirname(sed_path) + '/'
        pure_sn_path = roman_seds + 'salt2_spec_day'\
            + str(template_day)\
            + "_z" + "{:.4f}".format(sn_z).replace('.', 'p')\
            + "_av" + "{:.3f}".format(template_av).replace('.', 'p')\
            + ".txt"
        sed = np.genfromtxt(pure_sn_path, dtype=None,
                            names=['wav', 'flam'], encoding='ascii')
    else:
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

    # get sed flam through filter
    sed_filt_flam = filter_conv(filt['Wave_Angstroms'], filt['Throughput'],
                                obj_wav, obj_flam)

    # This must be converted to fnu to work with AB mag
    # i.e., multiply by lam^2/c where lam is the pivot wav
    # for the F106 filter
    sed_filt_fnu = (10552**2 / 3e18) * sed_filt_flam
    implied_mag = -2.5 * np.log10(sed_filt_fnu) - 48.6

    required_fnu = np.power(10, ((obj_mag + 48.6) / -2.5))

    # Compute a scaling factor to scale the SED
    scalefac = required_fnu / sed_filt_fnu

    if verbose:
        print('Flam thru filter:', sed_filt_flam)
        print('Fnu thru filter:', sed_filt_fnu)
        print('Required Fnu:', required_fnu)

    # Now update the SED related quantities
    sed_filt_fnu *= scalefac
    implied_mag = -2.5 * np.log10(sed_filt_fnu) - 48.6

    if verbose:
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


def get_appmag_x1d_f106(x1d_wav, x1d_flam):

    # Now convolve with F106
    x1d_flam = filter_conv(filt['Wave_Angstroms'], filt['Throughput'],
                           x1d_wav, x1d_flam)

    # Convert to fnu and get AB mag
    lam_pivot = 10552.0  # hardcoded for F105W
    fnu_conv = lam_pivot**2 * x1d_flam / 3e18
    appmag = -2.5 * np.log10(fnu_conv) - 48.6

    return appmag


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
               bottom - int(ext_diff / 2),
               top + int(ext_diff / 2)]
    elif y_extent > x_extent:
        ext_diff = y_extent - x_extent
        ext = [left - int(ext_diff / 2),
               right + int(ext_diff / 2),
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

    # Most host extracted spectra will not match the template.
    # Need to figure out if the supplied host mag
    # is too faint or if it is an issue with pylinear
    # flux calibration.
    # Get teh apparent mag of the extracted spectrum
    # and scale the template to match that.
    host_appmag = get_appmag_x1d_f106(host_wav, host_flam)
    sn_appmag = get_appmag_x1d_f106(sn_wav, sn_flam)

    sn_template_wav, sn_template_flam = read_sed_scale(sn_sed_path, snmag)
    host_template_wav, host_template_flam = \
        read_sed_scale(host_sed_path, host_appmag)

    print('\nMatched/SN Seg ID:   ', sn_segid,
          '\nSN magnitude:        ', '{:.3f}'.format(snmag),
          '\nSN magnitude (from x1d):', '{:.3f}'.format(sn_appmag),
          '\nHost galaxy mag:     ', '{:.3f}'.format(hostmag),
          '\nHost galaxy mag (from x1d):', '{:.3f}'.format(host_appmag),
          '\nHost galaxy Seg ID:  ', host_segid,
          '\nInserted object type:', obj_type,
          '\n---------------------\n')

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

    if not os.path.isdir(figdir):
        os.mkdir(figdir)

    figname = figdir + 'sn_' + str(sn_segid) + '_Y106_0_' + \
        str(detector) + '.pdf'
    fig.savefig(figname, dpi=300, bbox_inches='tight')

    return None


def plot_best_fit(axes, sn_segid, sn_spec_name, detector, exptime,
                  wav, flam, ferr):

    # ----- Get sampler and flat samples
    fitting_resdir = extdir\
        + 'roman_slitless_sims_results/fitting_results/'
    img_suffix = 'Y106_0_' + str(detector)
    snstr = str(sn_segid) + '_' + img_suffix + '_' + exptime
    emcee_savefile = fitting_resdir + 'emcee_sampler_sn' \
        + snstr + '.h5'

    sampler = emcee.backends.HDFBackend(emcee_savefile)
    burn_in = 200
    thinning_steps = 30
    flat_samples = sampler.get_chain(discard=burn_in,
                                     thin=thinning_steps, flat=True)

    # corner estimates
    cq_z = corner.quantile(x=flat_samples[:, 0], q=[0.16, 0.5, 0.84])
    cq_day = corner.quantile(x=flat_samples[:, 1], q=[0.16, 0.5, 0.84])
    cq_av = corner.quantile(x=flat_samples[:, 2], q=[0.16, 0.5, 0.84])

    # ----- Now plot best fit and uncertainties
    model_count = 0
    ind_list = []
    all_chi2 = []

    while model_count <= 200:

        ind = int(np.random.randint(len(flat_samples), size=1))
        ind_list.append(ind)

        # make sure sample has correct shape
        sample = flat_samples[ind]
        
        model_okay = 0

        sample = sample.reshape(3)

        # Get the parameters of the sample
        model_z = sample[0]
        model_day = sample[1]
        model_av = sample[2]

        # Check that the model is within +-1 sigma
        # of value inferred by corner contours
        if (model_z >= cq_z[0]) and (model_z <= cq_z[2]) and \
           (model_day >= cq_day[0]) and (model_day <= cq_day[2]) and \
           (model_av >= cq_av[0]) and (model_av <= cq_av[2]):

            model_okay = 1

        # Now plot if the model is okay
        if model_okay:

            m = model_sn(wav, sample[0], sample[1], sample[2])

            a = np.nansum(flam * m / ferr**2) / np.nansum(m**2 / ferr**2)
            m = m * a

            axes.plot(wav, m, color='magenta', lw=4.0, alpha=0.02, zorder=2)

            model_count += 1

            chi2 = np.nansum((m - flam)**2/ferr**2)
            all_chi2.append(chi2)

    axes.text(x=0.58, y=0.3, s='---  best fit models', 
              verticalalignment='top', horizontalalignment='left', 
              transform=axes.transAxes, color='magenta', size=14)

    return all_chi2


def underest_z_figure(ext_hdu, sn_segid, sedlst, detector, exptime,
                      figdir):

    sn_idx = int(np.where(sedlst['segid'] == sn_segid)[0])
    sn_spec_name = os.path.basename(sedlst['sed_path'][sn_idx])

    if 'contam' in sn_spec_name:
        h1 = sn_spec_name.split('_')[1]
        host_segid = int(h1[4:])

        host_wav = ext_hdu[('SOURCE', host_segid)].data['wavelength']
        host_flam = ext_hdu[('SOURCE', host_segid)].data['flam'] * 1e-17
        host_ferr_lo = ext_hdu[('SOURCE', host_segid)].data['flounc'] * 1e-17
        host_ferr_hi = ext_hdu[('SOURCE', host_segid)].data['fhiunc'] * 1e-17

    elif 'salt2' in sn_spec_name:
        host_segid = -99

    # ---------------
    # Get extracted 1d spec
    sn_wav = ext_hdu[('SOURCE', sn_segid)].data['wavelength']
    sn_flam = ext_hdu[('SOURCE', sn_segid)].data['flam'] * 1e-17
    sn_ferr_lo = ext_hdu[('SOURCE', sn_segid)].data['flounc'] * 1e-17
    sn_ferr_hi = ext_hdu[('SOURCE', sn_segid)].data['fhiunc'] * 1e-17
    sn_ferr = (sn_ferr_lo + sn_ferr_hi)/2

    # ---------------
    # Get redshift
    sn_sed_path = sedlst['sed_path'][sn_idx]

    # Get redshift to put in axes title
    sn_basename = os.path.basename(sn_sed_path)
    tl = sn_basename.split('_')
    sn_z = tl[3].replace('z', '').replace('p', '.')
    redshift = float(sn_z)

    # ----- First get truth values
    template_inputs = get_template_inputs(sn_spec_name)
    print(template_inputs)

    truth_dict = {}
    truth_dict['z'] = template_inputs[0]
    truth_dict['phase'] = template_inputs[1]
    truth_dict['Av'] = template_inputs[2]

    # ---------------
    # ---- figure and subplots
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    ax.set_ylabel(r'$\mathrm{f_\lambda\ [erg\, s^{-1}\, cm^{-2}\, \AA^{-1}]}$',  # noqa
                  fontsize=18)
    ax.set_xlabel(r'$\mathrm{\lambda\ [\AA]}$', fontsize=18)

    # Plot extracted spectra
    ax.plot(sn_wav, sn_flam, color='k', lw=2.0,
            label='pyLINEAR x1d spec SN', zorder=3)
    ax.fill_between(sn_wav, sn_flam - sn_ferr_lo, sn_flam + sn_ferr_hi,
                    color='gray', alpha=0.5)

    if host_segid != -99:
        ax.plot(host_wav, host_flam, color='crimson', lw=1.5,
                label='pyLINEAR x1d spec host-galaxy',
                zorder=3)
        # ax.fill_between(host_wav, host_flam - host_ferr_lo,
        #                 host_flam + host_ferr_hi,
        #                 color='gray', alpha=0.5, zorder=3)

    # ---- Limits
    # Decide Y limits based on flux limits
    wmin = 7600.0
    wmax = 18200.0
    wav_idx = np.where((sn_wav >= 11000) & (sn_wav <= 17500))
    sn_min_flux = np.nanmin(sn_flam[wav_idx])
    sn_max_flux = np.nanmax(sn_flam[wav_idx])

    if host_segid != -99:
        host_min_flux = np.nanmin(host_flam[wav_idx])
        host_max_flux = np.nanmax(host_flam[wav_idx])

        min_flux = min([sn_min_flux, host_min_flux])
        max_flux = max([sn_max_flux, host_max_flux])
    else:
        min_flux = sn_min_flux
        max_flux = sn_max_flux

    # ax.set_ylim(min_flux * 0.8, max_flux * 1.5)
    ax.set_ylim(2e-20, 5e-18)
    ax.set_xlim(wmin, wmax)

    ax.set_yscale('log')

    # ---- mark some prominent features
    # mark_lines(ax, redshift)

    # ---- Axes title
    # ax.set_title('1-hr prism; z = '
    #              + '{:.3f}'.format(redshift) + '  '
    #              + 'zest = ' + '{:.3f}'.format(zest) + '  '
    #              + 'Age True = ' + '{:.1f}'.format(phase_true) + '  '
    #              + 'Age = ' + '{:.1f}'.format(phase),
    #              fontsize=14)

    # ---- Plot the best fit and uncertainties
    all_chi2 = plot_best_fit(ax, sn_segid, sn_spec_name, detector, exptime,
                             sn_wav, sn_flam, sn_ferr)

    # ---- Input template at correct redshift
    m = model_sn(sn_wav,
                 template_inputs[0],
                 template_inputs[1],
                 template_inputs[2])
    a = np.nansum(sn_flam * m / sn_ferr**2) / np.nansum(m**2 / sn_ferr**2)
    m = m * a

    ax.plot(sn_wav, m, color='dodgerblue', lw=2.0,
            label='SN input template')

    # print('True template chi2::', np.sum((m - sn_flam)**2/sn_ferr**2))
    # print('All best fit models chi2:\n', all_chi2)

    ax.legend(loc='lower right', frameon=False, fontsize=14)

    # -----
    if not os.path.isdir(figdir):
        os.mkdir(figdir)

    figname = figdir + 'sn_' + str(sn_segid) + '_Y106_0_' + \
        str(detector) + '_' + exptime + '_paperfig.pdf'
    fig.savefig(figname, dpi=300, bbox_inches='tight')

    fig.clear()
    plt.close(fig)

    return None


def paperfigure(ext_hdu, sn_segid, host_segid, sedlst,
                detector, exptime, sub_const=None,
                figdir=roman_slitless_dir + 'figures/sn_spectra_inspect/'):
    """
    This is just a pared down version of the above display_segmap_and_spec()
    function to produce figures for the paper. See that function for notes.
    """

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
    # Read in SED templates for SN and host
    sn_sed_idx = int(np.where(sedlst['segid'] == sn_segid)[0])
    host_sed_idx = int(np.where(sedlst['segid'] == host_segid)[0])

    sn_sed_path = sedlst['sed_path'][sn_sed_idx]
    host_sed_path = sedlst['sed_path'][host_sed_idx]

    # Get redshift to put in axes title
    sn_basename = os.path.basename(sn_sed_path)
    tl = sn_basename.split('_')
    sn_z = tl[3].replace('z', '').replace('p', '.')
    redshift = float(sn_z)

    # Most host extracted spectra will not match the template.
    # Need to figure out if the supplied host mag
    # is too faint or if it is an issue with pylinear
    # flux calibration.
    # Get teh apparent mag of the extracted spectrum
    # and scale the template to match that.
    host_appmag = get_appmag_x1d_f106(host_wav, host_flam)

    sn_appmag = get_sn_mag_from_z(redshift)

    sn_template_wav, sn_template_flam = read_sed_scale(sn_sed_path, sn_appmag)
    host_template_wav, host_template_flam = \
        read_sed_scale(host_sed_path, host_appmag)

    # ---------------
    # ---- figure and subplots
    fig = plt.figure(figsize=(15, 4))
    ax = fig.add_subplot(111)

    # ---- Set fontsize for legend, labels and title
    label_fs = 18
    legend_fs = 15

    # ---- Set labels
    ax.set_ylabel(r'$\mathrm{f_\lambda\ [erg\, s^{-1}\, cm^{-2}\, \AA^{-1}]}$',  # noqa
                  fontsize=label_fs)
    ax.set_xlabel(r'$\mathrm{\lambda\ [\AA]}$', fontsize=label_fs)

    # Plot extracted spectra
    ax.plot(sn_wav, sn_flam, color='k', lw=1.5, label='pyLINEAR x1d spec SN')
    ax.fill_between(sn_wav, sn_flam - sn_ferr_lo, sn_flam + sn_ferr_hi,
                    color='gray', alpha=0.5)

    # ax2.plot(host_wav, host_flam, color='k', lw=1.5,
    #          label='pyLINEAR x1d spec host-galaxy',
    #          zorder=3)
    # ax2.fill_between(host_wav, host_flam - host_ferr_lo,
    #                  host_flam + host_ferr_hi,
    #                  color='gray', alpha=0.5, zorder=3)

    # ---- Plot templates
    ax.plot(sn_template_wav, sn_template_flam,
            color='dodgerblue', lw=2.0,
            label='SN template'
            + '\n' + 'SN F106 mag: ' + '{:.2f}'.format(sn_appmag))

    # If we need to subtract a constant from host flambda
    if sub_const is not None:
        host_template_flam -= sub_const
        ax.set_ylabel(r'$\mathrm{f_\lambda\ + constant}$',  # noqa
                      fontsize=label_fs)

    ax.plot(host_template_wav, host_template_flam,
            color='crimson', lw=1.5,
            label='Host-galaxy template'
            + '\n' + 'Host-galaxy F106 mag: ' + '{:.2f}'.format(host_appmag),
            alpha=0.7, zorder=2)

    # ---- Legend
    ax.legend(loc='lower right', frameon=False, fontsize=legend_fs)

    # ---- Limits
    # Decide Y limits based on flux limits
    wmin = 7600.0
    wmax = 18200.0
    wav_idx = np.where((sn_wav >= 11000) & (sn_wav <= 17500))
    sn_min_flux = np.nanmin(sn_flam[wav_idx])
    sn_max_flux = np.nanmax(sn_flam[wav_idx])

    if host_segid != -99:
        host_min_flux = np.nanmin(host_flam[wav_idx])
        host_max_flux = np.nanmax(host_flam[wav_idx])

        min_flux = min([sn_min_flux, host_min_flux])
        max_flux = max([sn_max_flux, host_max_flux])
    else:
        min_flux = sn_min_flux
        max_flux = sn_max_flux

    ax.set_ylim(min_flux * 0.1, max_flux * 2.0)
    # ax.set_ylim(1e-21, 1e-17)
    ax.set_xlim(wmin, wmax)

    ax.set_yscale('log')

    # Force X limits to be the same for both axes
    ax.set_xlim(7600.0, 18200.0)

    ax.tick_params(which='both', labelsize=18)

    # ---- Axes title
    ax.set_title('1-hour prism spectrum; z = '
                 + '{:.3f}'.format(redshift), fontsize=label_fs+2)

    # ---- Save figure
    if not os.path.isdir(figdir):
        os.mkdir(figdir)

    figname = figdir + 'sn_' + str(sn_segid) + '_Y106_0_' + \
        str(detector) + '_' + exptime + '_paperfig.pdf'
    fig.savefig(figname, dpi=300, bbox_inches='tight')

    return None


def get_host_segid(sedlst, sn_segid):

    sn_idx = int(np.where(sedlst['segid'] == sn_segid)[0])
    sn_spec_name = os.path.basename(sedlst['sed_path'][sn_idx])

    if 'contam' in sn_spec_name:
        h1 = sn_spec_name.split('_')[1]
        host_segid = int(h1[4:])
    else:
        host_segid = -99

    return host_segid


if __name__ == '__main__':

    # ---------------
    # Info for code to plot
    detector = int(sys.argv[1])
    sn_segid = int(sys.argv[2])
    exptime = sys.argv[3]

    try:
        sub_const = float(sys.argv[4])  # only needed for paperfigure()
    except IndexError:
        sub_const = None

    # THIS CODE IS INTENDED TO ONLY BE RUN AFTER
    # PLFFSN2 FINISHES AND NOT ON PLFFSN2.
    # You will also need to rsync the following from PLFFSN2:
    # 1. *SNadded.fits, *SNadded_segmap.fits, and *SNadded.npy
    # 2. SED LST files
    # 3. x1d results
    ######################

    # ---------------
    # Read in npy file with all inserted object info
    # img_sim_dir = extdir + 'roman_direct_sims/sims2021/K_5degimages_part1/'
    # insert_npy_path = img_sim_dir + '5deg_Y106_0_' +\
    #     detector + '_SNadded.npy'

    # insert_cat = np.load(insert_npy_path)

    # # Grab all inserted segids
    # all_inserted_segids = insert_cat[:, -1].astype(np.int64)

    # ---------------
    # Read in extracted spectra
    resdir = extdir + 'roman_slitless_sims_results/'
    x1d = fits.open(resdir + 'romansim_prism_Y106_0_'
                    + str(detector) + '_' + exptime + '_x1d.fits')

    # ---------------
    # Read in sedlst
    sedlst_path = extdir\
        + 'pylinear_lst_files/sed_Y106_0_' + str(detector) + '.lst'
    sedlst = np.genfromtxt(sedlst_path, dtype=None,
                           names=['segid', 'sed_path'], encoding='ascii')

    host_segid = get_host_segid(sedlst, sn_segid)
    print('Host SegID:', host_segid)

    # ---------------
    # paperfigure(x1d, sn_segid, host_segid, sedlst,
    #             detector, exptime, sub_const)
    underest_z_figure(x1d, sn_segid, sedlst, detector, exptime,
                      figdir=roman_slitless_dir + 
                      'figures/sn_spectra_inspect/')

    sys.exit(0)

    # ---------------
    # Read in image and segmap
    img_path = img_sim_dir + '5deg_Y106_0_' + detector + '_SNadded.fits'
    segmap_path = img_sim_dir + '5deg_Y106_0_' + \
        detector + '_SNadded_segmap.fits'

    imdat = fits.getdata(img_path)
    segmap = fits.getdata(segmap_path)

    # ---------------
    # Call display func
    # display_segmap_and_spec(x1d, insert_cat, all_inserted_segids, sn_segid,
    #                         imdat, segmap, sedlst)
