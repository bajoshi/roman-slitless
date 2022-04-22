import numpy as np
from astropy.io import fits

import os
import sys
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from fit_sn_romansim import get_optimal_position

"""
Code to investigate the source of the increase in completeness
at 26.0 <~ mag <~ 26.5 relative to 25.5 <~ mag <~ 26.0. Why do
fainter objects seem to have significantly better completeness
(~30% relative to 10%)?

This code plots all SN spectra within the two magnitude bins
(edges can be specified below) to two separate pdf files which
you can look at in more detail.

Because this can easily become a rabbit hole, will continue
this investigation later with the changes below.

Need to add:
1. Add input templates and fits to the plot.
2. Label some important features in the spectra.
3. Add info in text to plot: z_true, z_inferred with err,
z_acc, mag, segid, and img suffix.
"""


def plot_spec(wav, flam, felo, fehi, pdf):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    wav_idx = np.where((wav >= 7800) & (wav <= 17800))[0]

    wav = wav[wav_idx]
    flam = flam[wav_idx]
    felo = felo[wav_idx]
    fehi = fehi[wav_idx]

    ax.plot(wav, flam, lw=2.0, color='k')
    ax.fill_between(wav, flam - felo, flam + fehi,
                    color='gray', alpha=0.5)

    ax.set_xlim(7800, 17800)

    # plt.show()
    pdf.savefig(fig)

    fig.clear()
    plt.close(fig)

    return None


def loop_over_magbin(mag_idx, pdf):

    num_overlap = 0

    for i in range(len(mag_idx)):

        current_mag = mags[mag_idx][i]

        img_suffix = cat['img_suffix'][mag_idx][i]
        segid = cat['SNSegID'][mag_idx][i]

        overlap = cat['overlap'][mag_idx][i]

        print(i, current_mag, img_suffix, segid, overlap)

        if overlap:
            num_overlap += 1
        else:

            x1d = fits.open(results_dir + 'romansim_prism_'
                            + img_suffix + '_1200s_x1d.fits')

            wav = x1d[('SOURCE', segid)].data['wavelength']
            flam = x1d[('SOURCE', segid)].data['flam']\
                * pylinear_flam_scale_fac
            ferr_lo = x1d[('SOURCE', segid)].data['flounc']\
                * pylinear_flam_scale_fac
            ferr_hi = x1d[('SOURCE', segid)].data['fhiunc']\
                * pylinear_flam_scale_fac

            # ferr = (ferr_lo + ferr_hi) / 2.0

            plot_spec(wav, flam, ferr_lo, ferr_hi, pdf)

            x1d.close()

    print(len(mag_idx), num_overlap)
    print('{:.2f}'.format(num_overlap / len(mag_idx)), 'fraction overlap.')

    return None


def rerun_optpos_write(img_suffix, segid, fh, all_snr):

    # Loop over all exposure times
    all_exptimes = ['400s', '1200s', '3600s']

    for j in range(len(all_exptimes)):

        current_snr = all_snr[j]
        exptime = all_exptimes[j]

        if current_snr > 3.0:

            # ---- Get spectrum and optimal position
            x1d = fits.open(results_dir + 'romansim_prism_'
                            + img_suffix + '_'
                            + exptime + '_x1d.fits')

            wav = x1d[('SOURCE', segid)].data['wavelength']
            flam = x1d[('SOURCE', segid)].data['flam']\
                * pylinear_flam_scale_fac
            ferr_lo = x1d[('SOURCE', segid)].data['flounc']\
                * pylinear_flam_scale_fac
            ferr_hi = x1d[('SOURCE', segid)].data['fhiunc']\
                * pylinear_flam_scale_fac

            ferr = (ferr_lo + ferr_hi) / 2.0

            z_prior, phase_prior, av_prior = \
                get_optimal_position(wav, flam, ferr)

            x1d.close()

            # ---- Now construct part of new line to write
            fh.write('{:.4f}'.format(z_prior) + '  ')
            fh.write('-9999.0' + '  ')
            fh.write('-9999.0' + '  ')

            fh.write('{:.1f}'.format(phase_prior) + '  ')
            fh.write('-9999.0' + '  ')
            fh.write('-9999.0' + '  ')

            fh.write('{:.3f}'.format(av_prior) + '  ')
            fh.write('-9999.0' + '  ')
            fh.write('-9999.0' + '  ')

        else:
            fh.write('-9999.0' + '  ')
            fh.write('-9999.0' + '  ')
            fh.write('-9999.0' + '  ')

            fh.write('-9999.0' + '  ')
            fh.write('-9999.0' + '  ')
            fh.write('-9999.0' + '  ')

            fh.write('-9999.0' + '  ')
            fh.write('-9999.0' + '  ')
            fh.write('-9999.0' + '  ')

    return None


def rewrite_results(cat, resfile, m_low, m_high):
    """
    This function is to check if the new larger 
    optimal pos finding window improves the answer.
    Accepting the priors as the final answer for now.

    Will rewrite the results catalog with the new 
    estimates and check the completeness curve again.
    Run completenessvsexptime with the new file once
    this code is done.

    Loops over the catalog and for any SN within the
    provided mag range it will redo the optimal estimate
    and rewrite that row in the catalog.
    """

    # New results filename
    new_resfile = resfile.replace('.txt', '_new_optpos.txt')

    # First grab all lines from the old file
    with open(resfile, 'r') as fh_old:
        alllines = fh_old.readlines()

    # Open new file and loop over all rows
    with open(new_resfile, 'w') as fh:

        # Write header
        fh.write(alllines[0])

        for i in tqdm(range(len(cat))):

            current_mag = cat['Y106mag'][i]

            if (current_mag > m_low) and (current_mag < m_high):

                # Args to pass
                img_suffix = cat['img_suffix'][i]
                segid = cat['SNSegID'][i]
                true_z = cat['z_true'][i]
                true_phase = cat['phase_true'][i]
                true_av = cat['Av_true'][i]

                overlap = str(cat['overlap'][i])

                snr1 = cat['SNR400'][i]
                snr2 = cat['SNR1200'][i]
                snr3 = cat['SNR3600'][i]

                fh.write(img_suffix + '  ' + str(segid) + '  ')
                fh.write('{:.4f}'.format(true_z) + '  '
                         + str(true_phase) + '  ')
                fh.write('{:.3f}'.format(true_av) + '  ')
                fh.write(overlap + '  ')
                fh.write('{:.2f}'.format(current_mag) + '  ')
                fh.write('{:.2f}'.format(snr1) + '  ')
                fh.write('{:.2f}'.format(snr2) + '  ')
                fh.write('{:.2f}'.format(snr3) + '  ')

                all_snr = [snr1, snr2, snr3]

                # get new line to write
                rerun_optpos_write(img_suffix, segid, fh, all_snr)
                fh.write('\n')

            else:
                fh.write(alllines[i+1])

    return None


extdir = '/Volumes/Joshi_external_HDD/Roman/'
results_dir = extdir + 'roman_slitless_sims_results/run1/'
resfile = results_dir + 'fitting_results/zrecovery_pylinear_sims_pt0.txt'

cat = np.genfromtxt(resfile, dtype=None, names=True, encoding='ascii')

mags = cat['Y106mag']

m1 = 18.0
m2 = 26.0
m3 = 26.0
m4 = 29.0

pylinear_flam_scale_fac = 1e-17

rewrite_results(cat, resfile, m1, m4)

sys.exit(0)

bright = np.where((mags < m2) & (mags > m1))[0]
faint = np.where((mags < m4) & (mags > m3))[0]

pdf_bright = PdfPages(results_dir + 'completeness_26magbump_bright.pdf')
pdf_faint = PdfPages(results_dir + 'completeness_26magbump_faint.pdf')

loop_over_magbin(bright, pdf_bright)
loop_over_magbin(faint, pdf_faint)

pdf_bright.close()
pdf_faint.close()
