import numpy as np
from astropy.io import fits

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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

            plot_spec(wav, flam, ferr_lo, ferr_hi, pdf)

            x1d.close()

    print(len(mag_idx), num_overlap)
    print('{:.2f}'.format(num_overlap / len(mag_idx)), 'fraction overlap.')

    return None


extdir = '/Volumes/Joshi_external_HDD/Roman/'
results_dir = extdir + 'roman_slitless_sims_results/'
resfile = results_dir + 'fitting_results/zrecovery_pylinear_sims_pt0.txt'

cat = np.genfromtxt(resfile, dtype=None, names=True, encoding='ascii')

mags = cat['Y106mag']

m1 = 25.5
m2 = 26.0
m3 = 26.0
m4 = 26.5

bright = np.where((mags < m2) & (mags > m1))[0]
faint = np.where((mags < m4) & (mags > m3))[0]

pylinear_flam_scale_fac = 1e-17

pdf_bright = PdfPages(results_dir + 'completeness_26magbump_bright.pdf')
pdf_faint = PdfPages(results_dir + 'completeness_26magbump_faint.pdf')

loop_over_magbin(bright, pdf_bright)
loop_over_magbin(faint, pdf_faint)

pdf_bright.close()
pdf_faint.close()
