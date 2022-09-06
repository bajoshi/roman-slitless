import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import sys

# Define directories
extdir = "/Volumes/Joshi_external_HDD/Roman/"
ext_spectra_dir = extdir + "roman_slitless_sims_results/"
results_dir = ext_spectra_dir + 'fitting_results/'
new_results_dir = ext_spectra_dir + 'fitting_results_resamp/'


def grab_arrays(qty, exptime):
    """
    Using this function ensures that only results for
    the same SN get matched up. Useful if the reprocessing
    is still incomplete.
    """

    # Generate labels for header cols
    if qty == 'redshift':
        qty_lbl = 'z'
    elif qty == 'phase':
        qty_lbl = 'phase'
    elif qty == 'av':
        qty_lbl = 'Av'

    qty_lbl = qty_lbl + str(exptime)
    qty_lbl_lo = qty_lbl + '_lowerr'
    qty_lbl_hi = qty_lbl + '_uperr'

    # Now explicitly loop over each row while asserting
    #  that the same SN gets matched each time
    qty_old = []
    qty_old_lo = []
    qty_old_hi = []

    qty_new = []
    qty_new_lo = []
    qty_new_hi = []

    for i in range(len(oldres)):

        img_suffix_old = oldres['img_suffix'][i]
        sn_id_old = oldres['SNSegID'][i]

        snname_new_idx = np.where((newres['img_suffix'] == img_suffix_old) &
                                  (newres['SNSegID'] == sn_id_old))[0]

        if len(snname_new_idx) == 1:
            snname_new_idx = int(snname_new_idx)

            # print(img_suffix_old, sn_id_old, '   ',
            #       newres['img_suffix'][snname_new_idx],
            #       newres['SNSegID'][snname_new_idx])

            if (oldres[qty_lbl][i] == -9999.0) or \
               (newres[qty_lbl][i] == -9999.0):
                continue

            qty_old.append(oldres[qty_lbl][i])
            qty_old_lo.append(oldres[qty_lbl_lo][i])
            qty_old_hi.append(oldres[qty_lbl_hi][i])

            qty_new.append(newres[qty_lbl][i])
            qty_new_lo.append(newres[qty_lbl_lo][i])
            qty_new_hi.append(newres[qty_lbl_hi][i])

        # another check lathough should never be triggered
        elif len(snname_new_idx) > 1:
            print('IMG:', img_suffix_old)
            print('SNSegID:', sn_id_old)
            print('Too many matches. Exiting.')
            sys.exit(0)

    # Convert to numpy arrays and return
    qty_old = np.asarray(qty_old)
    qty_old_lo = np.asarray(qty_old_lo)
    qty_old_hi = np.asarray(qty_old_hi)

    qty_new = np.asarray(qty_new)
    qty_new_lo = np.asarray(qty_new_lo)
    qty_new_hi = np.asarray(qty_new_hi)

    return (qty_old, qty_old_lo, qty_old_hi,
            qty_new, qty_new_lo, qty_new_hi)


if __name__ == '__main__':

    # Read in old and new (resampled) file
    oldresfile = results_dir + 'zrecovery_pylinear_sims_pt0.txt'
    oldres = np.genfromtxt(oldresfile, dtype=None,
                           names=True, encoding='ascii')

    newresfile = new_results_dir + 'zrecovery_pylinear_sims_pt0_resamp.txt'
    newres = np.genfromtxt(newresfile, dtype=None,
                           names=True, encoding='ascii')

    # Assign required arrays
    # ----------
    (z400_old, z400_old_lo, z400_old_hi,
        z400_new, z400_new_lo, z400_new_hi) =\
        grab_arrays('redshift', 400)
    (phase400_old, phase400_old_lo, phase400_old_hi,
        phase400_new, phase400_new_lo, phase400_new_hi) =\
        grab_arrays('phase', 400)
    (av400_old, av400_old_lo, av400_old_hi,
        av400_new, av400_new_lo, av400_new_hi) =\
        grab_arrays('av', 400)

    # ----------
    (z1200_old, z1200_old_lo, z1200_old_hi,
        z1200_new, z1200_new_lo, z1200_new_hi) =\
        grab_arrays('redshift', 1200)
    (phase1200_old, phase1200_old_lo, phase1200_old_hi,
        phase1200_new, phase1200_new_lo, phase1200_new_hi) =\
        grab_arrays('phase', 1200)
    (av1200_old, av1200_old_lo, av1200_old_hi,
        av1200_new, av1200_new_lo, av1200_new_hi) =\
        grab_arrays('av', 1200)

    # ----------
    (z3600_old, z3600_old_lo, z3600_old_hi,
        z3600_new, z3600_new_lo, z3600_new_hi) =\
        grab_arrays('redshift', 3600)
    (phase3600_old, phase3600_old_lo, phase3600_old_hi,
        phase3600_new, phase3600_new_lo, phase3600_new_hi) =\
        grab_arrays('phase', 3600)
    (av3600_old, av3600_old_lo, av3600_old_hi,
        av3600_new, av3600_new_lo, av3600_new_hi) =\
        grab_arrays('av', 3600)

    # Figure
    fig = plt.figure(figsize=(9, 9))
    gs = GridSpec(12, 12, wspace=0.02, hspace=0.02,
                  left=0.05, right=0.95,
                  top=0.95, bottom=0.05)

    ewidth = 0.2
    ecol = 'r'
    
    # first row -- lowest exptime; 400 seconds
    # second row -- intermediate exptime; 1200 seconds
    # third row -- longest exptime; 3600 seconds
    ax1 = fig.add_subplot(gs[:3, 1:4])
    ax2 = fig.add_subplot(gs[:3, 5:8])
    ax3 = fig.add_subplot(gs[:3, 9:])

    ax4 = fig.add_subplot(gs[4:7, 1:4])
    ax5 = fig.add_subplot(gs[4:7, 5:8])
    ax6 = fig.add_subplot(gs[4:7, 9:])

    ax7 = fig.add_subplot(gs[8:11, 1:4])
    ax8 = fig.add_subplot(gs[8:11, 5:8])
    ax9 = fig.add_subplot(gs[8:11, 9:])

    # 
    ax1.errorbar(z400_new, z400_old,
                 xerr=[z400_new_lo, z400_new_hi],
                 yerr=[z400_old_lo, z400_old_hi],
                 elinewidth=ewidth, ms=1,
                 ecolor=ecol,
                 fmt='o', color='k')
    ax2.errorbar(phase400_new, phase400_old,
                 xerr=[phase400_new_lo, phase400_new_hi],
                 yerr=[phase400_old_lo, phase400_old_hi],
                 elinewidth=ewidth, ms=1,
                 ecolor=ecol,
                 fmt='o', color='k')
    ax3.errorbar(av400_new, av400_old,
                 xerr=[av400_new_lo, av400_new_hi],
                 yerr=[av400_old_lo, av400_old_hi],
                 elinewidth=ewidth, ms=1,
                 ecolor=ecol,
                 fmt='o', color='k')

    # 
    ax4.errorbar(z1200_new, z1200_old,
                 xerr=[z1200_new_lo, z1200_new_hi],
                 yerr=[z1200_old_lo, z1200_old_hi],
                 elinewidth=ewidth, ms=1,
                 ecolor=ecol,
                 fmt='o', color='k')
    ax5.errorbar(phase1200_new, phase1200_old,
                 xerr=[phase1200_new_lo, phase1200_new_hi],
                 yerr=[phase1200_old_lo, phase1200_old_hi],
                 elinewidth=ewidth, ms=1,
                 ecolor=ecol,
                 fmt='o', color='k')
    ax6.errorbar(av1200_new, av1200_old,
                 xerr=[av1200_new_lo, av1200_new_hi],
                 yerr=[av1200_old_lo, av1200_old_hi],
                 elinewidth=ewidth, ms=1,
                 ecolor=ecol,
                 fmt='o', color='k')

    # 
    ax7.errorbar(z3600_new, z3600_old,
                 xerr=[z3600_new_lo, z3600_new_hi],
                 yerr=[z3600_old_lo, z3600_old_hi],
                 elinewidth=ewidth, ms=1,
                 ecolor=ecol,
                 fmt='o', color='k')
    ax8.errorbar(phase3600_new, phase3600_old,
                 xerr=[phase3600_new_lo, phase3600_new_hi],
                 yerr=[phase3600_old_lo, phase3600_old_hi],
                 elinewidth=ewidth, ms=1,
                 ecolor=ecol,
                 fmt='o', color='k')
    ax9.errorbar(av3600_new, av3600_old,
                 xerr=[av3600_new_lo, av3600_new_hi],
                 yerr=[av3600_old_lo, av3600_old_hi],
                 elinewidth=ewidth, ms=1,
                 ecolor=ecol,
                 fmt='o', color='k')

    # Axes limits

    # Axes labels

    plt.show()

    sys.exit(0)
