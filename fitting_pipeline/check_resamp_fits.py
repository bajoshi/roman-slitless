import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import os
import sys

# Define directories
extdir = "/Volumes/Joshi_external_HDD/Roman/"
ext_spectra_dir = extdir + "roman_slitless_sims_results/"
results_dir = ext_spectra_dir + 'fitting_results/'
new_results_dir = ext_spectra_dir + 'fitting_results_resamp/'

home = os.getenv('HOME')
roman_slitless_dir = home + '/Documents/GitHub/roman-slitless/'
figdir = roman_slitless_dir + 'figures/'


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

    qty_true = []

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

            # Check with a S/N cut as well
            # if oldres['SNR' + str(exptime)][i] < 10.0:
            #     continue

            # Also return the true values
            if qty == 'redshift':
                ztrue = oldres['z_true'][i]
                qty_true.append(ztrue)
            elif qty == 'phase':
                phasetrue = oldres['phase_true'][i]
                qty_true.append(phasetrue)
            elif qty == 'av':
                avtrue = oldres['Av_true'][i]
                qty_true.append(avtrue)

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

    qty_true = np.asarray(qty_true)

    return (qty_old, qty_old_lo, qty_old_hi,
            qty_new, qty_new_lo, qty_new_hi,
            qty_true)


def get_sigmaz(oldz, newz, ztrue):

    old_zdiff = oldz - ztrue
    old_sigmaz = old_zdiff / (1 + ztrue)

    new_zdiff = newz - ztrue
    new_sigmaz = new_zdiff / (1 + ztrue)

    return old_sigmaz, new_sigmaz


def construct_color_array(old_sigmaz, new_sigmaz, sigmaz_thresh):

    # Have to construct a list first because
    # I can't think of a way to create an empty
    # numpy array of strings of len 1 and have the
    # array be of a certain len.
    color_arr = []

    for c in range(len(old_sigmaz)):
        # Case 1: both pass
        if (np.abs(old_sigmaz[c]) <= sigmaz_thresh) and\
             (np.abs(new_sigmaz[c]) <= sigmaz_thresh):
            color_arr.append('k')
        # Case 2: old sampling pass but new fail
        if (np.abs(old_sigmaz[c]) <= sigmaz_thresh) and\
                (np.abs(new_sigmaz[c]) > sigmaz_thresh):
            color_arr.append('b')
        # Case 3: new sampling pass but old fail
        if (np.abs(old_sigmaz[c]) > sigmaz_thresh) and\
                (np.abs(new_sigmaz[c]) <= sigmaz_thresh):
            color_arr.append('r')
        # Case 4: both fail
        if (np.abs(old_sigmaz[c]) > sigmaz_thresh) and\
                (np.abs(new_sigmaz[c]) > sigmaz_thresh):
            color_arr.append('none')

    color_arr = np.array(color_arr, dtype=str)

    # Also print to screen how many points in each bin
    pass_idx = np.where(color_arr == 'k')[0]
    blue_idx = np.where(color_arr == 'b')[0]
    red_idx = np.where(color_arr == 'r')[0]

    print('\nNumber of points for each method being better:')
    print('Total points incl failures:', len(old_sigmaz))
    print('both sampling give good results:', len(pass_idx))
    print('constant sampling is better:', len(blue_idx))
    print('new sampling is better:', len(red_idx))

    old_pass_frac = (len(pass_idx) + len(blue_idx)) / len(old_sigmaz)
    new_pass_frac = (len(pass_idx) + len(red_idx)) / len(old_sigmaz)
    print('Total pass fraction OLD:', '{:.4f}'.format(old_pass_frac))
    print('Total pass fraction NEW:', '{:.4f}'.format(new_pass_frac))
    print('DIFF:', '{:.4f}'.format(new_pass_frac - old_pass_frac))

    return color_arr


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
        z400_new, z400_new_lo, z400_new_hi,
        ztrue_400) =\
        grab_arrays('redshift', 400)
    (phase400_old, phase400_old_lo, phase400_old_hi,
        phase400_new, phase400_new_lo, phase400_new_hi,
        phasetrue_400) =\
        grab_arrays('phase', 400)
    (av400_old, av400_old_lo, av400_old_hi,
        av400_new, av400_new_lo, av400_new_hi,
        avtrue_400) =\
        grab_arrays('av', 400)

    # ----------
    (z1200_old, z1200_old_lo, z1200_old_hi,
        z1200_new, z1200_new_lo, z1200_new_hi,
        ztrue_1200) =\
        grab_arrays('redshift', 1200)
    (phase1200_old, phase1200_old_lo, phase1200_old_hi,
        phase1200_new, phase1200_new_lo, phase1200_new_hi,
        phasetrue_1200) =\
        grab_arrays('phase', 1200)
    (av1200_old, av1200_old_lo, av1200_old_hi,
        av1200_new, av1200_new_lo, av1200_new_hi,
        avtrue_1200) =\
        grab_arrays('av', 1200)

    # ----------
    (z3600_old, z3600_old_lo, z3600_old_hi,
        z3600_new, z3600_new_lo, z3600_new_hi,
        ztrue_3600) =\
        grab_arrays('redshift', 3600)
    (phase3600_old, phase3600_old_lo, phase3600_old_hi,
        phase3600_new, phase3600_new_lo, phase3600_new_hi,
        phasetrue_3600) =\
        grab_arrays('phase', 3600)
    (av3600_old, av3600_old_lo, av3600_old_hi,
        av3600_new, av3600_new_lo, av3600_new_hi,
        avtrue_3600) =\
        grab_arrays('av', 3600)

    # Generate color arrays
    # Color points where the result is too far
    # from the truth for either sampling method.
    # Also, remember that there are no redshifts in
    # the simulation that are z < 0.5.
    # Four cases:
    # first set sigma_z threshold = 0.01 or 0.02
    # 1. Both methods agree with truth to 
    # within sigma_z threshold. So neither sampling
    # is better. These points are colored black.
    # 2. Old constant sampling is within threshold
    # and new sampling is not. Points colored blue.
    # 3. New sampling is within threshold and old
    # sampling is not. Points colored red.
    # 4. Both methods failed. These points are
    # discarded because again neither method is better.

    # Set threshold
    sigmaz_thresh = 0.01

    # Get the sigma_z arrays
    # NOTE: While the truth values are the same
    # NOT every SN gets fit for each exptime so the
    # truth redshift arrays are assigned separately
    # to account for increasing SNe being fit at longer
    # exptimes. E.g., the 20 min exptime only has about
    # half the sample size of the 3 hr exptime.
    old_sigmaz_400, new_sigmaz_400 = get_sigmaz(z400_old, z400_new, ztrue_400)
    old_sigmaz_1200, new_sigmaz_1200 = get_sigmaz(z1200_old, z1200_new,
                                                  ztrue_1200)
    old_sigmaz_3600, new_sigmaz_3600 = get_sigmaz(z3600_old, z3600_new,
                                                  ztrue_3600)

    # Now determine where the points are relative
    # to the threshold and decide colors.
    color_z400 = construct_color_array(old_sigmaz_400, new_sigmaz_400,
                                       sigmaz_thresh)
    color_z1200 = construct_color_array(old_sigmaz_1200, new_sigmaz_1200,
                                        sigmaz_thresh)
    color_z3600 = construct_color_array(old_sigmaz_3600, new_sigmaz_3600,
                                        sigmaz_thresh)

    # Figure
    fig = plt.figure(figsize=(9, 9))
    gs = GridSpec(12, 12, wspace=0.02, hspace=0.02,
                  left=0.05, right=0.95,
                  top=0.95, bottom=0.05)

    ewidth = 0.0
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
    ax1.scatter(z400_new, z400_old, facecolors=color_z400, s=2)
    ax2.scatter(phase400_new, phase400_old, facecolors='k', s=2)
    ax3.scatter(av400_new, av400_old, facecolors='k', s=2)

    # 
    ax4.scatter(z1200_new, z1200_old, facecolors=color_z1200, s=2)
    ax5.scatter(phase1200_new, phase1200_old, facecolors='k', s=2)
    ax6.scatter(av1200_new, av1200_old, facecolors='k', s=2)

    # 
    ax7.scatter(z3600_new, z3600_old, facecolors=color_z3600, s=2)
    ax8.scatter(phase3600_new, phase3600_old, facecolors='k', s=2)
    ax9.scatter(av3600_new, av3600_old, facecolors='k', s=2)

    # Axes limits
    ax1.set_xlim(-0.05, 3.05)
    ax1.set_ylim(-0.05, 3.05)
    ax4.set_xlim(-0.05, 3.05)
    ax4.set_ylim(-0.05, 3.05)
    ax7.set_xlim(-0.05, 3.05)
    ax7.set_ylim(-0.05, 3.05)

    ax2.set_xlim(-20, 20)
    ax2.set_ylim(-20, 20)
    ax5.set_xlim(-20, 20)
    ax5.set_ylim(-20, 20)
    ax8.set_xlim(-20, 20)
    ax8.set_ylim(-20, 20)

    ax3.set_xlim(-0.1, 5.1)
    ax3.set_ylim(-0.1, 5.1)
    ax6.set_xlim(-0.1, 5.1)
    ax6.set_ylim(-0.1, 5.1)
    ax9.set_xlim(-0.1, 5.1)
    ax9.set_ylim(-0.1, 5.1)

    # Axes labels
    ax1.set_xlabel('Redshift resampled', fontsize=11)
    ax1.set_ylabel('Redshift const-samp', fontsize=11)
    ax4.set_xlabel('Redshift resampled', fontsize=11)
    ax4.set_ylabel('Redshift const-samp', fontsize=11)
    ax7.set_xlabel('Redshift resampled', fontsize=11)
    ax7.set_ylabel('Redshift const-samp', fontsize=11)

    ax2.set_xlabel('Phase resampled', fontsize=11)
    ax2.set_ylabel('Phase const-samp', fontsize=11)
    ax5.set_xlabel('Phase resampled', fontsize=11)
    ax5.set_ylabel('Phase const-samp', fontsize=11)
    ax8.set_xlabel('Phase resampled', fontsize=11)
    ax8.set_ylabel('Phase const-samp', fontsize=11)

    ax3.set_xlabel('Av resampled', fontsize=11)
    ax3.set_ylabel('Av const-samp', fontsize=11)
    ax6.set_xlabel('Av resampled', fontsize=11)
    ax6.set_ylabel('Av const-samp', fontsize=11)
    ax9.set_xlabel('Av resampled', fontsize=11)
    ax9.set_ylabel('Av const-samp', fontsize=11)

    # Title for exposure times
    ax2.set_title('Exposure time: 20 min', fontsize=13)
    ax5.set_title('Exposure time: 1 hr', fontsize=13)
    ax8.set_title('Exposure time: 3 hr', fontsize=13)

    fig.savefig(figdir + 'resampling_check.pdf', dpi=200,
                bbox_inches='tight')
    fig.clear()
    plt.close(fig)

    # Plot histograms in different redshift bins to check
    # any redshift dependence.
    # We are going to make a histogram of the redshift diff
    # as a function of the old redshift. The redshift diff
    # is the abs value of the difference between the old
    # and new recovered redshift.
    # This will also be binned in z.
    # e.g., make a background histogram of all zdiff
    # now choose all old z within 0 <= z < 0.5
    # overplot a step histogram of z diff for just these values
    # do the same for 0.5 <= z < 1.0 and 1.0 <= z < 2.0 and z >= 2.0
    # The number of "outliers" in these histograms, say all
    # zdiff > some number, should be quantified and quoted in some way
    # in the paper.
    fig = plt.figure(figsize=(9, 2.5))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    sigmazdiff_400 = np.abs(old_sigmaz_400 - new_sigmaz_400)
    sigmazdiff_1200 = np.abs(old_sigmaz_1200 - new_sigmaz_1200)
    sigmazdiff_3600 = np.abs(old_sigmaz_3600 - new_sigmaz_3600)

    # define range
    rng = (0.0, 0.6)

    ax1.hist(sigmazdiff_400, 30, histtype='step',
             color='b', range=rng)
    ax2.hist(sigmazdiff_1200, 30, histtype='step',
             color='b', range=rng)
    ax3.hist(sigmazdiff_3600, 30, histtype='step',
             color='b', range=rng)

    # Redshift bins
    zbin1_400 = np.where((ztrue_400 >= 0.5) & (ztrue_400 < 1.0))[0]
    zbin2_400 = np.where((ztrue_400 >= 1.0) & (ztrue_400 < 2.0))[0]
    zbin3_400 = np.where((ztrue_400 >= 2.0) & (ztrue_400 < 3.0))[0]

    zbin1_1200 = np.where((ztrue_1200 >= 0.5) & (ztrue_1200 < 1.0))[0]
    zbin2_1200 = np.where((ztrue_1200 >= 1.0) & (ztrue_1200 < 2.0))[0]
    zbin3_1200 = np.where((ztrue_1200 >= 2.0) & (ztrue_1200 < 3.0))[0]

    zbin1_3600 = np.where((ztrue_3600 >= 0.5) & (ztrue_3600 < 1.0))[0]
    zbin2_3600 = np.where((ztrue_3600 >= 1.0) & (ztrue_3600 < 2.0))[0]
    zbin3_3600 = np.where((ztrue_3600 >= 2.0) & (ztrue_3600 < 3.0))[0]

    ax1.hist(sigmazdiff_400[zbin1_400], 30, histtype='step',
             color='seagreen', range=rng)
    ax1.hist(sigmazdiff_400[zbin2_400], 30, histtype='step',
             color='orchid', range=rng)
    ax1.hist(sigmazdiff_400[zbin3_400], 30, histtype='step',
             color='crimson', range=rng)

    ax2.hist(sigmazdiff_1200[zbin1_1200], 30, histtype='step',
             color='seagreen', range=rng)
    ax2.hist(sigmazdiff_1200[zbin2_1200], 30, histtype='step',
             color='orchid', range=rng)
    ax2.hist(sigmazdiff_1200[zbin3_1200], 30, histtype='step',
             color='crimson', range=rng)

    ax3.hist(sigmazdiff_3600[zbin1_3600], 30, histtype='step',
             color='seagreen', range=rng)
    ax3.hist(sigmazdiff_3600[zbin2_3600], 30, histtype='step',
             color='orchid', range=rng)
    ax3.hist(sigmazdiff_3600[zbin3_3600], 30, histtype='step',
             color='crimson', range=rng)

    # Title for exposure times
    ax1.set_title('Exposure time: 20 min', fontsize=12)
    ax2.set_title('Exposure time: 1 hr', fontsize=12)
    ax3.set_title('Exposure time: 3 hr', fontsize=12)

    fig.savefig(figdir + 'resampling_check_hist.pdf', dpi=200,
                bbox_inches='tight')
    fig.clear()
    plt.close(fig)

    # Figure to check scatter vs z 
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    # Only incl what passes threshold
    old_pass_400 = np.where(np.abs(old_sigmaz_400) <= sigmaz_thresh)[0]
    new_pass_400 = np.where(np.abs(new_sigmaz_400) <= sigmaz_thresh)[0]

    old_pass_1200 = np.where(np.abs(old_sigmaz_1200) <= sigmaz_thresh)[0]
    new_pass_1200 = np.where(np.abs(new_sigmaz_1200) <= sigmaz_thresh)[0]

    old_pass_3600 = np.where(np.abs(old_sigmaz_3600) <= sigmaz_thresh)[0]
    new_pass_3600 = np.where(np.abs(new_sigmaz_3600) <= sigmaz_thresh)[0]

    ax1.scatter(ztrue_400[old_pass_400], old_sigmaz_400[old_pass_400],
                color='b', s=2)
    ax1.scatter(ztrue_400[new_pass_400], new_sigmaz_400[new_pass_400],
                color='r', s=2)

    ax2.scatter(ztrue_1200[old_pass_1200], old_sigmaz_1200[old_pass_1200],
                color='b', s=2)
    ax2.scatter(ztrue_1200[new_pass_1200], new_sigmaz_1200[new_pass_1200],
                color='r', s=2)

    ax3.scatter(ztrue_3600[old_pass_3600], old_sigmaz_3600[old_pass_3600],
                color='b', s=2)
    ax3.scatter(ztrue_3600[new_pass_3600], new_sigmaz_3600[new_pass_3600],
                color='r', s=2)

    fig.savefig(figdir + 'resampling_check_scatter.pdf', dpi=200,
                bbox_inches='tight')
    fig.clear()
    plt.close(fig)

    sys.exit(0)
