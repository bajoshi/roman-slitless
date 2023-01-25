import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np

from tqdm import tqdm

from get_template_inputs import get_template_inputs
from get_snr import get_snr

import os
import sys
home = os.getenv('HOME')
roman_slitless_dir = home + '/Documents/GitHub/roman-slitless/'


def plot_z_mag_snr(mag, snr, redshift, savepath=None):

    # Make fig and choose cmap
    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(111)

    ax.set_xlabel('F106 magnitude', fontsize=16)
    ax.set_ylabel('SNR of extracted 1d spectrum', fontsize=16)

    cmaps = [plt.cm.get_cmap('Oranges_r'),
             plt.cm.get_cmap('Purples_r'),
             plt.cm.get_cmap('Blues_r'),
             plt.cm.get_cmap('Greens_r'),
             plt.cm.get_cmap('Reds_r')]

    # Plot
    for i in range(mag.shape[0]):
        cax = ax.scatter(mag[i], snr[i], s=12, c=redshift[i],
                         cmap=cmaps[i], facecolors='None')

    # Colorbar and label
    cbar = fig.colorbar(cax)
    cbar.set_label('Redshift', fontsize=16)

    # Add horizontal line at SNR=3.0
    ax.axhline(y=3.0, ls='--', color='k', lw=2.0)

    # limits
    ax.set_ylim(-10.0, 50.0)

    # Save figure if needed
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches='tight')
    else:
        plt.show()
        fig.clear()
        plt.close(fig)

    return None


def get_counts(segid, dirdat, segdat):

    obj_pix = np.where(segdat == segid)
    counts = np.sum(dirdat[obj_pix])

    return counts


def plot_z_mag_counts(cat, segmap_path, dirimg_path):

    # Open segmap and direct image
    dhdu = fits.open(dirimg_path)
    shdu = fits.open(segmap_path)

    dirdat = dhdu[0].data
    segdat = shdu[0].data

    # Get counts for each object in catalog
    counts = np.zeros(len(cat))
    for i in range(len(cat)):
        segid = cat['SNSegID'][i]
        counts[i] = get_counts(segid, dirdat, segdat)

    # Close hdus
    dhdu.close()
    shdu.close()

    # Plot
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)

    ax.set_xlabel('Y106 magnitude', fontsize=16)
    ax.set_ylabel('log(counts)', fontsize=16)

    cmap = plt.cm.get_cmap('viridis_r')

    cax = ax.scatter(cat['Y106mag'], np.log10(counts), s=20, c=cat['z_true'],
                     cmap=cmap, facecolors='None')

    cbar = fig.colorbar(cax)
    cbar.set_label('Redshift', fontsize=16)

    plt.show()
    fig.clear()
    plt.close(fig)

    return None


if __name__ == '__main__':

    extdir = '/Volumes/Joshi_external_HDD/Roman/'

    results_dir = extdir + 'roman_slitless_sims_results/'
    pylinear_lst_dir = extdir + 'pylinear_lst_files/'
    img_sim_dir = extdir + 'roman_direct_sims/sims2021/K_5degimages_part1/'

    img_suffix = 'Y106_0_1'
    exptime = ['_400s', '_1200s', '_3600s', '_10800s']

    # --------------- Get all SNe IDs
    sedlst_header = ['segid', 'sed_path']
    sedlst_path = pylinear_lst_dir + 'sed_' + img_suffix + '.lst'
    sedlst = np.genfromtxt(sedlst_path, dtype=None,
                           names=sedlst_header, encoding='ascii')
    # Counting only the uncontaminated spectra for now
    all_sn_segids = []
    for i in range(len(sedlst)):
        if ('salt' in sedlst['sed_path'][i]):
            all_sn_segids.append(sedlst['segid'][i])

    # --------------- Also read in inserted object catalog for mags
    cat_filename = img_sim_dir + '5deg_' + img_suffix + '_SNadded.npy'
    insert_cat = np.load(cat_filename)

    all_inserted_segids = np.array(insert_cat[:, -1], dtype=np.int64)

    # --------------- Collect needed arrays
    mag = np.zeros((len(exptime), len(all_sn_segids)))
    snr = np.zeros((len(exptime), len(all_sn_segids)))
    redshift = np.zeros((len(exptime), len(all_sn_segids)))

    for e in tqdm(range(len(exptime)), desc='Exptime', leave=False):

        resfile = 'romansim_prism_' + img_suffix + exptime[e] + '_x1d.fits'

        # --------------- Read in extracted spectra
        xhdu = fits.open(results_dir + resfile)

        for i in tqdm(range(len(all_sn_segids)), desc='Processing SN',
                      leave=False):

            current_segid = all_sn_segids[i]
            segid_idx = int(np.where(all_inserted_segids == current_segid)[0])

            # Get magnitude
            mag[e, i] = float(insert_cat[segid_idx, 2])

            # Get spectrum and SNR
            wav = xhdu[('SOURCE', current_segid)].data['wavelength']
            flam = xhdu[('SOURCE', current_segid)].data['flam'] * 1e-17

            snr[e, i] = float(get_snr(wav, flam))

            # Get redshift
            sed_idx = int(np.where(sedlst['segid'] == current_segid)[0])
            template_name = sedlst['sed_path'][sed_idx]
            inp = get_template_inputs(template_name)
            redshift[e, i] = inp[0]

    # --------------- Make plot and save
    savepath = roman_slitless_dir + 'figures/extracted_snr.pdf'

    plot_z_mag_snr(mag, snr, redshift, savepath)

    # --------------- Make a separate plot for the 1 hr and 20m exptimes
    fig, (ax1, ax2) = plt.subplots(figsize=(13, 5), nrows=1, ncols=2)

    ax1.set_xlabel('F106 magnitude', fontsize=16)
    ax2.set_xlabel('F106 magnitude', fontsize=16)

    ax1.set_ylabel('20 min SNR of extracted 1d spectrum', fontsize=16)
    ax2.set_ylabel('1 hr SNR of extracted 1d spectrum', fontsize=16)

    ax1.scatter(mag[0], snr[0], s=12, c='k', facecolors='None')
    ax2.scatter(mag[1], snr[1], s=12, c='k', facecolors='None')

    ax1.axhline(y=3.0, ls='--', color='k', lw=2.0)
    ax2.axhline(y=3.0, ls='--', color='k', lw=2.0)

    # Fixed y limits on both to be able to compare
    ax1.set_ylim(0, 13)
    ax2.set_ylim(0, 13)

    fig.savefig(savepath.replace('.pdf', '_short_exptimes.pdf'), dpi=200,
                bbox_inches='tight')

    # Close HDU
    xhdu.close()

    sys.exit(0)
