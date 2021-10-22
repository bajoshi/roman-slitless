import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np

from tqdm import tqdm

from get_template_inputs import get_template_inputs
from get_snr import get_snr

def plot_z_mag_snr(mag, snr, redshift):

    # Make fig and choose cmap
    fig = plt.figure(figsize=(9,5))
    ax = fig.add_subplot(111)

    ax.set_xlabel('Y106 magnitude', fontsize=16)
    ax.set_ylabel('1-hour SNR', fontsize=16)

    cmap = plt.cm.get_cmap('viridis_r')

    # Plot
    cax = ax.scatter(mag, snr, s=20, c=redshift, cmap=cmap, facecolors='None')

    # Colorbar and label
    cbar = fig.colorbar(cax)
    cbar.set_label('Redshift', fontsize=16)

    # Add horizontal line at SNR=5.0
    ax.axhline(y=5.0, ls='--', color='k', lw=2.0)

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
    fig = plt.figure(figsize=(9,5))
    ax = fig.add_subplot(111)

    ax.set_xlabel('Y106 magnitude', fontsize=16)
    ax.set_ylabel('log(counts)', fontsize=16)

    cmap = plt.cm.get_cmap('viridis_r')

    cax = ax.scatter(cat['Y106mag'], np.log10(counts), s=20, c=cat['z_true'], cmap=cmap, facecolors='None')

    cbar = fig.colorbar(cax)
    cbar.set_label('Redshift', fontsize=16)

    plt.show()
    fig.clear()
    plt.close(fig)

    return None

    


if __name__ == '__main__':

    results_dir = '/Volumes/Joshi_external_HDD/Roman/roman_slitless_sims_results/'
    pylinear_lst_dir = '/Volumes/Joshi_external_HDD/Roman/pylinear_lst_files/'
    img_sim_dir = '/Volumes/Joshi_external_HDD/Roman/roman_direct_sims/sims2021/K_5degimages_part1/'

    img_suffix = 'Y106_0_1'
    resfile = 'romansim_prism_' + img_suffix + '_1200s_x1d.fits'
    
    # --------------- Read in extracted spectra
    xhdu = fits.open(results_dir + resfile)

    # --------------- Get all SNe IDs
    # Read in sed.lst
    sedlst_header = ['segid', 'sed_path']
    sedlst_path = pylinear_lst_dir + 'sed_' + img_suffix + '.lst'
    sedlst = np.genfromtxt(sedlst_path, dtype=None, names=sedlst_header, encoding='ascii')

    # loop and find all SN segids
    all_sn_segids = []
    for i in range(len(sedlst)):
        if 'salt' in sedlst['sed_path'][i]:
            all_sn_segids.append(sedlst['segid'][i])

    print('ALL SN segids in this file:', all_sn_segids)

    # --------------- Also read in SExtractor catalog for mags
    cat_filename = img_sim_dir + '5deg_' + img_suffix + '_SNadded.cat'
    cat_header = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'ALPHA_J2000', 'DELTA_J2000', 
    'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_AUTO', 'MAGERR_AUTO', 'FLUX_RADIUS', 'FWHM_IMAGE']
    cat = np.genfromtxt(cat_filename, dtype=None, names=cat_header, encoding='ascii')

    # --------------- Collect needed arrays
    mag = []
    snr = []
    redshift = []

    for i in tqdm(range(len(all_sn_segids)), desc='Processing SN'):

        current_segid = all_sn_segids[i]
        segid_idx = int(np.where(cat['NUMBER'] == current_segid)[0])

        # Get magnitude
        mag.append(cat['MAG_AUTO'][segid_idx])

        # Get spectrum and SNR
        wav = xhdu[('SOURCE', current_segid)].data['wavelength']
        flam = xhdu[('SOURCE', current_segid)].data['flam'] * 1e-17

        snr.append(float(get_snr(wav, flam)))

        # Get redshift
        sed_idx = int(np.where(sedlst['segid'] == current_segid)[0])
        template_name = sedlst['sed_path'][sed_idx]
        inp = get_template_inputs(template_name)
        redshift.append(inp[0])

    # --------------- Make plot
    plot_z_mag_snr(mag, snr, redshift)

    # Close HDU
    xhdu.close()

















