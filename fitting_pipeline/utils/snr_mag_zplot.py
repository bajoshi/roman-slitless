import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np

def plot_z_mag_snr(cat):
    fig = plt.figure(figsize=(9,5))
    ax = fig.add_subplot(111)

    ax.set_xlabel('Y106 magnitude', fontsize=16)
    ax.set_ylabel('1-hour SNR', fontsize=16)

    cmap = plt.cm.get_cmap('viridis_r')

    cax = ax.scatter(cat['Y106mag'], cat['SNR1200'], s=20, c=cat['z_true'], cmap=cmap, facecolors='None')

    cbar = fig.colorbar(cax)
    cbar.set_label('Redshift', fontsize=16)

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

    