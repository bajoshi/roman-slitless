import numpy as np
from astropy.io import fits

import os
import sys

import matplotlib.pyplot as plt

extdir = '/Volumes/Joshi_external_HDD/Roman/roman_slitless_sims_results/'
pylinear_lst_dir = '/Volumes/Joshi_external_HDD/Roman/pylinear_lst_files/'
roman_direct_dir = '/Volumes/Joshi_external_HDD/Roman/roman_direct_sims/sims2021/K_5degimages_part1/'

comparefile = extdir + 'AB_25_18000s.txt'

home = os.getenv("HOME")
roman_slitless_dir = home + "/Documents/GitHub/roman-slitless/"
fitting_utils = roman_slitless_dir + "fitting_pipeline/utils/"

sys.path.append(fitting_utils)
from get_snr import get_snr
from gen_sed_lst import get_sn_z

cat_header = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'ALPHA_J2000', 'DELTA_J2000', 
'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_AUTO', 'MAGERR_AUTO', 'FLUX_RADIUS', 'FWHM_IMAGE']
pylinear_flam_scale_fac = 1e-17

def get_all_sn_segids(sedlst):

    # --------------- loop and find all SN segids
    all_sn_segids = []
    for i in range(len(sedlst)):
        if 'salt' in sedlst['sed_path'][i]:
            all_sn_segids.append(sedlst['segid'][i])

    return all_sn_segids

if __name__ == '__main__':

    refspec = np.genfromtxt(comparefile, dtype=None, names=True, encoding='ascii')
    print(refspec.dtype.names)

    # --------------- Read in pylinear x1d spectra 
    ext1 = fits.open(extdir + 'romansim_prism_Y106_0_1_6000s_x1d.fits')
    ext2 = fits.open(extdir + 'romansim_prism_Y106_0_2_6000s_x1d.fits')

    # Read in SED lsts and collect all SN spectra in one fits file

    # --------------- Read in sed.lst
    sedlst_header = ['segid', 'sed_path']

    sedlst_path1 = pylinear_lst_dir + 'sed_Y106_0_1.lst'
    sedlst1 = np.genfromtxt(sedlst_path1, dtype=None, names=sedlst_header, encoding='ascii')

    sedlst_path2 = pylinear_lst_dir + 'sed_Y106_0_2.lst'
    sedlst2 = np.genfromtxt(sedlst_path2, dtype=None, names=sedlst_header, encoding='ascii')

    all_sn_segids1 = get_all_sn_segids(sedlst1)
    all_sn_segids2 = get_all_sn_segids(sedlst2)

    # --------------- Now read in both catalogs
    cat_filename1 = roman_direct_dir + '5deg_Y106_0_1_SNadded.cat' 
    cat1 = np.genfromtxt(cat_filename1, dtype=None, names=cat_header, encoding='ascii')

    cat_filename2 = roman_direct_dir + '5deg_Y106_0_2_SNadded.cat' 
    cat2 = np.genfromtxt(cat_filename2, dtype=None, names=cat_header, encoding='ascii')

    # --------------- prep for looping
    allcats = [cat1, cat2]
    all_sn = [all_sn_segids1, all_sn_segids2]
    all_ext = [ext1, ext2]

    phdu = fits.PrimaryHDU()
    hdul = fits.HDUList(phdu)

    speccount = 0

    # Loop over all images
    for i in range(2):

        sn_segids = all_sn[i]
        cat = allcats[i]
        ext_hdu = all_ext[i]

       # Looop over all sn segids in file
        for segid in sn_segids:

            # get mag
            segid_idx = int(np.where(cat['NUMBER'] == segid)[0])
            mag = cat['MAG_AUTO'][segid_idx]

            if mag < 23.0:
                continue

            # Now get spectrum and uncertainty
            wav = ext_hdu[('SOURCE', segid)].data['wavelength']
            flam = ext_hdu[('SOURCE', segid)].data['flam'] * pylinear_flam_scale_fac

            ferr_lo = ext_hdu[('SOURCE', segid)].data['flounc'] * pylinear_flam_scale_fac
            ferr_hi = ext_hdu[('SOURCE', segid)].data['fhiunc'] * pylinear_flam_scale_fac

            ferr = (ferr_lo + ferr_hi) / 2

            # Now get avg SNR and SNR per pix
            derived_snr = get_snr(wav, flam)
            snr_pix = flam/ferr

            # Get redshift
            redshift = get_sn_z(mag)

            print(i, segid, mag, redshift, '{:.1f}'.format(np.nanmean(snr_pix)), '{:.1f}'.format(derived_snr))

            speccount += 1
            #print(snr_pix)

            #fig = plt.figure()
            #ax = fig.add_subplot(111)
            #ax.plot(wav, flam, color='k')
            #ax.fill_between(wav, flam-ferr_lo, flam+ferr_hi, color='gray', alpha=0.4)
            #plt.show()

            # Save to fits extension
            col1 = fits.Column(name='Wave[A]', format='E', array=wav)
            col2 = fits.Column(name='S/N', format='E', array=snr_pix)
            col3 = fits.Column(name='signal', format='E', array=flam)
            col4 = fits.Column(name='noise', format='E', array=ferr)

            hdu = fits.BinTableHDU.from_columns([col1, col2, col3, col4])
            hdu.header['magF106'] = mag
            hdu.header['z'] = redshift

            hdul.append(hdu)

    # Save HDUlist
    hdul.writeto(extdir + 'snspectra_pylinear.fits', overwrite=True)

    print('Total spectra in file:', speccount)

    sys.exit(0)











