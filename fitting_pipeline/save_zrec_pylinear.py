import numpy as np
import emcee
import corner
from astropy.io import fits
from astropy.convolution import convolve, Box1DKernel

import os
import sys
import glob
from tqdm import tqdm

ext_spectra_dir = "/Volumes/Joshi_external_HDD/Roman/roman_slitless_sims_results/"
results_dir = ext_spectra_dir + 'fitting_results/'
img_sim_dir = "/Volumes/Joshi_external_HDD/Roman/roman_direct_sims/sims2021/K_5degimages_part1/"

# Assign directories and custom imports
cwd = os.path.dirname(os.path.abspath(__file__))
fitting_utils = cwd + '/utils/'

roman_slitless_dir = os.path.dirname(cwd)

sys.path.append(fitting_utils)
from get_snr import get_snr
from get_template_inputs import get_template_inputs

# Set pylinear f_lambda scaling factor
pylinear_flam_scale_fac = 1e-17

def get_burn_thin(sampler):

    # Get autocorrelation time
    # Discard burn-in. You do not want to consider the burn in the corner plots/estimation.
    tau = sampler.get_autocorr_time(tol=0)

    if not np.any(np.isnan(tau)):
        burn_in = int(2 * np.max(tau))
        thinning_steps = int(0.5 * np.min(tau))
    else:
        burn_in = 200
        thinning_steps = 30

    return burn_in, thinning_steps

def get_correct_snr(ext_hdu, segid):

    wav  = ext_hdu[('SOURCE', segid)].data['wavelength']
    flam = ext_hdu[('SOURCE', segid)].data['flam'] * pylinear_flam_scale_fac

    return get_snr(wav, flam)

def check_and_write(fh, emcee_savefile):

    if os.path.isfile(emcee_savefile):
        write_to_file_data(fh, emcee_savefile)
    else:
        write_to_file_blank(fh)

    return None

def write_to_file_data(fh, emcee_savefile):

    # ----- Get flat samples
    sampler = emcee.backends.HDFBackend(emcee_savefile)
    burn_in, thinning_steps = get_burn_thin(sampler)
    flat_samples = sampler.get_chain(discard=burn_in, thin=thinning_steps, flat=True)

    # ----- Read in corner quantiles
    cq_z = corner.quantile(x=flat_samples[:, 0], q=[0.16, 0.5, 0.84])
    cq_day = corner.quantile(x=flat_samples[:, 1], q=[0.16, 0.5, 0.84])
    cq_av = corner.quantile(x=flat_samples[:, 2], q=[0.16, 0.5, 0.84])

    # -----
    z_lowerr = cq_z[1] - cq_z[0]
    z_uperr = cq_z[2] - cq_z[1]

    phase_lowerr = cq_day[1] - cq_day[0]
    phase_uperr = cq_day[2] - cq_day[1]

    av_lowerr = cq_av[1] - cq_av[0]
    av_uperr = cq_av[2] - cq_av[1]

    fh.write('{:.3f}'.format(cq_z[1]) + '  ')
    fh.write('{:.3f}'.format(z_lowerr) + '  ')
    fh.write('{:.3f}'.format(z_uperr) + '  ')

    fh.write('{:.1f}'.format(cq_day[1]) + '  ')
    fh.write('{:.1f}'.format(phase_lowerr) + '  ')
    fh.write('{:.1f}'.format(phase_uperr) + '  ')

    fh.write('{:.3f}'.format(cq_av[1]) + '  ')
    fh.write('{:.3f}'.format(av_lowerr) + '  ')
    fh.write('{:.3f}'.format(av_uperr) + '  ')

    return None

def write_to_file_blank(fh):

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

def get_faint_sn_mag(segid, segdata, dir_img):

    zeropoint = 26.264

    obj_idx = np.where(segdata == segid)
    all_counts = dir_img[obj_idx]

    total_counts = np.sum(all_counts)

    faint_mag = -2.5 * np.log10(total_counts) + zeropoint

    return faint_mag

# ---------------------------------------
exptime1 = '_400s'
exptime2 = '_1200s'
exptime3 = '_3600s'
exptime4 = '_10800s'

img_filt = 'Y106_'
ext_root = 'romansim_prism_'
res_hdr = ( '#  img_suffix  SNSegID  z_true  phase_true  Av_true  ' + 
            'Y106mag  SNR300  SNR1200  SNR3600  SNR6000  ' + 
            'z400  z400_lowerr  z400_uperr  ' + 
            'phase400  phase400_lowerr  phase400_uperr  ' + 
            'Av400  Av400_lowerr  Av400_uperr  ' + 

            'z1200  z1200_lowerr  z1200_uperr  ' + 
            'phase1200  phase1200_lowerr  phase1200_uperr  ' + 
            'Av1200  Av1200_lowerr  Av1200_uperr  ' + 
            
            'z3600  z3600_lowerr  z3600_uperr  ' + 
            'phase3600  phase3600_lowerr  phase3600_uperr  ' + 
            'Av3600  Av3600_lowerr  Av3600_uperr  ' + 
            
            'z10800  z10800_lowerr  z10800_uperr  ' + 
            'phase10800  phase10800_lowerr  phase10800_uperr  ' + 
            'Av10800  Av10800_lowerr  Av10800_uperr'
          )

# Header for SExtractor catalog
cat_header = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'ALPHA_J2000', 'DELTA_J2000', 
'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_AUTO', 'MAGERR_AUTO', 'FLUX_RADIUS', 'FWHM_IMAGE']

# Arrays to loop over
pointings = np.arange(0, 1)
detectors = np.arange(1, 2)

for pt in pointings:

    # Save results to text file
    resfile = results_dir + 'zrecovery_pylinear_sims_pt' + str(pt) + '.txt'

    with open(resfile, 'w') as fh:
        fh.write(res_hdr + '\n')

        for det in detectors:

            # ----- Get img suffix, segid, and truth values
            img_suffix = img_filt + str(pt) + '_' + str(det)

            # ----- Read in sed.lst
            sedlst_header = ['segid', 'sed_path']
            sedlst_path = '/Volumes/Joshi_external_HDD/Roman/pylinear_lst_files/' + 'sed_' + img_suffix + '.lst'
            sedlst = np.genfromtxt(sedlst_path, dtype=None, names=sedlst_header, encoding='ascii')

            # ----- Read in x1d file to get spectra for SNR
            ext_spec_filename1 = ext_spectra_dir + ext_root + img_suffix + exptime1 + '_x1d.fits'
            ext_hdu1 = fits.open(ext_spec_filename1)

            ext_spec_filename2 = ext_spectra_dir + ext_root + img_suffix + exptime2 + '_x1d.fits'
            ext_hdu2 = fits.open(ext_spec_filename2)

            ext_spec_filename3 = ext_spectra_dir + ext_root + img_suffix + exptime3 + '_x1d.fits'
            ext_hdu3 = fits.open(ext_spec_filename3)

            ext_spec_filename4 = ext_spectra_dir + ext_root + img_suffix + exptime4 + '_x1d.fits'
            ext_hdu4 = fits.open(ext_spec_filename4)

            # ----- Read in catalog from SExtractor
            cat_filename = img_sim_dir + '5deg_' + img_suffix + '_SNadded.cat'
            cat = np.genfromtxt(cat_filename, dtype=None, names=cat_header, encoding='ascii')

            # -----Read in segmentation map
            segmap = img_sim_dir + '5deg_' + img_suffix + '_segmap.fits'
            segdata = fits.getdata(segmap)

            # ----- Name of direct image
            dir_img_name = segmap.replace('_segmap.fits', '_SNadded.fits')
            dir_img = fits.getdata(dir_img_name)

            # ----- loop and find all SN segids
            all_sn_segids = []
            for i in range(len(sedlst)):
                if 'salt' in sedlst['sed_path'][i]:
                    all_sn_segids.append(sedlst['segid'][i])

            print('ALL SN segids in this file:', all_sn_segids)
            print(len(all_sn_segids), 'SN in', img_suffix + '\n')

            # ----- Now loop over all segids in this img
            for segid in tqdm(all_sn_segids, desc='Processing SN'):

                segid_idx = int(np.where(sedlst['segid'] == segid)[0])

                template_name = os.path.basename(sedlst['sed_path'][segid_idx])

                template_inputs = get_template_inputs(template_name)
                true_z     = template_inputs[0]
                true_phase = template_inputs[1]
                true_av    = template_inputs[2]

                # ----- Get SNR
                snr1 = get_correct_snr(ext_hdu1, segid)
                snr2 = get_correct_snr(ext_hdu2, segid)
                snr3 = get_correct_snr(ext_hdu3, segid)
                snr4 = get_correct_snr(ext_hdu4, segid)

                # ----- Get magnitude in Y106
                mag_idx = np.where(cat['NUMBER'] == segid)[0]
                if mag_idx.size:
                    mag_idx = int(mag_idx)
                    mag = cat['MAG_AUTO'][mag_idx]
                else:
                    mag = get_faint_sn_mag(segid, segdata, dir_img)

                # ----- Write to file
                # --- ID and true quantities
                fh.write(img_suffix + '  ' + str(segid) + '  ')
                fh.write('{:.3f}'.format(true_z) + '  ' + str(true_phase) + '  ')
                fh.write('{:.3f}'.format(true_av) + '  ')
                fh.write('{:.2f}'.format(mag)  + '  ')
                fh.write('{:.2f}'.format(snr1) + '  ')
                fh.write('{:.2f}'.format(snr2) + '  ')
                fh.write('{:.2f}'.format(snr3) + '  ')
                fh.write('{:.2f}'.format(snr4) + '  ')

                # ----- Construct the filenames for this segid
                snstr1 = str(segid) + '_' + img_suffix + exptime1
                emcee_savefile1 = results_dir + 'emcee_sampler_sn' + snstr1 + '.h5'

                snstr2 = str(segid) + '_' + img_suffix + exptime2
                emcee_savefile2 = results_dir + 'emcee_sampler_sn' + snstr2 + '.h5'

                snstr3 = str(segid) + '_' + img_suffix + exptime3
                emcee_savefile3 = results_dir + 'emcee_sampler_sn' + snstr3 + '.h5'

                snstr4 = str(segid) + '_' + img_suffix + exptime4
                emcee_savefile4 = results_dir + 'emcee_sampler_sn' + snstr4 + '.h5'

                # ----------------
                check_and_write(fh, emcee_savefile1)
                check_and_write(fh, emcee_savefile2)
                check_and_write(fh, emcee_savefile3)
                check_and_write(fh, emcee_savefile4)
                fh.write('\n')

            ext_hdu1.close()
            ext_hdu2.close()
            ext_hdu3.close()
            ext_hdu4.close()







