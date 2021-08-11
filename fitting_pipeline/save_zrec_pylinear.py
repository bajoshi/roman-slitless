import numpy as np
import emcee
import corner
from astropy.io import fits

import os
import sys
import glob

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


# ---------------------------------------
#exptime1 = '_900s'
#exptime2 = '_1800s'
#exptime3 = '_3600s'

exptime1 = '_1500s'
exptime2 = '_6000s'

img_filt = 'Y106_'
ext_root = 'romansim_prism_'

# Header for the results file
"""
res_hdr = '#  img_suffix  SNSegID  z_true  phase_true  Av_true  ' + \
          'Y106mag  SNR900  SNR1800  SNR3600  ' + \
          'z900  z900_lowerr  z900_uperr  ' + \
          'phase900  phase900_lowerr  phase900_uperr  ' + \
          'Av900  Av900_lowerr  Av900_uperr  ' + \
          'z1800  z1800_lowerr  z1800_uperr  ' + \
          'phase1800  phase1800_lowerr  phase1800_uperr  ' + \
          'Av1800  Av1800_lowerr  Av1800_uperr  ' + \
          'z3600  z3600_lowerr  z3600_uperr  ' + \
          'phase3600  phase3600_lowerr  phase3600_uperr  ' + \
          'Av3600  Av3600_lowerr  Av3600_uperr'
"""
res_hdr = '#  img_suffix  SNSegID  z_true  phase_true  Av_true  ' + \
          'Y106mag  SNR900  SNR3600  ' + \
          'z900  z900_lowerr  z900_uperr  ' + \
          'phase900  phase900_lowerr  phase900_uperr  ' + \
          'Av900  Av900_lowerr  Av900_uperr  ' + \
          'z3600  z3600_lowerr  z3600_uperr  ' + \
          'phase3600  phase3600_lowerr  phase3600_uperr  ' + \
          'Av3600  Av3600_lowerr  Av3600_uperr'

# Header for SExtractor catalog
cat_header = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'ALPHA_J2000', 'DELTA_J2000', 
'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_AUTO', 'MAGERR_AUTO', 'FLUX_RADIUS', 'FWHM_IMAGE']

# Arrays to loop over
pointings = np.arange(0, 1)
detectors = np.arange(1, 3)

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

            #ext_spec_filename3 = ext_spectra_dir + ext_root + img_suffix + exptime3 + '_x1d.fits'
            #ext_hdu3 = fits.open(ext_spec_filename3)

            # ----- Read in catalog from SExtractor
            cat_filename = img_sim_dir + '5deg_' + img_suffix + '_SNadded.cat'
            cat = np.genfromtxt(cat_filename, dtype=None, names=cat_header, encoding='ascii')

            # ----- loop and find all SN segids
            all_sn_segids = []
            for i in range(len(sedlst)):
                if 'salt' in sedlst['sed_path'][i]:
                    all_sn_segids.append(sedlst['segid'][i])

            print('ALL SN segids in this file:', all_sn_segids)
            print(len(all_sn_segids), 'SN in', img_suffix + '\n')

            # ----- Now loop over all segids in this img
            for segid in all_sn_segids:

                segid_idx = int(np.where(sedlst['segid'] == segid)[0])

                template_name = os.path.basename(sedlst['sed_path'][segid_idx])

                template_inputs = get_template_inputs(template_name)
                true_z     = template_inputs[0]
                true_phase = template_inputs[1]
                true_av    = template_inputs[2]

                # ----- Get SNR
                wav1 = ext_hdu1[('SOURCE', segid)].data['wavelength']
                flam1 = ext_hdu1[('SOURCE', segid)].data['flam'] * pylinear_flam_scale_fac
                wav2 = ext_hdu2[('SOURCE', segid)].data['wavelength']
                flam2 = ext_hdu2[('SOURCE', segid)].data['flam'] * pylinear_flam_scale_fac
                #wav3 = ext_hdu3[('SOURCE', segid)].data['wavelength']
                #flam3 = ext_hdu3[('SOURCE', segid)].data['flam'] * pylinear_flam_scale_fac

                snr1 = get_snr(wav1, flam1)
                snr2 = get_snr(wav2, flam2)
                #snr3 = get_snr(wav3, flam3)

                # ----- Get magnitude in Y106
                mag_idx = int(np.where(cat['NUMBER'] == segid)[0])
                mag = cat['MAG_AUTO'][mag_idx]

                # ----- Write to file
                # --- ID and true quantities
                fh.write(img_suffix + '  ' + str(segid) + '  ')
                fh.write('{:.3f}'.format(true_z) + '  ' + str(true_phase) + '  ')
                fh.write('{:.3f}'.format(true_av) + '  ')
                fh.write('{:.2f}'.format(mag)  + '  ')
                fh.write('{:.2f}'.format(snr1) + '  ')
                fh.write('{:.2f}'.format(snr2) + '  ')
                #fh.write('{:.2f}'.format(snr3) + '  ')

                # ----- Construct the filenames for this segid
                snstr1 = str(segid) + '_' + img_suffix + exptime1
                emcee_savefile1 = results_dir + \
                                  'emcee_sampler_sn' + snstr1 + '.h5'

                snstr2 = str(segid) + '_' + img_suffix + exptime2
                emcee_savefile2 = results_dir + \
                                  'emcee_sampler_sn' + snstr2 + '.h5'

                #snstr3 = str(segid) + '_' + img_suffix + exptime3
                #emcee_savefile3 = results_dir + \
                #                  'emcee_sampler_sn' + snstr3 + '.h5'

                # Make sure the sampler file exists
                if os.path.isfile(emcee_savefile1):

                    # ----- Get flat samples
                    sampler1 = emcee.backends.HDFBackend(emcee_savefile1)
                    burn_in1, thinning_steps1 = get_burn_thin(sampler1)
                    flat_samples1 = sampler1.get_chain(discard=burn_in1, thin=thinning_steps1, flat=True)

                    # ----- Read in corner quantiles
                    cq_z1 = corner.quantile(x=flat_samples1[:, 0], q=[0.16, 0.5, 0.84])
                    cq_day1 = corner.quantile(x=flat_samples1[:, 1], q=[0.16, 0.5, 0.84])
                    cq_av1 = corner.quantile(x=flat_samples1[:, 2], q=[0.16, 0.5, 0.84])

                    # --- EXPTIME 900 seconds  # wide survey
                    z900_lowerr = cq_z1[1] - cq_z1[0]
                    z900_uperr = cq_z1[2] - cq_z1[1]

                    phase900_lowerr = cq_day1[1] - cq_day1[0]
                    phase900_uperr = cq_day1[2] - cq_day1[1]

                    av900_lowerr = cq_av1[1] - cq_av1[0]
                    av900_uperr = cq_av1[2] - cq_av1[1]

                    fh.write('{:.3f}'.format(cq_z1[1]) + '  ')
                    fh.write('{:.3f}'.format(z900_lowerr) + '  ')
                    fh.write('{:.3f}'.format(z900_uperr) + '  ')

                    fh.write('{:.1f}'.format(cq_day1[1]) + '  ')
                    fh.write('{:.1f}'.format(phase900_lowerr) + '  ')
                    fh.write('{:.1f}'.format(phase900_uperr) + '  ')

                    fh.write('{:.3f}'.format(cq_av1[1]) + '  ')
                    fh.write('{:.3f}'.format(av900_lowerr) + '  ')
                    fh.write('{:.3f}'.format(av900_uperr) + '  ')

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


                if os.path.isfile(emcee_savefile2):

                    # ----- Get flat samples
                    sampler2 = emcee.backends.HDFBackend(emcee_savefile2)
                    burn_in2, thinning_steps2 = get_burn_thin(sampler2)
                    flat_samples2 = sampler2.get_chain(discard=burn_in2, thin=thinning_steps2, flat=True)

                    # ----- Read in corner quantiles
                    cq_z2 = corner.quantile(x=flat_samples2[:, 0], q=[0.16, 0.5, 0.84])
                    cq_day2 = corner.quantile(x=flat_samples2[:, 1], q=[0.16, 0.5, 0.84])
                    cq_av2 = corner.quantile(x=flat_samples2[:, 2], q=[0.16, 0.5, 0.84])

                    # --- EXPTIME 1800 seconds
                    z1800_lowerr = cq_z2[1] - cq_z2[0]
                    z1800_uperr = cq_z2[2] - cq_z2[1]

                    phase1800_lowerr = cq_day2[1] - cq_day2[0]
                    phase1800_uperr = cq_day2[2] - cq_day2[1]

                    av1800_lowerr = cq_av2[1] - cq_av2[0]
                    av1800_uperr = cq_av2[2] - cq_av2[1]

                    fh.write('{:.3f}'.format(cq_z2[1]) + '  ')
                    fh.write('{:.3f}'.format(z1800_lowerr) + '  ')
                    fh.write('{:.3f}'.format(z1800_uperr) + '  ')

                    fh.write('{:.1f}'.format(cq_day2[1]) + '  ')
                    fh.write('{:.1f}'.format(phase1800_lowerr) + '  ')
                    fh.write('{:.1f}'.format(phase1800_uperr) + '  ')

                    fh.write('{:.3f}'.format(cq_av2[1]) + '  ')
                    fh.write('{:.3f}'.format(av1800_lowerr) + '  ')
                    fh.write('{:.3f}'.format(av1800_uperr) + '\n')

                else:
                    fh.write('-9999.0' + '  ')
                    fh.write('-9999.0' + '  ')
                    fh.write('-9999.0' + '  ')

                    fh.write('-9999.0' + '  ')
                    fh.write('-9999.0' + '  ')
                    fh.write('-9999.0' + '  ')

                    fh.write('-9999.0' + '  ')
                    fh.write('-9999.0' + '  ')
                    fh.write('-9999.0' + '\n')

                """
                if os.path.isfile(emcee_savefile2):

                    # ----- Get flat samples
                    sampler3 = emcee.backends.HDFBackend(emcee_savefile3)
                    burn_in3, thinning_steps3 = get_burn_thin(sampler3)
                    flat_samples3 = sampler3.get_chain(discard=burn_in3, thin=thinning_steps3, flat=True)

                    # ----- Read in corner quantiles
                    cq_z3 = corner.quantile(x=flat_samples3[:, 0], q=[0.16, 0.5, 0.84])
                    cq_day3 = corner.quantile(x=flat_samples3[:, 1], q=[0.16, 0.5, 0.84])
                    cq_av3 = corner.quantile(x=flat_samples3[:, 2], q=[0.16, 0.5, 0.84])

                    # --- EXPTIME 3600 seconds
                    z3600_lowerr = cq_z3[1] - cq_z3[0]
                    z3600_uperr = cq_z3[2] - cq_z3[1]

                    phase3600_lowerr = cq_day3[1] - cq_day3[0]
                    phase3600_uperr = cq_day3[2] - cq_day3[1]

                    av3600_lowerr = cq_av3[1] - cq_av3[0]
                    av3600_uperr = cq_av3[2] - cq_av3[1]

                    fh.write('{:.3f}'.format(cq_z3[1]) + '  ')
                    fh.write('{:.3f}'.format(z3600_lowerr) + '  ')
                    fh.write('{:.3f}'.format(z3600_uperr) + '  ')

                    fh.write('{:.1f}'.format(cq_day3[1]) + '  ')
                    fh.write('{:.1f}'.format(phase3600_lowerr) + '  ')
                    fh.write('{:.1f}'.format(phase3600_uperr) + '  ')

                    fh.write('{:.3f}'.format(cq_av3[1]) + '  ')
                    fh.write('{:.3f}'.format(av3600_lowerr) + '  ')
                    fh.write('{:.3f}'.format(av3600_uperr) + '\n')

                else:
                    fh.write('-9999.0' + '  ')
                    fh.write('-9999.0' + '  ')
                    fh.write('-9999.0' + '  ')

                    fh.write('-9999.0' + '  ')
                    fh.write('-9999.0' + '  ')
                    fh.write('-9999.0' + '  ')

                    fh.write('-9999.0' + '  ')
                    fh.write('-9999.0' + '  ')
                    fh.write('-9999.0' + '\n')
                """

            ext_hdu1.close()
            ext_hdu2.close()
            #ext_hdu3.close()





