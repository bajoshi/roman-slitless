import numpy as np
import emcee
import corner

import os
import sys
import glob

ext_spectra_dir = "/Volumes/Joshi_external_HDD/Roman/roman_slitless_sims_results/"
results_dir = ext_spectra_dir + 'fitting_results/'

# Assign directories and custom imports
cwd = os.path.dirname(os.path.abspath(__file__))
fitting_utils = cwd + '/utils/'

roman_slitless_dir = os.path.dirname(cwd)

sys.path.append(fitting_utils)
from get_snr import get_snr
from get_template_inputs import get_template_inputs

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
exptime1 = '_900s'
exptime2 = '_1800s'
exptime3 = '_3600s'

img_filt = 'Y106_'

# Save results to text file
resfile = results_dir + 'zrecovery_pylinear_sims.txt'
res_hdr = '#  img_suffix  SNSegID  z_true  phase_true  Av_true  ' + \
          'z900  z900_lowerr  z900_uperr  ' + \
          'phase900  phase900_lowerr  phase900_uperr  ' + \
          'Av900  Av900_lowerr  Av900_uperr  ' + \
          'z1800  z1800_lowerr  z1800_uperr  ' + \
          'phase1800  phase1800_lowerr  phase1800_uperr  ' + \
          'Av1800  Av1800_lowerr  Av1800_uperr  ' + \
          'z3600  z3600_lowerr  z3600_uperr  ' + \
          'phase3600  phase3600_lowerr  phase3600_uperr  ' + \
          'Av3600  Av3600_lowerr  Av3600_uperr'

with open(resfile, 'w') as fh:
    fh.write(res_hdr + '\n')

    # Arrays to loop over
    pointings = np.arange(0, 1)
    detectors = np.arange(1, 10, 1)

    for pt in pointings:
        for det in detectors:

            # ----- Get img suffix, segid, and truth values
            img_suffix = img_filt + str(pt) + '_' + str(det)

            if img_suffix == 'Y106_0_2':
                continue

            # ----- Read in sed.lst
            sedlst_header = ['segid', 'sed_path']
            sedlst_path = roman_slitless_dir + '/pylinear_lst_files/' + 'sed_' + img_suffix + '.lst'
            sedlst = np.genfromtxt(sedlst_path, dtype=None, names=sedlst_header, encoding='ascii')

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
                #snr1 = get_snr(wav1, flam1)

                # ----- Write to file
                # --- ID and true quantities
                fh.write(img_suffix + '  ' + str(segid) + '  ')
                fh.write('{:.3f}'.format(true_z) + '  ' + str(true_phase) + '  ')
                fh.write('{:.3f}'.format(true_av) + '  ')# + '{:.2f}'.format(snr) + '  ')

                # ----- Construct the filenames for this segid
                snstr1 = str(segid) + '_' + img_suffix + exptime1
                emcee_savefile1 = results_dir + \
                                  'emcee_sampler_sn' + snstr1 + '.h5'

                snstr2 = str(segid) + '_' + img_suffix + exptime2
                emcee_savefile2 = results_dir + \
                                  'emcee_sampler_sn' + snstr2 + '.h5'

                snstr3 = str(segid) + '_' + img_suffix + exptime3
                emcee_savefile3 = results_dir + \
                                  'emcee_sampler_sn' + snstr3 + '.h5'

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

                    # --- EXPTIME 900 seconds
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
                    fh.write('{:.3f}'.format(av1800_uperr) + '  ')

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

