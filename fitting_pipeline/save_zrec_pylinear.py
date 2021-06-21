import numpy as np

ext_spectra_dir = "/Volumes/Joshi_external_HDD/Roman/roman_slitless_sims_results/"
results_dir = ext_spectra_dir + 'fitting_results/'

# Save results to text file
resfile = results_dir + 'zrecovery_pylinear_sims.txt'
res_hdr = '#  img_suffix  SNSegID  z_true  phase_true  Av_true  ' + \
          'z900  phase900  Av900  ' + \
          'z1800  phase1800  Av1800  ' + \
          'z3600  phase3600  Av3600  '

with open(resfile, 'w') as fh:
    fh.write(res_hdr + '\n')