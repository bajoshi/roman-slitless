from astropy.io import fits
import numpy as np

import matplotlib.pyplot as plt

import os
import sys
import warnings

home = os.getenv('HOME')

sens_dir = home + '/Documents/HST_Roman_sensitivities/'

# To suppress a warning about division by zero 
# when we take the ratio of sensitivities.
warnings.filterwarnings('ignore')

# ------- HST 
hst_g102 = fits.open(sens_dir + 
                     'WFC3.IR.G102.cal.V4.32/WFC3.IR.G102.1st.sens.2.fits')
g102_wav = hst_g102[1].data['WAVELENGTH']
g102_sen = hst_g102[1].data['SENSITIVITY']

hst_g141 = fits.open(sens_dir + 
                     'WFC3.IR.G141.cal.V4.32/WFC3.IR.G141.1st.sens.2.fits')
g141_wav = hst_g141[1].data['WAVELENGTH']
g141_sen = hst_g141[1].data['SENSITIVITY']

# ------- Roman 
# roman_grism_1 = fits.open(sens_dir + 'Roman_g150_1_throughput_20190325.fits')
# g150_wav = roman_grism_1[1].data['Wavelength']
# g150_sen = roman_grism_1[1].data['Sensitivity']

# 'Roman_p127_1_throughput_20190325.fits')11
roman_prism = fits.open(sens_dir + 'Roman_p127_sens.fits')
p127_wav = roman_prism[1].data['Wavelength']
p127_sen = roman_prism[1].data['Sensitivity']

# --------------
# fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 7))

fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))

sens_unit = r"$\mathrm{[(e^-/s)/(erg\, s^{-1}\, cm^{-2}\, \AA^{-1})]}$"

ax1.set_xlabel(r'$\mathrm{Wavelength\ [\AA]}$', fontsize=15)
ax1.set_ylabel(r'$\mathrm{Sensitivity}$' + ' ' + sens_unit,
               fontsize=14)

ax1.plot(g102_wav, g102_sen, '--', color='blue', 
         label='HST WFC3/G102, +1 order')
ax1.plot(g141_wav, g141_sen, '--', color='crimson', 
         label='HST WFC3/G141, +1 order')

# ax.plot(g150_wav, g150_sen, color='purple', label='Roman grism, +1 order')
ax1.plot(p127_wav, p127_sen, color='dodgerblue', label='Roman prism')

ax1.legend(loc='upper left', fontsize=12, frameon=False)

ax1.set_xlim(7000, 19000)
ax1.set_yscale('log')
ax1.set_ylim(7e14, 1e18)

fig.savefig('figures/hst_roman_sensitivities.pdf', dpi=200, 
            bbox_inches='tight')

sys.exit(0)

# Now divide the sensitivities and find out by how much
# the Roman prism is more sensitive
# Start by creating an array to hold the HST grism
# sensitivities. Because there are two HST grisms, over 
# the entire wav range we are going to choose whichever
# one of the two grisms is more sensitive.
hst_sens = np.zeros(len(p127_wav))
for i in range(len(p127_wav)):

    current_wav = p127_wav[i]

    g102_wav_idx = np.argmin(abs(current_wav - g102_wav))
    g141_wav_idx = np.argmin(abs(current_wav - g141_wav))

    g102se = g102_sen[g102_wav_idx]
    g141se = g141_sen[g141_wav_idx]

    if g102se > g141se:
        hst_sens[i] = g102se
    else:
        hst_sens[i] = g141se

sens_ratio = p127_sen/hst_sens
ax2.plot(p127_wav, sens_ratio, color='k')

#ax1.plot(p127_wav, hst_sens, color='gray', lw=5.0)

# Get the average factor of improvement for the Roman 
# prism within the central wavelength coverage.
midwav_idx = np.where((p127_wav >= 9000) & (p127_wav <= 16000))[0]
print('Average of sensitivity ratio between 0.9 to 1.6 microns:', 
      '{:.3f}'.format(np.mean(sens_ratio[midwav_idx])))

g102_midwav_idx = np.where((p127_wav >= 9000) & (p127_wav <= 11000))[0]
print('Average of sensitivity ratio between 0.9 to 1.1 microns (for G102):', 
      '{:.3f}'.format(np.mean(sens_ratio[g102_midwav_idx])))

g141_midwav_idx = np.where((p127_wav >= 12000) & (p127_wav <= 16000))[0]
print('Average of sensitivity ratio between 1.2 to 1.6 microns (for G141):', 
      '{:.3f}'.format(np.mean(sens_ratio[g141_midwav_idx])))

ax2.set_ylabel('Sensitivity Ratio [Roman/HST]', fontsize=15)
ax2.set_xlim(7000, 19000)
ax2.set_ylim(1, 50)

fig.savefig('figures/hst_roman_sensitivities.pdf', dpi=200, 
            bbox_inches='tight')

sys.exit(0)
