import numpy as np
from astropy.io import fits

import matplotlib.pyplot as plt

import os
import sys

home = os.getenv('HOME')

sens_dir = home + '/Documents/HST_Roman_sensitivities/'

# ------- HST 
hst_g102 = fits.open(sens_dir + 'WFC3.IR.G102.cal.V4.32/WFC3.IR.G102.1st.sens.2.fits')
g102_wav = hst_g102[1].data['WAVELENGTH']
g102_sen = hst_g102[1].data['SENSITIVITY']

hst_g141 = fits.open(sens_dir + 'WFC3.IR.G141.cal.V4.32/WFC3.IR.G141.1st.sens.2.fits')
g141_wav = hst_g141[1].data['WAVELENGTH']
g141_sen = hst_g141[1].data['SENSITIVITY']

# ------- Roman 
roman_grism_1 = fits.open(sens_dir + 'Roman_g150_1_throughput_20190325.fits')
g150_wav = roman_grism_1[1].data['Wavelength']
g150_sen = roman_grism_1[1].data['Sensitivity']

roman_prism = fits.open(sens_dir + 'Roman_p127_1_throughput_20190325.fits')
p127_wav = roman_prism[1].data['Wavelength']
p127_sen = roman_prism[1].data['Sensitivity']

# --------------
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)

ax.set_xlabel('Wavelength', fontsize=15)
ax.set_ylabel('Sensitivity', fontsize=15)

ax.plot(g102_wav, g102_sen, '--', color='blue', label='HST WFC3/G102, +1 order')
ax.plot(g141_wav, g141_sen, '--', color='crimson', label='HST WFC3/G141, +1 order')

ax.plot(g150_wav, g150_sen, color='purple', label='Roman grism, +1 order')
ax.plot(p127_wav, p127_sen, color='dodgerblue', label='Roman prism')

ax.legend(loc=0, fontsize=12, frameon=False)

ax.set_xlim(7000, 20500)
ax.set_yscale('log')
ax.set_ylim(7e14, 3e17)

fig.savefig('figures/hst_roman_sensitivities.pdf', dpi=200, bbox_inches='tight')







