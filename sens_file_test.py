import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

import os
import sys
import socket

if 'plffsn2' in socket.gethostname():
    extdir = '/astro/ffsn/Joshi/'
    datadir = extdir + 'sensitivity_files/'
    pylinear_ref_dir = extdir + 'pylinear_ref_files/pylinear_config/Roman/'
else:
    home = os.getenv('HOME')
    datadir = '/Volumes/Joshi_external_HDD/Roman/sensitivity_files/'
    pylinear_ref_dir = home + '/Documents/pylinear_ref_files/pylinear_config/Roman/'  # noqa


def get_sens_v2_exptime(mag, flam_fac):

    # Read in manually copy pasted parts from Jeff Kruk's file
    datafile = datadir + 'abmag' + str(int(mag)) + '_prism_sens_kruk.txt'
    s = np.genfromtxt(datafile, dtype=None,
                      names=['wav', 'sp_ht', 'input_flam', 'counts',
                             'spec_zodi1', 'spec_zodi1p1', 'spec_zodi1p2'],
                      usecols=(0, 1, 3, 4, 7, 10, 13), skip_header=3,
                      encoding='ascii')

    wav = s['wav'] * 1e4  # convert microns to angstroms
    # print('Wavelength grid:', wav)

    exptime = 1001.91

    # Scale back to W/m2/micron
    # In Jeff Kruk's file they've been scaled up
    # by some factor dependent on the AB mag
    spec = s['spec_zodi1p1'] / exptime
    flam_watt_m2_micron = spec / flam_fac

    # Convert from W/m2/micron to erg/cm2/s/A
    # 1 W/m2/micron = 0.1 erg/cm2/s/A
    conv_fac = 0.1
    flam_cgs = flam_watt_m2_micron * conv_fac

    # Get the correct count rate
    # Note that the count rate in the file has been summed
    # over pixels vertically (perpendicular to the spectral trace)
    cps = s['counts'] / s['sp_ht']

    sens = cps / flam_cgs

    return wav, sens


def get_sens(mag, flam_fac):

    # Read in manually copy pasted parts from Jeff Kruk's file
    prism_file = datadir + 'abmag' + str(int(mag)) + '_prism_sens_kruk.txt'
    s = np.genfromtxt(prism_file, dtype=None,
                      names=['wav', 'sp_ht', 'flam', 'counts', 'snr'],
                      usecols=(0, 1, 3, 4, 5),
                      skip_header=3, encoding='ascii')

    wav = s['wav'] * 1e4  # convert microns to angstroms
    # print('Wavelength grid:', wav)
    flam_cgs = 0.1 * s['flam'] / flam_fac

    # Get the correct count rate
    # Note that the count rate in the file has been summed
    # over pixels vertically (perpendicular to the spectral trace)
    cps = s['counts'] / s['sp_ht']

    sens = cps / flam_cgs

    return wav, sens


# ----------
fig = plt.figure()
ax = fig.add_subplot(111)

mags = [19, 21, 23, 24]  # , 25]
flam_fac = [1e17, 1e18, 1e18, 1e19]  # , 1e19]

for i, mag in enumerate(mags):
    print('Working on mag:', i, mag)
    wav, sens = get_sens(mag, flam_fac[i])
    ax.plot(wav, sens, label='AB = ' + str(mag))

# Ensure that every curve above is identical
# i.e., they should all lie exactly on top of one another
# print(wav, len(wav))
# wav_idx = np.where((wav >= 7800) & (wav <= 18000))[0]
# print(wav[wav_idx], len(wav_idx))

ax.set_xlabel('Wavelength [Angstroms]', fontsize=14)
ax.set_ylabel('Sensitivity [count rate/Flambda]', fontsize=14)

# This block to fit with a polynomial only used
# with the exptime version of the get_sens func
"""
# Approx by some line fit to the central part of the curve
# This polynomial approximation is what will be written to file
cen_idx = np.where((wav >= 12000) & (wav <= 16500))[0]
wfit = wav[cen_idx]
sensfit = sens[cen_idx]

pp = np.polyfit(x=wfit, y=sensfit, deg=1)
p = np.poly1d(pp)

poly_sens = p(wav)

# Also force it to drop to zero
# below 7440 and above 18150
poly_sens_mod = np.zeros(len(wav))
for k in range(len(wav)):
    current_wav = wav[k]
    if current_wav <= 7440 or current_wav >= 18150:
        poly_sens_mod[k] = 0.0
    else:
        poly_sens_mod[k] = poly_sens[k]

ax.plot(wav, poly_sens_mod, color='k', lw=2.0)
"""

ax.legend(loc=0, fontsize=14)
# plt.show()

"""
# Now save to a txt file
with open(datadir + 'Roman_prism_sensitivity.txt', 'w') as fh:
    fh.write('#  Wav  Sensitivity' + '\n')
    for i in range(len(wav)):
        fh.write('{:.3f}'.format(wav[i])
                 + '  '
                 + '{:.3e}'.format(sens[i])
                 + '\n')
"""

# Also save in two other places
# 1. FITS file for pyLINEAR
# 2. In the folder that has HST grism
# sensitivities as well for comparison
home = os.getenv('HOME')
sens_dir = home + '/Documents/HST_Roman_sensitivities/'

col1 = fits.Column(name='Wavelength', format='E', array=wav)
col2 = fits.Column(name='Sensitivity', format='E', array=sens)
col3 = fits.Column(name='Error', format='E', array=np.zeros(len(sens)))
cols = fits.ColDefs([col1, col2, col3])

thdu = fits.BinTableHDU.from_columns(cols)

hdul = fits.HDUList()
hdul.append(thdu)

if 'plffsn2' not in socket.gethostname():
    hdul.writeto(sens_dir + 'Roman_p127_sens.fits', overwrite=True)

hdul.writeto(pylinear_ref_dir + 'Roman_p127_sens.fits', overwrite=True)

sys.exit(0)
