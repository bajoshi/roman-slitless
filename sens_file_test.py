import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

import os, sys

datadir = '/Volumes/Joshi_external_HDD/Roman/sensitivity_files/'


def get_sens(mag, flam_fac):

    # Read in manually copy pasted parts from Jeff Kruk's file
    datafile = datadir + 'abmag' + str(int(mag)) + '_prism_sens_kruk.txt'
    s = np.genfromtxt(datafile, dtype=None, 
                      names=['wav', 'sp_ht', 'input_flam', 'counts', 'snr'], 
                      usecols=(0, 1, 3, 4, 5), skip_header=3, 
                      encoding='ascii')
    
    wav = s['wav'] * 1e4  # convert microns to angstroms
    # print('Wavelength grid:', wav)

    # exptime = # 1001.91

    # Scale back to W/m2/micron
    # In Jeff Kruk's file they've been scaled up 
    # by some factor dependent on the AB mag
    flam_watt_m2_micron = s['input_flam'] / flam_fac

    # Convert from W/m2/micron to erg/cm2/s/A
    # 1 W/m2/micron = 0.1 erg/cm2/s/A
    conv_fac = 0.1
    flam_cgs = flam_watt_m2_micron / conv_fac
    
    # Get the correct count rate
    # Note that the count rate in the file has been summed 
    # over pixels vertically (perpendicular to the spectral trace)
    cps = s['counts'] * s['sp_ht']
    
    sens = cps / flam_cgs

    return wav, sens


# ----------
fig = plt.figure()
ax = fig.add_subplot(111)

mags = [19, 21, 23, 25]
flam_fac = [1e17, 1e18, 1e18, 1e19]

for i, mag in enumerate(mags):
    print('Working on mag:', i, mag)
    wav, sens = get_sens(mag, flam_fac[i])
    ax.plot(wav, sens, label='AB = ' + str(mag))

# Ensure that every curve above is identical
# i.e., they should all lie exactly on top of one another
#print(wav, len(wav))
#wav_idx = np.where((wav >= 7800) & (wav <= 18000))[0]
#print(wav[wav_idx], len(wav_idx))

ax.set_xlabel('Wavelength [Angstroms]', fontsize=14)
ax.set_ylabel('Sensitivity [count rate/Flambda]', fontsize=14)

ax.legend(loc=0, fontsize=14)
plt.show()

# Now save to a txt file
with open(datadir + 'Roman_prism_sensitivity.txt', 'w') as fh:
    fh.write('#  Wav  Sensitivity' + '\n')
    for i in range(len(wav)):
        fh.write('{:.3f}'.format(wav[i]) 
                 + '  ' 
                 + '{:.3e}'.format(sens[i])
                 + '\n')


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
hdul.writeto(sens_dir + 'Roman_p127_sens.fits', overwrite=True)

pylinear_ref_dir = home + '/Documents/pylinear_ref_files/pylinear_config/Roman/'
hdul.writeto(pylinear_ref_dir + 'Roman_p127_sens.fits', overwrite=True)
