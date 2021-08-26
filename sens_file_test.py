import numpy as np
import matplotlib.pyplot as plt

datadir = '/Volumes/Joshi_external_HDD/Roman/sensitivity_files/'

def get_sens(mag, flam_fac):

    # Read in manually copy pasted parts from Jeff Kruk's file
    s = np.genfromtxt(datadir + 'abmag' + str(int(mag)) + '_prism_sens_kruk.txt', 
        dtype=None, names=['wav', 'sp_ht', 'flam', 'counts', 'snr'], 
        usecols=(0, 1, 3, 4, 5), skip_header=3, encoding='ascii')
    
    wav = s['wav'] * 1e4  # convert microns to angstroms
    #print('Wavelength grid:', wav)
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

mags = [19, 21, 23, 25]
flam_fac = [1e17, 1e18, 1e18, 1e19]

for i, mag in enumerate(mags):
    wav, sens = get_sens(mag, flam_fac[i])
    ax.plot(wav, sens, label= 'AB = ' + str(mag))

ax.legend(loc=0, fontsize=14)

plt.show()

