import sncosmo
import numpy as np

import os
import sys

import matplotlib.pyplot as plt

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

cwd = os.getcwd()
throughput_dir = cwd + '/fitting_pipeline/utils/throughputs/'

# Making this global because I need a func that 
# will let me go from a SN redshift to apparent 
# mag in Roman F106.
# That will need the model to be global.
# Get the SN Ia model
model = sncosmo.Model(source='hsiao')
# by default it is the latest version


def get_sn_app_mag(sn_redshift, band=None):

    # First redshift the model
    model.set(z=redshift)

    # Now set the absolute mag
    model.set_source_peakabsmag(-19.0, 'bessellb', 'ab')

    # get model apparent mag
    appmag = model.bandmag(band, 'ab', time=0.0)

    return appmag


if __name__ == '__main__':

    # Read in the required bandpasses
    # Using WFC3/IR/F105W for Roman F106
    bessellb = sncosmo.get_bandpass('bessellb')
    hstf105 = sncosmo.get_bandpass('f105w')

    # Create an array of redshifts at which to
    # infer distance modulus
    redshift_arr = np.arange(0.01, 2.5, 0.01)
    distmod_inferred = np.zeros(len(redshift_arr))
    distmod_lcdm = np.zeros(len(redshift_arr))

    for i, redshift in enumerate(redshift_arr):

        appmag_f105w = get_sn_app_mag(redshift, band='f105w')
        appmag_bessellb = get_sn_app_mag(redshift, band='bessellb')

        distmod_inferred[i] = appmag_bessellb - (-19.0)
        # The -19.0 here is hte assumed absolute magnitude
        # in the same band as the above computed apparent mag.

        # LCDM distance modulus
        # First need dl
        dl_mpc = cosmo.luminosity_distance(redshift).value
        distmod_lcdm[i] = 5 * np.log10(dl_mpc) + 25

        print(i, 
              '{:.2f}'.format(redshift), 
              '{:.3f}'.format(appmag_bessellb))
    
    # Check if the inferred distance modulus follows LCDM
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel('Redshift', fontsize=15)
    ax.set_ylabel('Distance Modulus', fontsize=15)

    ax.scatter(redshift_arr, distmod_inferred, color='k',
               zorder=2, label='Inferred DM', s=20)

    # Also plot the LCDM line 
    ax.plot(redshift_arr, distmod_lcdm, color='crimson', lw=2.0,
            zorder=1, label='LCDM DM')

    ax.legend(loc=0, frameon=False, fontsize=13)

    plt.show()

    sys.exit(0)
