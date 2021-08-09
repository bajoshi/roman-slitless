import numpy as np
from scipy.interpolate import griddata
from scipy.integrate import trapz

import os
import sys

import matplotlib.pyplot as plt

home = os.getenv('HOME')
fitting_utils = home + "/Documents/GitHub/roman-slitless/fitting_pipeline/utils/"

# Also load in lookup table for luminosity distance
dl_cat = np.genfromtxt(fitting_utils + 'dl_lookup_table.txt', dtype=None, names=True)
# Get arrays 
dl_z_arr = np.asarray(dl_cat['z'], dtype=np.float64)
dl_cm_arr = np.asarray(dl_cat['dl_cm'], dtype=np.float64)
age_gyr_arr = np.asarray(dl_cat['age_gyr'], dtype=np.float64)

del dl_cat

def get_dl_at_z(z):

    adiff = np.abs(dl_z_arr - z)
    z_idx = np.argmin(adiff)
    dl = dl_cm_arr[z_idx]

    return dl

def apply_redshift(restframe_wav, restframe_lum, redshift):

    dl = get_dl_at_z(redshift)

    redshifted_wav = restframe_wav * (1 + redshift)
    redshifted_flux = restframe_lum / (4 * np.pi * dl * dl)

    return redshifted_wav, redshifted_flux

def filter_conv(filter_wav, filter_thru, spec_wav, spec_flam):

    # First grid the spectrum wavelengths to the filter wavelengths
    spec_on_filt_grid = griddata(points=spec_wav, values=spec_flam, xi=filter_wav)

    # Remove NaNs
    valid_idx = np.where(~np.isnan(spec_on_filt_grid))

    filter_wav = filter_wav[valid_idx]
    filter_thru = filter_thru[valid_idx]
    spec_on_filt_grid = spec_on_filt_grid[valid_idx]

    # Now do the two integrals
    num = trapz(y=spec_on_filt_grid * filter_thru, x=filter_wav)
    den = trapz(y=filter_thru, x=filter_wav)

    filter_flux = num / den

    return filter_flux

def main():

    # Read in SALT2 SN IA file from Lou
    salt2_spec = np.genfromtxt(fitting_utils + "salt2_template_0.txt", \
        dtype=None, names=['day', 'lam', 'flam'], encoding='ascii')

    # Read in g-band filter curve
    # This is HST ACS/WFC F435W
    # Make sure the g-band magnitude Lou gave you is after throughput
    #absmag = -19.5  # this is the abs AB mag at peak, i.e., day=0
    absmag = -18.4  # in F105W

    #redshift = 0.00001  # assuming this small redshift # neglecting k-corrections

    filt = np.genfromtxt(home + '/Documents/GitHub/massive-galaxies' + \
                         '/grismz_pipeline/F105W_IR_throughput.csv', \
                         delimiter=',', dtype=None, names=True, encoding='ascii', usecols=(1,2))
    # This is the throughput 
    # Can be checked by plotting and comparing plot with throughput on HST documentation website
    # https://stsci.edu/hst/instrumentation/acs/data-analysis/system-throughputs
    # For WFC3 filters:
    # https://hst-docs.stsci.edu/wfc3ihb/chapter-7-ir-imaging-with-wfc3/7-5-ir-spectral-elements
    lam_pivot = 10552.0  # in Angstroms
    lam_pivot *= 1e-8    # in cm
    speed_of_light_cms = 3e10

    # Pull out day=0 SALT2 template
    day_idx = np.where(salt2_spec['day'] == 0)[0]

    day0_template_lam = salt2_spec['lam'][day_idx]
    day0_template_readflam = salt2_spec['flam'][day_idx]
    # This quantity, i.e., day0_template_readflam, is NOT luminosity 
    # It is also NOT flux. The template has been scaled in some way.
    # This can be seen by checking the plot below.

    # PLot the day=0 template to check
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(day0_template_lam, day0_template_readflam)
    plt.show()
    """

    # Now apply redshift and compute magnitude through filter and then 
    # find scaling factor required to get the known abs mag through filter
    # ------ Apply redshift
    #sn_lam_z, sn_flam_z = \
    #apply_redshift(day0_template_lam, day0_template_readflam, redshift)

    # Now convolve with filter
    filter_flux = filter_conv(filt['Wave_Angstroms'], filt['Throughput'], day0_template_lam, day0_template_readflam)

    print("\nF-lambda of redshifted template through filter:", filter_flux)
    print("Required ABSOLUTE magnitude:", absmag)

    # Now compute the apparent magnitudes
    # Once through the filter and 
    # the other through the distance modulus equation (this is the correct one)
    fnu = filter_flux * lam_pivot**2 / speed_of_light_cms
    app_mag = -2.5 * np.log10(fnu) - 48.6
    print("\nAPPARENT magnitude for TEMPLATE:", app_mag)

    # ----
    #dl = get_dl_at_z(redshift)  # returns dl in cm
    #dl = dl * 3.241e-19  # convert to pc
    #print("\nLuminosity distance to given redshift:", "{:.3f}".format(dl), "parsecs.")

    app_mag_correct = absmag # since I'm assuming that the SN is at z=0 i.e., at dl=10 pc
    print("APPARENT magnitude from DISTANCE MODULUS equation:", app_mag_correct)

    # Find amount to shift apparent magnitude through filter to
    # match the correct apparent magnitude from distance modulus equation
    correct_fnu = 10**(-1 * 0.4 * (app_mag_correct + 48.6))
    scalefac = correct_fnu / fnu 

    print("\n----->Therefore, scaling factor:", '{:.3e}'.format(scalefac))

    # Now check that you get the correct apparent magnitude 
    # if you use this scalefactor and do the redshifting
    # and filter computation again.
    day0_template_llam = day0_template_readflam * scalefac

    #sn_lam_z_correct, sn_flam_z_correct = \
    #apply_redshift(day0_template_lam, day0_template_llam, redshift)
    filter_flux_recomp = \
    filter_conv(filt['Wave_Angstroms'], filt['Throughput'], day0_template_lam, day0_template_llam)
    print("\nFilter flux recomputed after scaling factor is applied:", filter_flux_recomp)
    fnu_recomp = filter_flux_recomp * lam_pivot**2 / speed_of_light_cms
    app_mag_recomp = -2.5 * np.log10(fnu_recomp) - 48.6
    print("\nAPPARENT magnitude for TEMPLATE after scaling factor is applied:", app_mag_recomp)
    print("This should match the correct apparent magnitude above.")

    print("\n")

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)

