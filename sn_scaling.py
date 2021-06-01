import numpy as np
from scipy.interpolate import griddata
from scipy.integrate import trapz

import os
import sys

import matplotlib.pyplot as plt

home = os.getenv('HOME')
stacking_utils = home + '/Documents/GitHub/stacking-analysis-pears/util_codes/'
template_dir = home + "/Documents/roman_slitless_sims_seds/"

import mcmc_host_and_sn_fit as mcfit
sys.path.append(stacking_utils)
import proper_and_lum_dist as cosmo

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
    salt2_spec = np.genfromtxt(template_dir + "salt2_template_0.txt", \
        dtype=None, names=['day', 'lam', 'flam'], encoding='ascii')

    # Read in g-band filter curve
    # This is HST ACS/WFC F435W
    # Make sure the g-band magnitude Lou gave you is after throughput
    gmag = -19.5  # this is the abs AB mag at peak, i.e., day=0

    redshift = 0.05  # assuming this small redshift # neglecting k-corrections

    f435w = np.genfromtxt('/Users/baj/Documents/GitHub/massive-galaxies' + \
                          '/grismz_pipeline/f435w_filt_curve.txt', \
                          dtype=None, names=True, encoding='ascii')
    # This is the throughput 
    # Can be checked by plotting and comparing plot with throughput on HST documentation website
    # https://stsci.edu/hst/instrumentation/acs/data-analysis/system-throughputs

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
    sn_lam_z, sn_flam_z = \
    mcfit.apply_redshift(day0_template_lam, day0_template_readflam, redshift)

    # Now convolve with filter
    filter_flux = filter_conv(f435w['wav'], f435w['trans'], sn_lam_z, sn_flam_z)

    print("\nFlux of redshifted template through filter:", filter_flux)
    print("Required ABSOLUTE magnitude:", gmag)

    # Now compute the apparent magnitudes
    # Once through the filter and 
    # the other through the distance modulus equation (this is the correct one)
    app_mag = -2.5 * np.log10(filter_flux)
    print("\nAPPARENT magnitude for TEMPLATE:", app_mag)

    # ----
    dl = cosmo.luminosity_distance(redshift)  # returns dl in Mpc
    dl = dl * 1e6  # convert to pc
    print("\nLuminosity distance to given redshift:", "{:.3f}".format(dl), "parsecs.")

    app_mag_correct = gmag + 5 * np.log10(dl/10)
    print("APPARENT magnitude from DISTANCE MODULUS equation:", app_mag_correct)

    # Find amount to shift apparent magnitude through filter to
    # match the correct apparent magnitude from distance modulus equation
    correct_filter_flux = 10**(-1 * 0.4 * app_mag_correct)
    scalefac = correct_filter_flux / filter_flux 

    print("\n----->Therefore, scaling factor:", scalefac)

    # Now check that you get the correct apparent magnitude 
    # if you use this scalefactor and do the redshifting
    # and filter computation again.
    day0_template_llam = day0_template_readflam * scalefac

    sn_lam_z_correct, sn_flam_z_correct = \
    mcfit.apply_redshift(day0_template_lam, day0_template_llam, redshift)
    filter_flux_recomp = \
    filter_conv(f435w['wav'], f435w['trans'], sn_lam_z_correct, sn_flam_z_correct)
    print("\nFilter flux recomputed after scaling factor is applied:", filter_flux_recomp)
    app_mag_recomp = -2.5 * np.log10(filter_flux_recomp)
    print("\nAPPARENT magnitude for TEMPLATE after scaling factor is applied:", app_mag_recomp)
    print("This should match the correct apparent magnitude above.")

    print("\n")

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)

