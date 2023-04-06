import numpy as np
from scipy.interpolate import griddata
from scipy.integrate import trapz

import os
import sys

import matplotlib.pyplot as plt

home = os.getenv('HOME')
fitting_utils = os.getcwd() + "/fitting_pipeline/utils/"

# Also load in lookup table for luminosity distance
dl_cat = np.genfromtxt(fitting_utils + 'dl_lookup_table.txt',
                       dtype=None, names=True)
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
    redshifted_flux = restframe_lum / (4 * np.pi * dl * dl * (1 + redshift))

    return redshifted_wav, redshifted_flux


def filter_conv(filter_wav, filter_thru, spec_wav, spec_flam):

    # First grid the spectrum wavelengths to the filter wavelengths
    spec_on_filt_grid = griddata(points=spec_wav, values=spec_flam,
                                 xi=filter_wav)

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

    cmtopc = 3.241e-19

    # Read in SALT2 SN IA file from Lou
    salt2_spec = np.genfromtxt(fitting_utils + "templates/salt2_template_0.txt",
                               dtype=None, names=['day', 'lam', 'flam'], 
                               encoding='ascii')

    # Read in g-band filter curve
    # This is HST ACS/WFC F435W
    # Make sure the g-band magnitude Lou gave you is after throughput
    # absmag = -19.5  # this is the abs AB mag at peak, i.e., day=0
    absmag = -18.4  # in F105W

    redshift = 0.00001  # assuming this small redshift 
    # neglecting k-corrections
    print('Assuming a small redshift of:', redshift)

    filt = np.genfromtxt(home + '/Documents/GitHub/massive-galaxies' + 
                         '/grismz_pipeline/F105W_IR_throughput.csv', 
                         delimiter=',', dtype=None, names=True, 
                         encoding='ascii', usecols=(1, 2))
    # This is the throughput 
    # Can be checked by plotting and comparing plot with 
    # throughput on HST documentation website
    # https://stsci.edu/hst/instrumentation/acs/data-analysis/system-throughputs
    # For WFC3 filters:
    # https://hst-docs.stsci.edu/wfc3ihb/chapter-7-ir-imaging-with-wfc3/7-5-ir-spectral-elements
    lam_pivot = 10552.0  # in Angstroms
    speed_of_light_ang = 3e18  # in Angstroms per second

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

    # First compute the mag of the base template through F105W
    # flam_basetemplate = filter_conv(filt['Wave_Angstroms'], 
    #                                 filt['Throughput'], 
    #                                 day0_template_lam, 
    #                                 day0_template_readflam)
    # fnu_basetemplate = lam_pivot**2 * flam_basetemplate / speed_of_light_ang
    # base_absmag = -2.5 * np.log10(fnu_basetemplate) - 48.6
    # I'm calling this absolute magnitude because I suspect
    # the template has luminosity density in erg/s/A
    # Not really using this anywhere though.

    # Now apply redshift and compute magnitude through filter and then 
    # find scaling factor required to get the known abs mag through filter
    # ------ Apply redshift
    sn_lam_z, sn_flam_z = \
        apply_redshift(day0_template_lam, day0_template_readflam, redshift)

    # Now convolve with filter
    filter_flux = filter_conv(filt['Wave_Angstroms'], filt['Throughput'], 
                              sn_lam_z, sn_flam_z)

    print("\nF-lambda of redshifted template through filter:", filter_flux)
    print("Required ABSOLUTE magnitude:", absmag)

    # Now compute the apparent magnitudes
    # Once through the filter and 
    # the other through the distance modulus equation (this is the correct one)
    fnu = filter_flux * lam_pivot**2 / speed_of_light_ang
    app_mag = -2.5 * np.log10(fnu) - 48.6
    print("APPARENT magnitude for TEMPLATE:", app_mag)

    # ----
    dl = get_dl_at_z(redshift)  # returns dl in cm
    dl = dl * cmtopc  # convert to pc
    print("\nLuminosity distance to given redshift:", 
          "{:.3f}".format(dl), "parsecs.")

    app_mag_correct = absmag + 5 * np.log10(dl / 10)
    print("APPARENT magnitude from DISTANCE MODULUS equation:", 
          app_mag_correct)

    # Find amount to shift apparent magnitude through filter to
    # match the correct apparent magnitude from distance modulus equation
    correct_fnu = 10**(-1 * 0.4 * (app_mag_correct + 48.6))
    scalefac = correct_fnu / fnu 

    print("\n----->Therefore, scaling factor:", '{:.3e}'.format(scalefac))

    # Now check that you get the correct apparent magnitude 
    # if you use this scalefactor and do the redshifting
    # and filter computation again.
    day0_template_llam = day0_template_readflam * scalefac

    sn_lam_z_correct, sn_flam_z_correct = \
        apply_redshift(day0_template_lam, day0_template_llam, redshift)
    filter_flux_recomp = \
        filter_conv(filt['Wave_Angstroms'], filt['Throughput'], 
                    sn_lam_z_correct, sn_flam_z_correct)
    print("\nFilter flux recomputed after scaling factor is applied:", 
          filter_flux_recomp)
    fnu_recomp = filter_flux_recomp * lam_pivot**2 / speed_of_light_ang
    app_mag_recomp = -2.5 * np.log10(fnu_recomp) - 48.6
    print("APPARENT magnitude for TEMPLATE after scaling factor is applied:", 
          app_mag_recomp)
    print("This should match the correct apparent magnitude above.")

    print("\n")

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(day0_template_lam, day0_template_llam, color='k')
    # ax.set_xlim(8500.0, 13000.0)

    # ##########################################
    # ##########################################
    # ##########################################
    # Now confirm the scaling factor that you have 
    # by redshifting the scaled spectrum and making sure
    # that when the redshifted spectrum is convolved
    # with the F105W filter you get the expected
    # apparent magnitude.
    # -- check by comparing to the mag and redshift from
    # get_sn_z() in gen_sed_lst.py.
    # What you eventually want is that the scaling factor 
    # applied to the salt2 file simply gives flux in flam units.

    redshift_arr = np.arange(0.01, 3.01, 0.01)
    abmag = np.zeros(len(redshift_arr))

    for i in range(len(redshift_arr)):
        redshift = redshift_arr[i]
        sn_lam_z, sn_flam_z = apply_redshift(day0_template_lam, 
                                             day0_template_llam, redshift)

        filter_flam = filter_conv(filt['Wave_Angstroms'], filt['Throughput'], 
                                  sn_lam_z, sn_flam_z)
        # print('\nRedshifted flam through filter:', filter_flam)
        fnu = filter_flam * lam_pivot**2 / speed_of_light_ang

        mag = -2.5 * np.log10(fnu) - 48.6
        abmag[i] = mag

        print('{:.3f}'.format(mag), '  ', '{:.4f}'.format(redshift))

        # fig1 = plt.figure()
        # ax1 = fig1.add_subplot(111)
        # ax1.plot(sn_lam_z, sn_flam_z, color='crimson')
        # ax1.set_xlim(8500.0, 13000.0)
        # plt.show()
        # sys.exit()

    # Check plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.scatter(redshift_arr, abmag - absmag, s=15, color='k')
    
    axt = ax.twinx()
    axt.scatter(redshift_arr, abmag, s=2, color='r')

    ax.set_xlabel('Redshift', fontsize=12)
    ax.set_ylabel('Distance Modulus', fontsize=12)
    axt.set_ylabel('Apparent Magnitude (F105W)', fontsize=12)

    plt.show()

    return None


if __name__ == '__main__':
    main()
    sys.exit(0)
