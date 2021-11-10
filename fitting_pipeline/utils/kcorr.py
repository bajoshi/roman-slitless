import numpy as np
from scipy.interpolate import griddata, interp1d

def get_kcorr_Kim1996(sed_llam, sed_lam, redshift, filt_curve_x, filt_curve_y):

    #standard = 3631 * 1e-23  # in erg/s/Hz
    
    # Get wav and trans data from numpy record arrays
    x_wav = filt_curve_x['wav']
    x_trans = filt_curve_x['trans']

    y_wav = filt_curve_y['wav']
    y_trans = filt_curve_y['trans']

    # ---------------
    y1 = x_trans
    y2 = y_trans

    int1 = np.trapz(y=y1, x=x_wav)
    int2 = np.trapz(y=y2, x=y_wav)

    first_term = -2.5 * np.log10(int1 / int2)

    # ---------------
    # First interpolate
    sed_llam_interp_x = griddata(points=sed_lam, values=sed_llam, 
                                 xi=x_wav)
    sed_flux_interp_y = griddata(points=sed_lam*(1+redshift), 
                                 values=sed_llam/(1+redshift), 
                                 xi=y_wav)
    
    # Force NaNs to zero
    sed_llam_interp_x[np.where(np.isnan(sed_llam_interp_x))] = 0.0
    sed_flux_interp_y[np.where(np.isnan(sed_flux_interp_y))] = 0.0

    # Second term
    y3 = sed_llam_interp_x * x_trans
    y4 = sed_flux_interp_y * y_trans

    int3 = np.trapz(y=y3, x=x_wav)
    int4 = np.trapz(y=y4, x=y_wav)

    second_term = 2.5 * np.log10(int3 / int4)

    kcorr = first_term + second_term

    return kcorr

def get_kcorr_Hogg(sed_lnu, sed_nu, redshift, filt_curve_Q, filt_curve_R, verbose=False):
    """
    Returns the K-correction given a redshift and observed and 
    restframe bandpasses in which object magnitudes are measured. 
    It needs to be supplied with the object SED (L_nu and nu), 
    redshift, and with the rest frame and obs bandpasses. 

    This function uses the K-correction formula given in 
    eq 8 of Hogg et al. 2002.

    Arguments:
    sed_lnu: float array of luminosity density in erg/s/Hz
    sed_nu:  float array of frequency in Hz
    redshift: float scalar
    filt_curve_Q: numpy record array with two columns -- wav and trans
                  i.e., an ascii file with these two columns read with 
                  numpy genfromtxt. 
                  Wavelength in Angstroms.
                  Trans is actually throughput.
                  This is assumed to be the rest frame bandpass in 
                  which abs mag is known.
    filt_curve_R: numpy record array similar to filt_curve_Q above
                  This is assumed to be the observed frame bandpass in 
                  which app mag will be measured.
    verbose (bool; optional): parameter controlling verbosity 
                              default=False
                              Will show a plot if set to True

    Returns:
    kcorr_qr: float scalar
              K-correction dependent on redshift and on the bandpasses 
              Q and R with the above given quantities
    """

    speed_of_light_ang = 3e18  # angstroms per second

    # Redshift the spectrum
    nu_obs = sed_nu / (1+redshift)
    lnu_obs = sed_lnu * (1+redshift)
    # this does NOT need a (1+z)???
    # See eq 9 in Hogg+2002, it includes L_nu((1+z)*nu_obs)
    # This is just L_nu(nu_em). However the integral is still
    # performed in observed frequency space.

    # Convert filter wavlengths to frequency
    filt_curve_R_nu = np.divide(speed_of_light_ang, filt_curve_R['wav'])
    filt_curve_Q_nu = np.divide(speed_of_light_ang, filt_curve_Q['wav'])

    # Find indices where filter and spectra frequencies match
    R_nu_filt_idx = np.where((nu_obs <= filt_curve_R_nu[0]) & \
                             (nu_obs >= filt_curve_R_nu[-1]))[0]
    Q_nu_filt_idx = np.where((sed_nu <= filt_curve_Q_nu[0]) & \
                             (sed_nu >= filt_curve_Q_nu[-1]))[0]

    # Make sure the filter curve and the SED are on the same wavelength grid.
    # Filter R is in obs frame
    # Filter Q is in rest frame
    filt_curve_R_interp_obs = griddata(points=filt_curve_R_nu, 
        values=filt_curve_R['trans'], xi=nu_obs[R_nu_filt_idx], 
        method='linear', fill_value=0.0)
    filt_curve_Q_interp_rf = griddata(points=filt_curve_Q_nu, 
        values=filt_curve_Q['trans'], xi=sed_nu[Q_nu_filt_idx], 
        method='linear', fill_value=0.0)

    # Define standard for AB magnitdues
    # i.e., 3631 Janskys in L_nu units
    standard = 3631 * 1e-23  # in erg/s/Hz

    # Define integrands
    y1 = lnu_obs[R_nu_filt_idx] * filt_curve_R_interp_obs / nu_obs[R_nu_filt_idx]
    y2 = standard * filt_curve_Q_interp_rf / sed_nu[Q_nu_filt_idx]
    y3 = standard * filt_curve_R_interp_obs / nu_obs[R_nu_filt_idx]
    y4 = sed_lnu[Q_nu_filt_idx] * filt_curve_Q_interp_rf / sed_nu[Q_nu_filt_idx]

    # Now get the integrals required within the K-correction formula
    integral1 = np.trapz(y=y1, x=nu_obs[R_nu_filt_idx])
    integral2 = np.trapz(y=y2, x=sed_nu[Q_nu_filt_idx])
    integral3 = np.trapz(y=y3, x=nu_obs[R_nu_filt_idx])
    integral4 = np.trapz(y=y4, x=sed_nu[Q_nu_filt_idx])

    # Compute K-correction
    kcorr_qr = -2.5 * np.log10((1+redshift) * integral1 * integral2 / (integral3 * integral4))

    if verbose:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(figsize=(12,4.5), nrows=1, ncols=2)

        # ------------- Frequency space
        ax1t = ax1.twinx()
        
        # plot spectra
        ax1.plot(sed_nu, sed_lnu, color='k', label='Original spectrum')
        ax1.plot(nu_obs, lnu_obs, color='tab:red', label='Redshifted spectrum')

        # plot bandpasses
        ax1t.plot(nu_obs[R_nu_filt_idx], filt_curve_R_interp_obs, color='tab:olive', label='WFC3/IR/F105W')
        ax1t.plot(sed_nu[Q_nu_filt_idx], filt_curve_Q_interp_rf, color='royalblue', label='ACS/WFC/F435W')

        # Labels, limits, and legend
        ax1.legend(loc='upper left', frameon=False, fontsize=12)
        ax1t.legend(loc='upper right', frameon=False, fontsize=12)

        ax1.set_xscale('log')
        ax1.set_xlim(1e14, 2e15)

        ax1.set_xlabel('Frequency (Hz)', fontsize=13)
        ax1.set_ylabel('Flux density (erg/s/cm2/Hz)', fontsize=13)

        # ------------- Wavelength space
        ax2t = ax2.twinx()

        sed_lam = speed_of_light_ang / sed_nu
        sed_llam = speed_of_light_ang * sed_lnu / sed_lam**2
        
        # plot spectra
        ax2.plot(sed_lam, sed_llam, color='k', label='Original spectrum')
        ax2.plot(sed_lam*(1+redshift), sed_llam/(1+redshift), color='tab:red', label='Redshifted spectrum')

        # plot bandpasses
        ax2t.plot(filt_curve_R['wav'], filt_curve_R['trans'], color='tab:olive', label='WFC3/IR/F105W')
        ax2t.plot(filt_curve_Q['wav'], filt_curve_Q['trans'], color='royalblue', label='ACS/WFC/F435W')

        # Labels, limits, and legend
        ax2.legend(loc='upper center', frameon=False, fontsize=12)
        ax2t.legend(loc='center right', frameon=False, fontsize=12)
        ax2.set_xlim(2500, 15000)

        ax2.set_xlabel('Wavelength (Angstroms)', fontsize=13)
        ax2.set_ylabel('Flux density (erg/s/cm2/A)', fontsize=13)
        ax2t.set_ylabel('Throughput', fontsize=13)

        plt.pause(0.1)

        fig.clear()
        plt.close(fig)

    return kcorr_qr

if __name__ == '__main__':
    
    # This runs a couple tests on the above function
    # using a SN Ia spectrum at peak.
    import matplotlib.pyplot as plt
    import sys
    import os

    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

    speed_of_light_ang = 3e18  # angstroms per second

    # SN Ia spectrum from Lou
    salt2_spec = np.genfromtxt("templates/salt2_template_0.txt", 
        dtype=None, names=['day', 'lam', 'llam'], encoding='ascii')

    # Also load in lookup table for luminosity distance
    #dl_cat = np.genfromtxt('dl_lookup_table.txt', dtype=None, names=True)
    ## Get arrays 
    #dl_z_arr = np.asarray(dl_cat['z'], dtype=np.float64)
    #dl_cm_arr = np.asarray(dl_cat['dl_cm'], dtype=np.float64)
    #age_gyr_arr = np.asarray(dl_cat['age_gyr'], dtype=np.float64)
    #del dl_cat

    # Get day 0 spectrum
    day0_idx = np.where(salt2_spec['day'] == 0)[0]

    day0_lam = salt2_spec['lam'][day0_idx]
    day0_llam = salt2_spec['llam'][day0_idx]

    # Convert to l_nu and nu
    day0_nu  = speed_of_light_ang / day0_lam
    day0_lnu = day0_lam**2 * day0_llam / speed_of_light_ang

    # Read in the two required filter curves
    # While the column label says transmission
    # it is actually the throughput that we want.
    # B and Y band in this case
    f105 = np.genfromtxt('throughputs/F105W_IR_throughput.csv', 
                         delimiter=',', dtype=None, names=['wav','trans'], 
                         encoding='ascii', usecols=(1,2), skip_header=1)

    f435 = np.genfromtxt('throughputs/f435w_filt_curve.txt', 
                         dtype=None, names=['wav','trans'], 
                         encoding='ascii')

    bessel_b = np.genfromtxt('throughputs/Bessel_B.txt', dtype=None, 
        names=['wav','trans'], encoding='ascii')

    bessel_r = np.genfromtxt('throughputs/Bessel_R.txt', dtype=None, 
        names=['wav','trans'], encoding='ascii')

    zarr = np.arange(0.01, 3.0, 0.01)
    kcor_arr = np.zeros(len(zarr))
    dl_K_sum_arr = np.zeros(len(zarr))  # dl in Mpc

    for i in range(len(zarr)):
        redshift = zarr[i]
        kcor = get_kcorr_Hogg(day0_llam, day0_lam, redshift, bessel_b, bessel_r)
        kcor_arr[i] = kcor

        # Now compute the sum of 5log(dl) at the z and K-correction
        #z_idx = np.argmin(abs(dl_z_arr - redshift))
        #dl_cm = dl_cm_arr[z_idx]
        #dl_mpc = dl_cm / 3.086e24

        dl_mpc = cosmo.luminosity_distance(redshift).value  # using astropy # should give hte same result

        s = 5 * np.log10(dl_mpc) + kcor + 25.0

        dl_K_sum_arr[i] = s

        print(i, 
         '{:.2f}'.format(redshift), 
         '{:.2f}'.format(kcor), 
         '{:.2f}'.format(dl_mpc), 
         '{:.2f}'.format(s))

    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111)
    
    ax.set_title('K-correction for SN Ia spec at peak', fontsize=14)
    ax.set_xlabel('Redshift', fontsize=14)
    ax.set_ylabel('K-correction [F105W;obs - F435W;rest]', fontsize=14)

    ax.scatter(zarr, kcor_arr, s=5)
    
    axt = ax.twinx()
    axt.scatter(zarr, dl_K_sum_arr, s=5, color='k')

    axt.set_ylabel('m - M = 5log[dl(z)] + 25 + K(z)', fontsize=14)
    
    plt.show()
    sys.exit()

    # ----------------------
    # Another test: 
    # We have to make sure that given a SN apparent mag
    # with the assumption that it is at peak, that the 
    # redshift implied by the code below is cosmologically
    # consistent.

    # -------- Prep first
    # Need to save the dl and K-corr sum lookup
    #zrange = np.arange(0.0001, 8.0001, 0.0001)
    zrange = np.arange(0.01, 5.01, 0.01)
    # K-correction starts giving nonsense beyond z~9.5
    # I'm stopping at z=5 which is the limit beyond which
    # I cannot redshift the SALT2 spectrum. SALT2 spec starts 
    # at 1700A so at around z=5 it starts losing data
    # points within the F106 bandpass (which is the bandpass)
    # we're concerned with.

    # This array here only required for the test below
    # An array identical to this is being saved to the 
    # lookup table file here for use with other programs.
    dl_K_sum_lookup = np.zeros(len(zrange))

    # Open a txt file for saving
    with open('dl_lookup_table_Ksum.txt', 'w') as fh:

        fh.write('#  z  dl_cm  age_gyr  dl_K_sum' + '\n')

        for k in range(len(zrange)):

            z = zrange[k]
            
            print("Redshift:", z, end='\r')

            z_match = np.argmin(abs(dl_z_arr - z))
            
            dl_cm = dl_cm_arr[z_match]
            age_at_z = age_gyr_arr[z_match]

            # Now compute the sum of 5log(dl) at the z and K-correction
            kcor = get_kcorr(day0_lnu, day0_nu, z, f435, f105)
            dl_mpc = dl_cm / 3.086e24
            s = 5 * np.log10(dl_mpc) + 25 + kcor

            dl_K_sum_lookup[k] = s

            # Write to file
            fh.write('{:.4f}'.format(z)     + '  ' 
                     '{:.8e}'.format(dl_cm) + '  '
                     '{:.5e}'.format(age_at_z) + '  '
                     '{:.3f}'.format(s) + '\n')


    # ---------- Now do the test
    # Abs Mag of SN Ia in required band at peak
    absmag = -19.5  # assumed abs mag of SN Ia in HST ACS/F435W

    app_mag_arr = np.arange(16.0, 27.51, 0.01)
    matched_redshifts = np.zeros(len(app_mag_arr))

    for j in range(len(app_mag_arr)):
        mag = app_mag_arr[j]
        match_sum = mag - absmag
        z_idx = np.argmin(abs(dl_K_sum_lookup - match_sum))

        matched_redshifts[j] = dl_z_arr[z_idx]

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    
    ax1.set_xlabel('Redshift', fontsize=14)
    ax1.set_ylabel('Apparent magnitude', fontsize=14)

    ax1.scatter(matched_redshifts, app_mag_arr, s=5, color='k')
    
    dist_mod = app_mag_arr - absmag
    ax1t = ax1.twinx()
    ax1t.scatter(matched_redshifts, dist_mod, s=5, color='k')

    ax1t.set_ylabel('Distance Modulus', fontsize=14)

    plt.show()
    sys.exit(0)






