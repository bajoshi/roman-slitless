import numpy as np

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

from lc_plot import read_lc
from kcorr import get_kcorr_Hogg

def get_restframe_mag(phase, band):

    lc_phase, lc_absmag = read_lc(band)

    phase_idx = np.argmin(abs(lc_phase - phase))
    abs_mag = lc_absmag[phase_idx]

    return abs_mag

def get_distance_modulus(redshift):

    dl = cosmo.luminosity_distance(redshift).value  # in Mpc
    dl *= 1e6  # convert to parsecs

    dm = 5 * np.log10(dl/10)

    return dm

def get_sn_mag_F106(phase, redshift):

    # First get the absolute magnitude for 
    # the chosen band at the given phase
    abs_band = 'I'
    abs_mag = get_restframe_mag(phase, abs_band)

    # ----------------------------
    # Now apply a K correction to get to the apparent
    # magnitude in F106
    speed_of_light_ang = 3e18  # angstroms per second

    # ------- SN Ia spectrum from Lou
    salt2_spec = np.genfromtxt("templates/salt2_template_0.txt", 
        dtype=None, names=['day', 'lam', 'llam'], encoding='ascii')

    # ---------
    # Get spectrum corresponding to required phase
    day_idx = np.where(salt2_spec['day'] == int(phase))[0]

    spec_lam  = salt2_spec['lam'][day_idx]
    spec_llam = salt2_spec['llam'][day_idx]

    # Convert to l_nu and nu
    spec_nu   = speed_of_light_ang / spec_lam
    spec_lnu  = spec_lam**2 * spec_llam / speed_of_light_ang

    # ---------
    # Read in the two required filter curves
    # While the column label says transmission
    # it is actually the throughput that we want.
    # B and Y band in this case
    f105 = np.genfromtxt('throughputs/F105W_IR_throughput.csv', 
                         delimiter=',', dtype=None, names=['wav','trans'], 
                         encoding='ascii', usecols=(1,2), skip_header=1)

    f435 = np.genfromtxt('throughputs/f435w_filt_curve.txt', 
                         dtype=None, names=['wav','trans'], encoding='ascii')

    f606 = np.genfromtxt('throughputs/f606w_filt_curve.txt', 
                         dtype=None, names=['wav','trans'], encoding='ascii')

    f814 = np.genfromtxt('throughputs/HST_ACS_WFC.F814W.dat', 
                         dtype=None, names=['wav','trans'], encoding='ascii')

    # ---------
    # Now get the K-correction and distance modulus
    kcor = get_kcorr_Hogg(spec_lnu, spec_nu, redshift, f814, f105)

    # Get distance modulus based on cosmology
    dist_mod = get_distance_modulus(redshift)

    sn_mag_f106 = abs_mag + dist_mod + kcor

    # Delete numpy record arrays that have been read in
    del salt2_spec
    del f105, f435, f606, f814

    return sn_mag_f106

if __name__ == '__main__':
    
    # Run a test on the above function to see if 
    # the K-corrections are working correctly
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    phase = 0
    z_arr = np.arange(0.01, 3.01, 0.05)

    all_snmag = []

    for redshift in tqdm(z_arr):
        snmag = get_sn_mag_F106(phase, redshift)
        all_snmag.append(snmag)
        #print('{:.3f}'.format(redshift), '{:.3f}'.format(snmag))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(z_arr, all_snmag, color='k', s=2)
    plt.show()









