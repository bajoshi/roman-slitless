import scipy.integrate as spint
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
astropy_cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)

# -------- Define cosmology -------- # 
# Flat Lambda CDM
H0 = 70.0  # km/s/Mpc
omega_m0 = 0.3
omega_lam0 = 1.0 - omega_m0

speed_of_light_kms = 299792.458  # km per s

def gen_lookup_table():

    zrange = np.arange(0.0001, 10.0001, 0.0001)

    # Open a txt file for saving
    with open('dl_lookup_table.txt', 'w') as fh:

        fh.write('#  z  dl_cm  dp_cm  age_gyr' + '\n')

        for j in range(len(zrange)):

            z = zrange[j]

            print("Redshift:", z, end='\r')

            # Get both distances
            dp_mpc = proper_distance(z)  # in Mpc
            dp_cm = dp_mpc * 3.086e24  # convert Mpc to cm

            dl_cm = dp_cm * (1+z)  # convert to lum dist

            # now get age
            age_at_z = astropy_cosmo.age(z).value  # in Gyr

            fh.write('{:.4f}'.format(z)     + '  ' 
                     '{:.8e}'.format(dl_cm) + '  '
                     '{:.8e}'.format(dp_cm) + '  '
                     '{:.5e}'.format(age_at_z) + '\n')

    print("Lookup table saved.")

    return None

def print_info():

    print("Flat Lambda CDM cosmology assumed.")
    print("H0: ", H0, "km/s/Mpc")
    print("Omega_m:", omega_m0)
    print("Omega_lambda:", "{:.3f}".format(omega_lam0))

    return None

def proper_distance(redshift):
    """
    This function will integrate 1/(a*a*H)
    between scale factor at emission to scale factor of 1.0.

    Will return proper distance in megaparsecs.
    """
    ae = 1 / (1 + redshift)

    p = lambda a: 1/(a*a*H0*np.sqrt((omega_m0/a**3) + omega_lam0 + ((1 - omega_m0 - omega_lam0)/a**2)))
    dp = spint.quadrature(p, ae, 1.0)

    dp = dp[0] * speed_of_light_kms

    return dp

def luminosity_distance(redshift):
    """
    Returns luminosity distance in megaparsecs for a given redshift.
    """

    # Get proper distance and multiply by (1+z)
    dp = proper_distance(redshift)  # returns answer in Mpc
    dl = dp * (1+redshift)  # dl also in Mpc

    return dl

def main():
    gen_lookup_table()
    return None

if __name__ == '__main__':
    main()


