import pandas
import numpy as np

import os
import sys

import corner

import matplotlib.pyplot as plt

import prospect.io.read_results as reader

home = os.getenv('HOME')
adap_dir = home + '/Documents/adap2021/'

def main():
    
    # Read in df
    df = pandas.read_pickle(adap_dir + 'GOODS_South_SNeIa_host_phot.pkl')

    # Read in catalog from Santini et al.
    names_header=['id', 'ra', 'dec', 'zbest', 'zphot', 'zphot_l68', 'zphot_u68', 'Mmed', 'smed', 'Mdeltau']
    santini_cat = np.genfromtxt(home + '/Documents/GitHub/massive-galaxies/santini_candels_cat.txt',\
                  names=names_header, usecols=(0,1,2,9,13,14,15,19,20,40), skip_header=187)

    # Set mathcing tolerances
    ra_tol = 0.3 / 3600  # arcseconds expressed in deg
    dec_tol = 0.3 / 3600  # arcseconds expressed in deg

    search_ra = santini_cat['ra']
    search_dec = santini_cat['dec']

    # Empty lists for storing masses
    santini_mass = []
    fit_mass = []

    # Loop over all of our objects
    for i in range(len(df)):

        obj_ra = df['RA'][i]
        obj_dec = df['DEC'][i]

        # Now match our object to ones in Santini's cat
        match_idx = np.where( (np.abs(search_ra  - obj_ra)  <= ra_tol) & \
                              (np.abs(search_dec - obj_dec) <= dec_tol) )[0]

        print("\nMatch index:", match_idx, len(match_idx))

        print("Object RA:", obj_ra)
        print("Object DEC:", obj_dec)

        print("Matched RA:", search_ra[match_idx])
        print("Matched DEC:", search_dec[match_idx])

        assert len(match_idx)==1

        # Now read in the fitting results and get our stellar masses
        galaxy_seq = df['Seq'][i]
        h5file = adap_dir +  "emcee_" + str(galaxy_seq) + ".h5"
        if not os.path.isfile(h5file):
            continue

        result, obs, _ = reader.results_from(h5file, dangerous=False)

        print("Read in results from:", h5file)

        trace = result['chain']
        thin = 5
        trace = trace[:, ::thin, :]

        samples = trace.reshape(trace.shape[0] * trace.shape[1], trace.shape[2])
        cq_mass = corner.quantile(x=samples[:, 0], q=[0.16, 0.5, 0.84])

        # Append ot plotting arrays
        fit_mass.append(np.log10(cq_mass[1]))
        santini_mass.append(np.log10(santini_cat['Mmed'][match_idx]))

    # Make plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel(r'$\mathrm{M_s\ (CANDELS;\ Santini\ et\,al.\ 2015)}$')
    ax.set_ylabel(r'$\mathrm{M_s\ (this\ work)}$')

    ax.scatter(santini_mass, fit_mass, color='k', label='UV-Optical-NIR-MIR')
    ax.plot(np.arange(7.0, 12.5, 0.01), np.arange(7.0, 12.5, 0.01), '--', color='steelblue')

    ax.set_xlim(7.0, 12.0)
    ax.set_ylim(7.0, 12.0)

    ax.legend(fontsize=10, frameon=False)

    fig.savefig(adap_dir + 'candels_mass_comparison.pdf', dpi=300, bbox_inches='tight')

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)