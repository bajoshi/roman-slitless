import pandas
import numpy as np
import scipy

import os
import sys

import corner

import matplotlib.pyplot as plt

import prospect.io.read_results as reader

home = os.getenv('HOME')
adap_dir = home + '/Documents/adap2021/'

def get_cq_mass(result):

    trace = result['chain']
    thin = 5
    trace = trace[:, ::thin, :]

    samples = trace.reshape(trace.shape[0] * trace.shape[1], trace.shape[2])
    cq_mass = corner.quantile(x=samples[:, 0], q=[0.16, 0.5, 0.84])

    return cq_mass

def fitfunc(x, b, m):
    return 10**b * np.power(x, m)

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
    santini_mass_err = []

    fit_mass_allbands = []
    fit_mass_allbands_err = []

    fit_mass_ubriz = []
    fit_mass_ubriz_err = []

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
        h5file_ubriz = adap_dir + "emcee_" + str(galaxy_seq) + ".h5"

        h5file_allbands = adap_dir + 'goodss_param_sfh/' + "emcee_" + str(galaxy_seq) + ".h5"

        result_optical, obs, _ = reader.results_from(h5file_ubriz, dangerous=False)
        result_all, obs, _ = reader.results_from(h5file_allbands, dangerous=False)

        print("Read in ubriz results from:", h5file_ubriz)
        print("Read in all bands results from:", h5file_allbands)

        cq_mass_optical = get_cq_mass(result_optical)
        cq_mass_all = get_cq_mass(result_all)

        # Append ot plotting arrays
        fit_mass_allbands.append(cq_mass_all[1])
        fit_mass_allbands_lowerr = cq_mass_all[1] - cq_mass_all[0]
        fit_mass_allbands_uperr = cq_mass_all[2] - cq_mass_all[1]
        fit_mass_allbands_err.append([fit_mass_allbands_lowerr, fit_mass_allbands_uperr])

        fit_mass_ubriz.append(cq_mass_optical[1])
        fit_mass_ubriz_lowerr = cq_mass_optical[1] - cq_mass_optical[0]
        fit_mass_ubriz_uperr = cq_mass_optical[2] - cq_mass_optical[1]
        fit_mass_ubriz_err.append([fit_mass_ubriz_lowerr, fit_mass_ubriz_uperr])

        santini_mass.append(float(santini_cat['Mmed'][match_idx]))
        santini_mass_err.append(float(santini_cat['smed'][match_idx]))


    # ---------Convert to numpy arrays and reshape
    santini_mass = np.array(santini_mass)
    fit_mass_allbands = np.array(fit_mass_allbands)
    fit_mass_ubriz = np.array(fit_mass_ubriz)

    santini_mass_err = np.array(santini_mass_err)

    fit_mass_ubriz_err = np.array(fit_mass_ubriz_err)
    fit_mass_ubriz_err = fit_mass_ubriz_err.reshape((2, len(df)))

    fit_mass_allbands_err = np.array(fit_mass_allbands_err)
    fit_mass_allbands_err = fit_mass_allbands_err.reshape((2, len(df)))

    # --------- Make plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel(r'$\mathrm{M_s\ (CANDELS;\ Santini\ et\,al.\ 2015)}$')
    ax.set_ylabel(r'$\mathrm{M_s\ (this\ work)}$')

    # plot points
    ax.errorbar(santini_mass, fit_mass_allbands, xerr=santini_mass_err, yerr=fit_mass_allbands_err,
        fmt='o', ms=5.5, elinewidth=1.3, color='k', label='UV-Optical-NIR-MIR')
    ax.errorbar(santini_mass, fit_mass_ubriz, xerr=santini_mass_err, yerr=fit_mass_ubriz_err,
        fmt='o', ms=4.5, elinewidth=0.9, color='forestgreen', label='ubriz')

    ax.plot(np.arange(1e7, 1e13, 1e11), np.arange(1e7, 1e13, 1e11), '--', color='deepskyblue')

    # add a regression line
    xdata = np.log10(santini_mass)
    ydata1 = np.log10(fit_mass_allbands)
    ydata2 = np.log10(fit_mass_ubriz)

    x_arr = np.logspace(5.0, 12.5, num=1000, base=10)

    m1, logb1 = np.polyfit(xdata, ydata1, 1)
    ax.plot(x_arr, 10**logb1 * x_arr**m1, '--', color='slategrey')

    m2, logb2 = np.polyfit(xdata, ydata2, 1)
    ax.plot(x_arr, 10**logb2 * x_arr**m2, '--', color='limegreen')

    # -------------- 
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlim(1e7, 1e12)
    ax.set_ylim(1e7, 1e12)

    ax.legend(fontsize=10, frameon=False)

    fig.savefig(adap_dir + 'candels_mass_comparison.pdf', dpi=300, bbox_inches='tight')

    # --------------
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    ax1.set_xlabel(r'$\mathrm{M_{s;\,(all\ bands)}}$')
    ax1.set_ylabel(r'$\mathrm{log(M_{s;\,(all\ bands)})  -  log(M_{s;\,(ubriz)}) }$')

    deltamass = np.log10(fit_mass_allbands) - np.log10(fit_mass_ubriz)
    ax1.axhline(y=0.0, ls='--', color='deepskyblue', zorder=1)
    ax1.scatter(fit_mass_allbands, deltamass, s=10, color='k', zorder=2)

    ax1.set_xscale('log')

    fig1.savefig(adap_dir + 'mass_residuals.pdf', dpi=300, bbox_inches='tight')

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)