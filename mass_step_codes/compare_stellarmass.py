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

def do_south_comparison():
    
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

    fit_mass_briz = []
    fit_mass_briz_err = []

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

        h5file_allbands = adap_dir + "goodss_param_sfh/all_bands/" + \
                          "emcee_South_" + str(galaxy_seq) + ".h5"
        h5file_ubriz    = adap_dir + "goodss_param_sfh/ubriz/"     + \
                          "emcee_South_" + str(galaxy_seq) + ".h5"
        h5file_briz     = adap_dir + "goodss_param_sfh/briz/"      + \
                          "emcee_South_" + str(galaxy_seq) + ".h5"

        result_all, obs, _ = reader.results_from(h5file_allbands, dangerous=False)
        result_ubriz, obs, _ = reader.results_from(h5file_ubriz, dangerous=False)
        result_briz, obs, _ = reader.results_from(h5file_briz, dangerous=False)

        print("Read in all bands results from:", h5file_allbands)
        print("Read in ubriz results from:", h5file_ubriz)
        print("Read in briz results from:", h5file_briz)

        cq_mass_all = get_cq_mass(result_all)
        cq_mass_ubriz = get_cq_mass(result_ubriz)
        cq_mass_briz = get_cq_mass(result_briz)

        # Append ot plotting arrays
        santini_mass.append(float(santini_cat['Mmed'][match_idx]))
        santini_mass_err.append(float(santini_cat['smed'][match_idx]))

        fit_mass_allbands.append(cq_mass_all[1])
        fit_mass_allbands_lowerr = cq_mass_all[1] - cq_mass_all[0]
        fit_mass_allbands_uperr = cq_mass_all[2] - cq_mass_all[1]
        fit_mass_allbands_err.append([fit_mass_allbands_lowerr, fit_mass_allbands_uperr])

        fit_mass_ubriz.append(cq_mass_ubriz[1])
        fit_mass_ubriz_lowerr = cq_mass_ubriz[1] - cq_mass_ubriz[0]
        fit_mass_ubriz_uperr = cq_mass_ubriz[2] - cq_mass_ubriz[1]
        fit_mass_ubriz_err.append([fit_mass_ubriz_lowerr, fit_mass_ubriz_uperr])

        fit_mass_briz.append(cq_mass_briz[1])
        fit_mass_briz_lowerr = cq_mass_briz[1] - cq_mass_briz[0]
        fit_mass_briz_uperr = cq_mass_briz[2] - cq_mass_briz[1]
        fit_mass_briz_err.append([fit_mass_briz_lowerr, fit_mass_briz_uperr])


    # ---------Convert to numpy arrays and reshape
    santini_mass = np.array(santini_mass)
    santini_mass_err = np.array(santini_mass_err)

    fit_mass_allbands = np.array(fit_mass_allbands)
    fit_mass_allbands_err = np.array(fit_mass_allbands_err)
    fit_mass_allbands_err = fit_mass_allbands_err.reshape((2, len(df)))

    fit_mass_ubriz = np.array(fit_mass_ubriz)
    fit_mass_ubriz_err = np.array(fit_mass_ubriz_err)
    fit_mass_ubriz_err = fit_mass_ubriz_err.reshape((2, len(df)))

    fit_mass_briz = np.array(fit_mass_briz)
    fit_mass_briz_err = np.array(fit_mass_briz_err)
    fit_mass_briz_err = fit_mass_briz_err.reshape((2, len(df)))

    # --------- Make plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel(r'$\mathrm{M_s\ (CANDELS;\ Santini\ et\,al.\ 2015)}$')
    ax.set_ylabel(r'$\mathrm{M_s\ (this\ work)}$')

    # plot points
    ax.errorbar(santini_mass, fit_mass_allbands, xerr=santini_mass_err, yerr=fit_mass_allbands_err,
        fmt='o', ms=5.5, elinewidth=1.3, color='k', label='UV-Optical-NIR-MIR')
    ax.errorbar(santini_mass, fit_mass_ubriz, xerr=santini_mass_err, yerr=fit_mass_ubriz_err,
        fmt='o', ms=4.5, elinewidth=0.9, color='darkviolet', label='ubriz')
    ax.errorbar(santini_mass, fit_mass_briz, xerr=santini_mass_err, yerr=fit_mass_briz_err,
        fmt='o', ms=4.5, elinewidth=0.9, color='forestgreen', label='briz')

    ax.plot(np.arange(1e7, 1e13, 1e11), np.arange(1e7, 1e13, 1e11), '--', color='deepskyblue')

    # add a regression line
    xdata = np.log10(santini_mass)
    ydata1 = np.log10(fit_mass_allbands)
    ydata2 = np.log10(fit_mass_ubriz)
    ydata3 = np.log10(fit_mass_briz)

    x_arr = np.logspace(5.0, 12.5, num=1000, base=10)

    m1, logb1 = np.polyfit(xdata, ydata1, 1)
    ax.plot(x_arr, 10**logb1 * x_arr**m1, '--', color='slategrey')

    m2, logb2 = np.polyfit(xdata, ydata2, 1)
    ax.plot(x_arr, 10**logb2 * x_arr**m2, '--', color='violet')

    m3, logb3 = np.polyfit(xdata, ydata3, 1)
    ax.plot(x_arr, 10**logb3 * x_arr**m3, '--', color='limegreen')

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

    ax1.set_xlabel(r'$\mathrm{M_{s;\,all\ bands}}$')
    ax1.set_ylabel(r'$\mathrm{log(M_{s;\,all\ bands})  -  log(M_{s;\,(u)briz}) }$')

    deltamass1 = np.log10(fit_mass_allbands) - np.log10(fit_mass_ubriz)
    deltamass2 = np.log10(fit_mass_allbands) - np.log10(fit_mass_briz)

    ax1.axhline(y=0.0, ls='--', color='deepskyblue', zorder=1)

    dm1_lbl = r'$\mathrm{log(M_{s;\,all}) - log(M_{s;\,ubriz})}$'
    dm2_lbl = r'$\mathrm{log(M_{s;\,all}) - log(M_{s;\,briz})}$'

    ax1.scatter(fit_mass_allbands, deltamass1, s=12, color='darkviolet', zorder=2, label=dm1_lbl)
    ax1.scatter(fit_mass_allbands, deltamass2, s=10, color='forestgreen', zorder=2, label=dm2_lbl)

    ax1.set_xscale('log')

    ax1.legend(fontsize=10, frameon=False)

    fig1.savefig(adap_dir + 'mass_residuals.pdf', dpi=300, bbox_inches='tight')

    return None

def main():

    # Empty lists for storing masses
    fit_mass_allbands = []
    fit_mass_allbands_err = []

    fit_mass_ubriz = []
    fit_mass_ubriz_err = []

    fit_mass_briz = []
    fit_mass_briz_err = []

    for field in ['North', 'South']:

        # Read in catalog from Lou
        if 'North' in field:
            df = pandas.read_pickle(adap_dir + 'GOODS_North_SNeIa_host_phot.pkl')
            key = 'ID'

        elif 'South' in field:
            df = pandas.read_pickle(adap_dir + 'GOODS_South_SNeIa_host_phot.pkl')
            key = 'Seq'

        # Loop over all of our objects
        for i in range(len(df)):

            # Now read in the fitting results and get our stellar masses
            galaxy_seq = df[key][i]

            h5file_allbands = adap_dir + "goodss_param_sfh/all_bands/" + "emcee_" + \
                              field + "_" + str(galaxy_seq) + ".h5"
            h5file_ubriz    = adap_dir + "goodss_param_sfh/ubriz/"     + "emcee_" + \
                              field + "_" + str(galaxy_seq) + ".h5"
            h5file_briz     = adap_dir + "goodss_param_sfh/briz/"      + "emcee_" + \
                              field + "_" + str(galaxy_seq) + ".h5"

            result_all, obs, _ = reader.results_from(h5file_allbands, dangerous=False)
            result_ubriz, obs, _ = reader.results_from(h5file_ubriz, dangerous=False)
            result_briz, obs, _ = reader.results_from(h5file_briz, dangerous=False)

            cq_mass_all = get_cq_mass(result_all)
            cq_mass_ubriz = get_cq_mass(result_ubriz)
            cq_mass_briz = get_cq_mass(result_briz)

            # Append ot plotting arrays
            fit_mass_allbands.append(cq_mass_all[1])
            fit_mass_allbands_lowerr = cq_mass_all[1] - cq_mass_all[0]
            fit_mass_allbands_uperr = cq_mass_all[2] - cq_mass_all[1]
            fit_mass_allbands_err.append([fit_mass_allbands_lowerr, fit_mass_allbands_uperr])

            fit_mass_ubriz.append(cq_mass_ubriz[1])
            fit_mass_ubriz_lowerr = cq_mass_ubriz[1] - cq_mass_ubriz[0]
            fit_mass_ubriz_uperr = cq_mass_ubriz[2] - cq_mass_ubriz[1]
            fit_mass_ubriz_err.append([fit_mass_ubriz_lowerr, fit_mass_ubriz_uperr])

            fit_mass_briz.append(cq_mass_briz[1])
            fit_mass_briz_lowerr = cq_mass_briz[1] - cq_mass_briz[0]
            fit_mass_briz_uperr = cq_mass_briz[2] - cq_mass_briz[1]
            fit_mass_briz_err.append([fit_mass_briz_lowerr, fit_mass_briz_uperr])

    # ---------Convert to numpy arrays and reshape
    fit_mass_allbands = np.array(fit_mass_allbands)
    fit_mass_allbands_err = np.array(fit_mass_allbands_err)
    fit_mass_allbands_err = fit_mass_allbands_err.reshape((2, 66))

    fit_mass_ubriz = np.array(fit_mass_ubriz)
    fit_mass_ubriz_err = np.array(fit_mass_ubriz_err)
    fit_mass_ubriz_err = fit_mass_ubriz_err.reshape((2, 66))

    fit_mass_briz = np.array(fit_mass_briz)
    fit_mass_briz_err = np.array(fit_mass_briz_err)
    fit_mass_briz_err = fit_mass_briz_err.reshape((2, 66))

    # ------------------ histogram and KDE
    #fig = plt.figure()
    #ax = fig.add_subplot(111)

    # ------------------ make residual figure
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    ax1.set_xlabel(r'$\mathrm{log(M_{s;\,all\ bands})}$')
    ax1.set_ylabel(r'$\mathrm{log(M_{s;\,all\ bands})  -  log(M_{s;\,(u)briz}) }$')

    deltamass1 = np.log10(fit_mass_allbands) - np.log10(fit_mass_ubriz)
    deltamass2 = np.log10(fit_mass_allbands) - np.log10(fit_mass_briz)

    ax1.axhline(y=0.0, ls='--', color='deepskyblue', zorder=1)

    dm1_lbl = r'$\mathrm{log(M_{s;\,all}) - log(M_{s;\,ubriz})}$'
    dm2_lbl = r'$\mathrm{log(M_{s;\,all}) - log(M_{s;\,briz})}$'

    ax1.scatter(np.log10(fit_mass_allbands), deltamass1, s=12, color='darkviolet', zorder=2, label=dm1_lbl)
    ax1.scatter(np.log10(fit_mass_allbands), deltamass2, s=10, color='forestgreen', zorder=2, label=dm2_lbl)

    # Fit a line to the points
    x_arr = np.logspace(5.0, 12.5, num=1000, base=10)

    xdata = np.log10(fit_mass_allbands)

    m1, b1 = np.polyfit(xdata, deltamass1, 1)
    ax1.plot(x_arr, b1 + x_arr*m1, '--', color='violet')

    m2, b2 = np.polyfit(xdata, deltamass2, 1)
    ax1.plot(x_arr, b2 + x_arr*m2, '--', color='limegreen')

    ax1.legend(fontsize=10, frameon=False)
    ax1.set_xlim(7.8, 12.2)
    ax1.set_ylim(-1.6, 0.8)

    fig1.savefig(adap_dir + 'mass_residuals.pdf', dpi=300, bbox_inches='tight')

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)