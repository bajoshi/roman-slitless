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

    # List for storing redshifts
    redshifts = []

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


            redshifts.append(df['zbest'][i])


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

    # --------
    xdata = np.log10(fit_mass_allbands)
    x_arr = np.arange(5.0, 13.0, 0.01)

    # ------------------ histogram and KDE
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel(r'$\mathrm{log(M_s)}$')
    ax.set_ylabel(r'$\mathrm{Normalized\ Density}$')

    from sklearn.neighbors import KernelDensity
    from sklearn.model_selection import GridSearchCV, LeaveOneOut

    from scipy.stats import gaussian_kde

    xdata_for_kde = xdata[:, None]
    x2 = np.log10(fit_mass_ubriz)[:, None]
    x3 = np.log10(fit_mass_briz)[:, None]
    x_arr_for_kde = x_arr[:, None]

    # ---- get bandwidth estimates
    bandwidths = 10 ** np.linspace(-1, 1, 100)
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths},
                        cv=LeaveOneOut())

    grid.fit(xdata_for_kde)
    bw1 = grid.best_params_['bandwidth']
    grid.fit(x2)
    bw2 = grid.best_params_['bandwidth']
    grid.fit(x3)
    bw3 = grid.best_params_['bandwidth']

    # Now estimate KDEs
    kde1 = KernelDensity(kernel='gaussian', bandwidth=0.25).fit(xdata_for_kde)
    log_dens1 = kde1.score_samples(x_arr_for_kde)
    kde2 = KernelDensity(kernel='gaussian', bandwidth=0.25).fit(x2)
    log_dens2 = kde2.score_samples(x_arr_for_kde)
    kde3 = KernelDensity(kernel='gaussian', bandwidth=0.25).fit(x3)
    log_dens3 = kde3.score_samples(x_arr_for_kde)

    # Plot KDEs
    ax.plot(x_arr, np.exp(log_dens1), color='k', lw=2.5, label='UV-Optical-NIR-MIR', zorder=2)
    ax.plot(x_arr, np.exp(log_dens2), color='mediumblue', lw=1.4, label='ubriz', zorder=1)
    ax.plot(x_arr, np.exp(log_dens3), color='darkturquoise', lw=1.4, label='briz', zorder=1)

    # KDEs using Scipy
    x1_kde = gaussian_kde(xdata)
    ax.plot(x_arr, x1_kde(x_arr), ls='--', color='k', lw=2.5, zorder=2)
    x2_kde = gaussian_kde(np.log10(fit_mass_ubriz))
    ax.plot(x_arr, x2_kde(x_arr), ls='--', color='mediumblue', lw=1.4, zorder=2)
    x3_kde = gaussian_kde(np.log10(fit_mass_briz))
    ax.plot(x_arr, x3_kde(x_arr), ls='--', color='darkturquoise', lw=1.4, zorder=2)

    ax.legend(loc=2, fontsize=10, frameon=False)

    ax.set_xlim(7.5, 12.5)

    fig.savefig(adap_dir + 'mass_dist.pdf', dpi=300, bbox_inches='tight')

    # ------------------ make residual figure
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    ax1.set_xlabel(r'$\mathrm{log(M_{s;\,all\ bands})}$')
    ax1.set_ylabel(r'$\mathrm{log(M_{s;\,all\ bands})  -  log(M_{s;\,(u)briz}) }$')

    deltamass1 = xdata - np.log10(fit_mass_ubriz)
    deltamass2 = xdata - np.log10(fit_mass_briz)

    xdata_err = np.empty((2, 66))
    deltamass1_err = np.empty((2, 66))
    deltamass2_err = np.empty((2, 66))

    for j in range(len(xdata)):
        xd = fit_mass_allbands[j]
        xdl = np.abs(np.log10(1 - fit_mass_allbands_err[0, j]/xd))
        xdu = np.log10(1 + fit_mass_allbands_err[1, j]/xd)
        xdata_err[:, j] = [xdl, xdu]

        val1 = fit_mass_ubriz[j]
        dm1l = np.abs(np.log10(1 - fit_mass_allbands_err[0, j]/xd) + np.log10(1 + fit_mass_ubriz_err[1, j]/val1))
        dm1u = np.log10(1 + fit_mass_allbands_err[1, j]/xd) + np.log10(1 - fit_mass_ubriz_err[0, j]/val1)
        deltamass1_err[:, j] = [dm1l, dm1u]

        val2 = fit_mass_briz[j]
        dm2l = np.abs(np.log10(1 - fit_mass_allbands_err[0, j]/xd) + np.log10(1 + fit_mass_briz_err[1, j]/val2))
        dm2u = np.log10(1 + fit_mass_allbands_err[1, j]/xd) + np.log10(1 - fit_mass_briz_err[0, j]/val2)
        deltamass2_err[:, j] = [dm2l, dm2u]

    ax1.axhline(y=0.0, ls='--', color='k', zorder=1)

    dm1_lbl = r'$\mathrm{log(M_{s;\,all}) - log(M_{s;\,ubriz})}$'
    dm2_lbl = r'$\mathrm{log(M_{s;\,all}) - log(M_{s;\,briz})}$'

    #ax1.errorbar(xdata, deltamass1, xerr=xdata_err, yerr=deltamass1_err,
    #    fmt='o', ms=2.0, elinewidth=1.0, ecolor='mediumblue',
    #    color='mediumblue', zorder=2, label=dm1_lbl)
    #ax1.errorbar(xdata, deltamass2, xerr=xdata_err, yerr=deltamass2_err,
    #    fmt='o', ms=2.0, elinewidth=1.0, ecolor='darkturquoise',
    #    color='darkturquoise', zorder=2, label=dm2_lbl)

    ax1.scatter(xdata, deltamass1, s=12, 
        color='mediumblue', zorder=2, label=dm1_lbl)
    ax1.scatter(xdata, deltamass2, s=10, 
        color='darkturquoise', zorder=2, label=dm2_lbl)

    # Fit a line to the points
    m1, b1 = np.polyfit(xdata, deltamass1, 1)
    m2, b2 = np.polyfit(xdata, deltamass2, 1)

    ax1.plot(x_arr, b1 + x_arr*m1, '--', color='mediumblue')
    ax1.plot(x_arr, b2 + x_arr*m2, '--', color='darkturquoise')

    print("Errors for the points and the line estimate --")

    ax1.legend(fontsize=10, frameon=False)
    ax1.set_xlim(6.8, 12.5)
    ax1.set_ylim(-1.6, 0.8)

    ax1.text(x=0.38, y=0.15, s=r'$\mathrm{Slope}\,=\,$' + "{:.2f}".format(m1), 
        verticalalignment='top', horizontalalignment='left', 
        transform=ax.transAxes, color='mediumblue', size=14)
    ax1.text(x=0.38, y=0.11, s=r'$\mathrm{Slope}\,=\,$' + "{:.2f}".format(m2), 
        verticalalignment='top', horizontalalignment='left', 
        transform=ax.transAxes, color='darkturquoise', size=14)

    fig1.savefig(adap_dir + 'mass_residuals.pdf', dpi=300, bbox_inches='tight')

    # --------------
    # Histograms of measurement significance
    allbands_sig = fit_mass_allbands / np.mean(fit_mass_allbands_err, axis=0)
    ubriz_sig = fit_mass_ubriz / np.mean(fit_mass_ubriz_err, axis=0)
    briz_sig = fit_mass_briz / np.mean(fit_mass_briz_err, axis=0)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)

    # Resahpe data
    allbands_sig_kde = allbands_sig[:, None]
    ubriz_sig_kde = ubriz_sig[:, None]
    briz_sig_kde = briz_sig[:, None]

    xsig = np.arange(0.0, 10.0, 0.01)
    xsig_kde = xsig[:, None]

    # Estimate optimal bandwidth
    # I think I can use the same grid of bandwidths as before
    grid.fit(allbands_sig_kde)
    bw1 = grid.best_params_['bandwidth']

    print("BW1:", bw1)

    # Now estimate KDEs
    kde1 = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(allbands_sig_kde)
    log_dens1 = kde1.score_samples(xsig_kde)
    kde2 = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(ubriz_sig_kde)
    log_dens2 = kde2.score_samples(xsig_kde)
    kde3 = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(briz_sig_kde)
    log_dens3 = kde3.score_samples(xsig_kde)

    # Plot KDEs
    ax2.plot(xsig, np.exp(log_dens1), color='k', lw=2.5, label='UV-Optical-NIR-MIR', zorder=2)
    ax2.plot(xsig, np.exp(log_dens2), color='mediumblue', lw=1.4, label='ubriz', zorder=1)
    ax2.plot(xsig, np.exp(log_dens3), color='darkturquoise', lw=1.4, label='briz', zorder=1)

    plt.show()

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)






