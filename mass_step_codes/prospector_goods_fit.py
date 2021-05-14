import time
import sys
import os

import h5py
import numpy as np
import scipy
import pandas

import matplotlib.pyplot as plt
import matplotlib as mpl

import fsps
import sedpy
import prospect
import emcee
import corner
import dynesty

from prospect.models.sedmodel import SedModel
from prospect.models.templates import TemplateLibrary
from prospect.models import priors
from prospect.likelihood import lnlike_spec, lnlike_phot, write_log
from prospect.likelihood import chi_spec, chi_phot
from prospect.fitting import lnprobfn
from prospect.fitting import fit_model
from prospect.io import write_results as writer
import prospect.io.read_results as reader
from prospect.utils.obsutils import fix_obs

home = os.getenv('HOME')
adap_dir = home + '/Documents/adap2021/'
utils_dir = home + '/Documents/GitHub/roman-slitless/fitting_pipeline/utils/'

sys.path.append(utils_dir)
from convert_to_sci_not import convert_to_sci_not as csn

def build_obs(fluxes, fluxes_unc, bandpasses, **extras):
    """Build a dictionary of observational data. 

    :param fluxes:
        Observed fluxes

    :param bandpasses:
        Bandpasses for the quoted observed fluxes

    :returns obs:
        A dictionary of observational data to use in the fit.
    """

    # The obs dictionary, empty for now
    obs = {}

    # These are the names of the relevant filters, 
    # in the same order as the photometric data (see below)

    # For our data
    """
       'CTIO_U', 'VIMOS_U',
       'ACS_F435W', 'ACS_F606W', 'ACS_F775W', 'ACS_F814W',
       'ACS_F850LP', 'WFC3_F098M', 'WFC3_F105W',
       'WFC3_F125W', 'WFC3_F160W', 'ISAAC_KS', 'HAWKI_KS',
       'IRAC_CH1', 'IRAC_CH2', 'IRAC_CH3', 'IRAC_CH4'
    """

    ground_u = ['decam_u', 'LBT_LBCB_besselU']
    hst = ['acs_wfc_f435w', 'acs_wfc_f606w', 'acs_wfc_f775w', 'acs_wfc_f814w', 'acs_wfc_f850lp',
           'wfc3_ir_f098m', 'wfc3_ir_f105w', 'wfc3_ir_f125w', 'wfc3_ir_f140w', 'wfc3_ir_f160w']
    k_bands = ['HAWKI_Ks', 'Subaru_MOIRCS_K', 'CFHT_Wircam_Ks']
    spitzer = ['spitzer_irac_ch'+n for n in ['1','2','3','4']]

    filternames = ground_u + hst + k_bands + spitzer

    # This code block is necessary to ensure that we
    # handle negative since we're ignoring them, i.e.,
    # need to make sure that we do not load the same
    # filters each time. The filters are only loaded
    # depending on the filters given in useable_filters.
    filternames_toload = []
    for ft in filternames:
        if ft == 'decam_u':
            bp = 'CTIO'
        elif ft == 'LBT_LBCB_besselU':
            bp = 'LBC'
        elif ft == 'HAWKI_Ks':
            bp = 'HAWKI'
        elif ft == 'Subaru_MOIRCS_K':
            bp = 'MOIRCS'
        elif ft == 'CFHT_Wircam_Ks':
            bp = 'CFHT'
        else:
            bp = (ft.split('_')[-1]).upper()
        #print('\nft:', ft)
        for j in range(len(bandpasses)):
            bandpass = bandpasses[j]
            #print('bp and bandpass:', bp, bandpass)

            if bp in bandpass:
                #print("----- Added:", ft)
                filternames_toload.append(ft)
                break

    #print("Filter names to load:", filternames_toload, len(filternames_toload))
    #sys.exit(0)

    # And here we instantiate the `Filter()` objects using methods in `sedpy`,
    # and put the resultinf list of Filter objects in the "filters" key of the `obs` dictionary
    obs["filters"] = sedpy.observate.load_filters(filternames_toload)

    # Now we store the measured fluxes for a single object, **in the same order as "filters"**
    # The units of the fluxes need to be maggies (Jy/3631).
    f = fluxes * 1e-6 / 3631  # Since our fluxes are in micro Janskys
    mags = -2.5 * np.log10(f)
    obs["maggies"] = 10**(-0.4*mags)

    # And now we store the uncertainties (again in units of maggies)
    f_unc = fluxes_unc * 1e-6 / 3631
    maggies_unc = -2.5 * np.log10(f_unc)
    obs["maggies_unc"] = 10**(-0.4*maggies_unc)

    # This is an array of effective wavelengths for each of the filters.  
    # It is not necessary, but it can be useful for plotting so we store it here as a convenience
    obs["phot_wave"] = np.array([f.wave_effective for f in obs["filters"]])

    # We do not have a spectrum, so we set some required elements of the obs dictionary to None.
    # (this would be a vector of vacuum wavelengths in angstroms)
    obs["wavelength"] = None
    # (this would be the spectrum in units of maggies)
    obs["spectrum"] = None
    # (spectral uncertainties are given here)
    obs['unc'] = None
    # (again, to ignore a particular wavelength set the value of the
    #  corresponding elemnt of the mask to *False*)
    obs['mask'] = None

    # This function ensures all required keys are present in the obs dictionary,
    # adding default values if necessary
    obs = fix_obs(obs)

    return obs

def plot_data(obs):

    wphot = obs["phot_wave"]

    # establish bounds
    xmin, xmax = np.min(wphot)*0.8, np.max(wphot)/0.8
    ymin, ymax = obs["maggies"].min()*0.8, obs["maggies"].max()/0.4

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)

    # plot all the data
    ax.plot(wphot, obs['maggies'],
         label='All observed photometry',
         marker='o', markersize=12, alpha=0.8, ls='', lw=3,
         color='slateblue')

    # overplot only the data we intend to fit
    mask = obs["phot_mask"]
    ax.errorbar(wphot[mask], obs['maggies'][mask], 
             yerr=obs['maggies_unc'][mask], 
             label='Photometry to fit',
             marker='o', markersize=8, alpha=0.8, ls='', lw=3,
             ecolor='tomato', markerfacecolor='none', markeredgecolor='tomato', 
             markeredgewidth=3)

    # plot Filters
    for f in obs['filters']:
        w, t = f.wavelength.copy(), f.transmission.copy()
        t = t / t.max()
        t = 10**(0.2*(np.log10(ymax/ymin)))*t * ymin
        ax.loglog(w, t, lw=3, color='gray', alpha=0.7)

    # prettify
    ax.set_xlabel('Wavelength [A]')
    ax.set_ylabel('Flux Density [maggies]')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc='best', fontsize=20)
    fig.tight_layout()

    plt.show()

    return None

def build_model(object_redshift=None, fixed_metallicity=None, add_duste=False, **extras):
    """Build a prospect.models.SedModel object
    
    :param object_redshift: (optional, default: None)
        If given, produce spectra and observed frame photometry appropriate 
        for this redshift. Otherwise, the redshift will be zero.
        
    :param fixed_metallicity: (optional, default: None)
        If given, fix the model metallicity (:math:`log(Z/Z_sun)`) to the given value.
        
    :param add_duste: (optional, default: False)
        If `True`, add dust emission and associated (fixed) parameters to the model.
        
    :returns model:
        An instance of prospect.models.SedModel
    """

    # Get (a copy of) one of the prepackaged model set dictionaries.
    # This is, somewhat confusingly, a dictionary of dictionaries, keyed by parameter name
    model_params = TemplateLibrary["continuity_sfh"]
    #model_params = TemplateLibrary["parametric_sfh"]

    # This will give the stellar mass as the surviving
    # mass which is what we want. Otherwise by default
    # it gives total mass formed.
    #model_params["mass"]["units"] = 'mstar'
    # Set dust type. # Fixed for parametric sfh
    #model_params["dust_type"] = 4

    # Let's make some changes to initial values appropriate for our objects and data
    #model_params["logmass"]["init"] = 10
    #model_params["logzsol"]["init"] = -0.5
    #model_params["dust2"]["init"] = 0.05
    #model_params["mass"]["init"] = 1e10
    #model_params["tage"]["init"] = 4.0
    #model_params["tau"]["init"] = 1.0

    # Choose priors
    # Priors not specified in here are left at default values
    
    #model_params["mass"]["prior"] = priors.LogUniform(mini=1e8, maxi=1e12)
    #model_params["tau"]["prior"] = priors.LogUniform(mini=1e-1, maxi=1e2)

    # If we are going to be using emcee, it is useful to provide a 
    # minimum scale for the cloud of walkers (the default is 0.1)
    #model_params["mass"]["disp_floor"] = 1e8
    #model_params["tau"]["disp_floor"] = 1.0
    #model_params["tage"]["disp_floor"] = 1.0
    
    # Change the model parameter specifications based on some keyword arguments
    if fixed_metallicity is not None:
        # make it a fixed parameter
        model_params["logzsol"]["isfree"] = False
        #And use value supplied by fixed_metallicity keyword
        model_params["logzsol"]['init'] = fixed_metallicity 

    if object_redshift is not None:
        # make sure zred is fixed
        model_params["zred"]['isfree'] = False
        # And set the value to the object_redshift keyword
        model_params["zred"]['init'] = object_redshift

    if add_duste:
        # Add dust emission (with fixed dust SED parameters)
        # Since `model_params` is a dictionary of parameter specifications, 
        # and `TemplateLibrary` returns dictionaries of parameter specifications, 
        # we can just update `model_params` with the parameters described in the 
        # pre-packaged `dust_emission` parameter set.
        model_params.update(TemplateLibrary["dust_emission"])
        
    # Now instantiate the model object using this dictionary of parameter specifications
    model = SedModel(model_params)

    return model

def build_sps(zcontinuous=1, **extras):
    """
    :param zcontinuous: 
        A vlue of 1 insures that we use interpolation between SSPs to 
        have a continuous metallicity parameter (`logzsol`)
        See python-FSPS documentation for details
    """
    from prospect.sources import FastStepBasis
    #from prospect.sources import CSPSpecBasis
    sps = FastStepBasis(zcontinuous=zcontinuous)
    return sps

def main(field, galaxy_seq):

    #vers = (np.__version__, scipy.__version__, h5py.__version__, fsps.__version__, prospect.__version__)
    #print("Numpy: {}\nScipy: {}\nH5PY: {}\nFSPS: {}\nProspect: {}".format(*vers))

    # -------------- Decide field and filters
    # Read in catalog from Lou
    if 'North' in field:
        df = pandas.read_pickle(adap_dir + 'GOODS_North_SNeIa_host_phot.pkl')

        all_filters = ['LBC_U_FLUX',
        'ACS_F435W_FLUX', 'ACS_F606W_FLUX', 'ACS_F775W_FLUX', 'ACS_F814W_FLUX',
        'ACS_F850LP_FLUX', 'WFC3_F105W_FLUX', 'WFC3_F125W_FLUX',
        'WFC3_F140W_FLUX', 'WFC3_F160W_FLUX', 'MOIRCS_K_FLUX', 'CFHT_Ks_FLUX',
        'IRAC_CH1_SCANDELS_FLUX', 'IRAC_CH2_SCANDELS_FLUX', 'IRAC_CH3_FLUX',
        'IRAC_CH4_FLUX']

        #all_filters = ['LBC_U_FLUX', 'ACS_F435W_FLUX', 'ACS_F606W_FLUX', 
        #'ACS_F775W_FLUX', 'ACS_F850LP_FLUX']

        seq = np.array(df['ID'])
        i = int(np.where(seq == galaxy_seq)[0])

    elif 'South' in field:
        df = pandas.read_pickle(adap_dir + 'GOODS_South_SNeIa_host_phot.pkl')

        all_filters = ['CTIO_U_FLUX', 'ACS_F435W_FLUX', 'ACS_F606W_FLUX', 'ACS_F775W_FLUX', 
        'ACS_F814W_FLUX', 'ACS_F850LP_FLUX', 'WFC3_F098M_FLUX', 'WFC3_F105W_FLUX', 
        'WFC3_F125W_FLUX', 'WFC3_F160W_FLUX', 'HAWKI_KS_FLUX',
        'IRAC_CH1_FLUX', 'IRAC_CH2_FLUX', 'IRAC_CH3_FLUX', 'IRAC_CH4_FLUX']

        #all_filters = ['CTIO_U_FLUX', 'ACS_F435W_FLUX', 'ACS_F606W_FLUX', 
        #'ACS_F775W_FLUX', 'ACS_F850LP_FLUX']

        seq = np.array(df['Seq'])
        i = int(np.where(seq == galaxy_seq)[0])

    #print('Read in pickle with the following columns:')
    #print(df.columns)
    #print('Rows in DataFrame:', len(df)

    print("Match index:", i, "for Seq:", galaxy_seq)

    # -------------- Preliminary stuff
    # Set up for emcee
    nwalkers = 1000
    niter = 500

    ndim = 5

    # Other set up
    obj_ra = df['RA'][i]
    obj_dec = df['DEC'][i]

    obj_z = df['zbest'][i]

    # ------------- Get obs data
    fluxes = []
    fluxes_unc = []
    useable_filters = []

    for ft in range(len(all_filters)):
        filter_name = all_filters[ft]

        flux = df[filter_name][i]
        fluxerr = df[filter_name + 'ERR'][i]

        if np.isnan(flux):
            continue

        if flux <= 0.0:
            continue

        if (fluxerr < 0) or np.isnan(fluxerr):
            fluxerr = 0.1 * flux

        fluxes.append(flux)
        fluxes_unc.append(fluxerr)
        useable_filters.append(filter_name)

    #print("\n")
    #print(df.loc[i])
    #print(fluxes, len(fluxes))
    #print(useable_filters, len(useable_filters))

    fluxes = np.array(fluxes)
    fluxes_unc = np.array(fluxes_unc)

    # Now build the prospector observation
    obs = build_obs(fluxes, fluxes_unc, useable_filters)

    # Set params for run
    run_params = {}
    run_params["object_redshift"] = obj_z
    run_params["fixed_metallicity"] = None
    run_params["add_duste"] = True
    #run_params["dust_type"] = 4

    #model = build_model(**run_params)
    #print("\nInitial free parameter vector theta:\n  {}\n".format(model.theta))
    #print("Initial parameter dictionary:\n{}".format(model.params))

    run_params["zcontinuous"] = 1

    # Generate the model SED at the initial value of theta
    #theta = model.theta.copy()
    #initial_spec, initial_phot, initial_mfrac = model.sed(theta, obs=obs, sps=sps)

    verbose = True
    run_params["verbose"] = verbose

    # Here we will run all our building functions
    obs = build_obs(fluxes, fluxes_unc, useable_filters)
    sps = build_sps(**run_params)
    model = build_model(**run_params)

    #plot_data(obs)
    #sys.exit(0)

    # --- start fitting ----
    # Set this to False if you don't want to do another optimization
    # before emcee sampling (but note that the "optimization" entry 
    # in the output dictionary will be (None, 0.) in this case)
    # If set to true then another round of optmization will be performed 
    # before sampling begins and the "optmization" entry of the output
    # will be populated.
    """
    run_params["optimize"] = False
    run_params["min_method"] = 'lm'
    # We'll start minimization from "nmin" separate places, 
    # the first based on the current values of each parameter and the 
    # rest drawn from the prior.  Starting from these extra draws 
    # can guard against local minima, or problems caused by 
    # starting at the edge of a prior (e.g. dust2=0.0)
    run_params["nmin"] = 5

    run_params["emcee"] = True
    run_params["dynesty"] = False
    # Number of emcee walkers
    run_params["nwalkers"] = nwalkers
    # Number of iterations of the MCMC sampling
    run_params["niter"] = niter
    # Number of iterations in each round of burn-in
    # After each round, the walkers are reinitialized based on the 
    # locations of the highest probablity half of the walkers.
    run_params["nburn"] = [8, 16, 32, 64]
    run_params["progress"] = True

    hfile = adap_dir + "emcee_" + field + "_" + str(galaxy_seq) + ".h5"

    if not os.path.isfile(hfile):

        print("Now running with Emcee.")
        output = fit_model(obs, model, sps, lnprobfn=lnprobfn, **run_params)

        print('done emcee in {0}s'.format(output["sampling"][1]))
    
        writer.write_hdf5(hfile, run_params, model, obs,
                          output["sampling"][0], output["optimization"][0],
                          tsample=output["sampling"][1],
                          toptimize=output["optimization"][1])
    
        print('Finished with Seq: ' + str(galaxy_seq))

    """

    print("Now running with Dynesty.")

    run_params["emcee"] = False
    run_params["dynesty"] = True
    run_params["nested_method"] = "rwalk"
    run_params["nlive_init"] = 400
    run_params["nlive_batch"] = 200
    run_params["nested_dlogz_init"] = 0.05
    run_params["nested_posterior_thresh"] = 0.05
    run_params["nested_maxcall"] = int(1e6)

    output = fit_model(obs, model, sps, lnprobfn=lnprobfn, **run_params)
    print('done dynesty in {0}s'.format(output["sampling"][1]))

    hfile = adap_dir + "dynesty_" + field + "_" + str(galaxy_seq) + "_csfh.h5"
    writer.write_hdf5(hfile, run_params, model, obs,
                      output["sampling"][0], output["optimization"][0],
                      tsample=output["sampling"][1],
                      toptimize=output["optimization"][1])

    print('Finished with Seq: ' + str(galaxy_seq))

    # -------------------------
    # Visualizing results

    results_type = "dynesty"  # "emcee" | "dynesty"
    # grab results (dictionary), the obs dictionary, and our corresponding models
    # When using parameter files set `dangerous=True`
    #result, obs, _ = reader.results_from("{}_" + str(galaxy_seq) + \
    #                 ".h5".format(results_type), dangerous=False)
    
    result, obs, _ = reader.results_from(adap_dir + results_type + "_" + \
                     field + "_" + str(galaxy_seq) + ".h5", dangerous=False)

    #The following commented lines reconstruct the model and sps object, 
    # if a parameter file continaing the `build_*` methods was saved along with the results
    #model = reader.get_model(result)
    #sps = reader.get_sps(result)
    
    # let's look at what's stored in the `result` dictionary
    print(result.keys())

    parnames = np.array(result['theta_labels'])
    print('Parameters in this model:', parnames)

    if results_type == "emcee":

        chosen = np.random.choice(result["run_params"]["nwalkers"], size=150, replace=False)
        tracefig = reader.traceplot(result, figsize=(10,6), chains=chosen)

        tracefig.savefig(adap_dir + 'trace_' + str(galaxy_seq) + '.pdf', dpi=200, bbox_inches='tight')

    else:
        tracefig = reader.traceplot(result, figsize=(10,6))
        tracefig.savefig(adap_dir + 'trace_' + str(galaxy_seq) + '.pdf', dpi=200, bbox_inches='tight')

    # Get chain for corner plot
    if results_type == 'emcee':

        trace = result['chain']
        thin = 5
        trace = trace[:, ::thin, :]

        samples = trace.reshape(trace.shape[0] * trace.shape[1], trace.shape[2])

    else:
        samples = result['chain']

    #math_parnames = [r'$\mathrm{log(Z_\odot)}$', r'$\mathrm{dust2}$', 
    #r'$zf_1$', r'$zf_2$', r'$zf_3$', r'$zf_4$', r'$zf_5$', 
    #r'$\mathrm{M_s}$', r'$\mathrm{f_{agn}}$', r'$\mathrm{agn_\tau}$', 
    #r'$\mathrm{dust_{ratio}}$', r'$\mathrm{dust_{index}}$']

    math_parnames = [r'$\mathrm{M_s}$', r'$\mathrm{log(Z_\odot)}$', 
    r'$\mathrm{dust2}$', r'$\mathrm{t_{age}}$', r'$\mathrm{log(\tau)}$']

    # Fix labels for corner plot and        
    # Figure out ranges for corner plot
    corner_range = []
    for d in range(ndim):

        # Get corner estimate and errors
        cq = corner.quantile(x=samples[:, d], q=[0.16, 0.5, 0.84])
        
        low_err = cq[1] - cq[0]
        up_err  = cq[2] - cq[1]

        # Decide the padding for the plot range
        # depending on how large the error is relative 
        # to the central estimate.
        if low_err * 2.5 >= cq[1]:
            sigma_padding_low = 1.2
        else: sigma_padding_low = 3.0

        if up_err * 2.5 >= cq[1]:
            sigma_padding_up = 1.2
        else: sigma_padding_up = 3.0

        low_lim = cq[1] - sigma_padding_low * low_err
        up_lim  = cq[1] + sigma_padding_up  * up_err
        
        corner_range.append((low_lim, up_lim))

        # Print estimate to screen
        if 'mass' in parnames[d]:
            pn  = '{:.3e}'.format(cq[1])
            pnu = '{:.3e}'.format(up_err)
            pnl = '{:.3e}'.format(low_err)
        else:
            pn  = '{:.3f}'.format(cq[1])
            pnu = '{:.3f}'.format(up_err)
            pnl = '{:.3f}'.format(low_err)

        print(parnames[d], ":  ", pn, "+", pnu, "-", pnl)

    # Corner plot
    cornerfig = corner.corner(samples, quantiles=[0.16, 0.5, 0.84], 
        labels=parnames, label_kwargs={"fontsize": 14},
        range=corner_range, smooth=0.5, smooth1d=0.5)

    # loop over all axes *again* and set title
    # because it won't let me set the 
    # format for soem titles separately
    # Looping has to be done twice because corner 
    # plotting has to be done to get the figure.
    corner_axes = np.array(cornerfig.axes).reshape((ndim, ndim))

    for d in range(ndim):
        # Get corner estimate and errors
        cq = corner.quantile(x=samples[:, d], q=[0.16, 0.5, 0.84])
        
        low_err = cq[1] - cq[0]
        up_err  = cq[2] - cq[1]

        ax_c = corner_axes[d, d]

        if 'mass' in parnames[d]:
            ax_c.set_title(math_parnames[d] + r"$ \, =\,$" + csn(cq[1], sigfigs=3) + \
            r"$\substack{+$" + csn(up_err, sigfigs=3) + r"$\\ -$" + \
            csn(low_err, sigfigs=3) + r"$}$", fontsize=11, pad=15)
        else:
            ax_c.set_title(math_parnames[d] + r"$ \, =\,$" + r"${:.3f}$".format(cq[1]) + \
            r"$\substack{+$" + r"${:.3f}$".format(up_err) + r"$\\ -$" + \
            r"${:.3f}$".format(low_err) + r"$}$", fontsize=11, pad=10)

    cornerfig.savefig(adap_dir + 'corner_' + str(galaxy_seq) + '.pdf', dpi=200, bbox_inches='tight')

    # maximum a posteriori (of the locations visited by the MCMC sampler)
    pmax = np.argmax(result['lnprobability'])
    if results_type == "emcee":
        p, q = np.unravel_index(pmax, result['lnprobability'].shape)
        theta_max = result['chain'][p, q, :].copy()
    else:
        theta_max = result["chain"][pmax, :]

    #print('Optimization value: {}'.format(theta_best))
    #print('MAP value: {}'.format(theta_max))

    # make SED plot for MAP model and some random model
    # randomly chosen parameters from chain
    randint = np.random.randint
    if results_type == "emcee":
        theta = result['chain'][randint(nwalkers), randint(niter)]
    else:
        theta = result["chain"][randint(len(result["chain"]))]

    # generate models
    a = 1.0 + model.params.get('zred', 0.0) # cosmological redshifting
    # photometric effective wavelengths
    wphot = obs["phot_wave"]
    # spectroscopic wavelengths
    if obs["wavelength"] is None:
        # *restframe* spectral wavelengths, since obs["wavelength"] is None
        wspec = sps.wavelengths
        wspec *= a #redshift them
    else:
        wspec = obs["wavelength"]

    # sps = reader.get_sps(result)  # this works if using parameter files
    mspec, mphot, mextra = model.mean_model(theta, obs, sps=sps)
    mspec_map, mphot_map, _ = model.mean_model(theta_max, obs, sps=sps)

    # establish bounds
    xmin, xmax = np.min(wphot)*0.8, np.max(wphot)/0.8
    ymin, ymax = obs["maggies"].min()*0.8, obs["maggies"].max()/0.4

    # Make plot of data and model
    fig3 = plt.figure(figsize=(9,4))
    ax3 = fig3.add_subplot(111)

    ax3.set_xlabel(r'$\mathrm{\lambda\ [\AA]}$', fontsize=15)
    #ax3.set_ylabel(r'$\mathrm{f_\lambda\ [erg\, s^{-1}\, cm^{-2}\, \AA^{-1}]}$', fontsize=15)
    ax3.set_ylabel(r'$\mathrm{Flux\ Density\ [maggies]}$', fontsize=15)

    ax3.loglog(wspec, mspec, label='Model spectrum (random draw)',
           lw=0.7, color='navy', alpha=0.7)
    ax3.loglog(wspec, mspec_map, label='Model spectrum (MAP)',
           lw=0.7, color='green', alpha=0.7)
    ax3.errorbar(wphot, mphot, label='Model photometry (random draw)',
             marker='s', markersize=10, alpha=0.8, ls='', lw=3, 
             markerfacecolor='none', markeredgecolor='blue', 
             markeredgewidth=3)
    ax3.errorbar(wphot, mphot_map, label='Model photometry (MAP)',
             marker='s', markersize=10, alpha=0.8, ls='', lw=3, 
             markerfacecolor='none', markeredgecolor='green', 
             markeredgewidth=3)
    ax3.errorbar(wphot, obs['maggies'], yerr=obs['maggies_unc'], 
             label='Observed photometry', ecolor='red', 
             marker='o', markersize=10, ls='', lw=3, alpha=0.8, 
             markerfacecolor='none', markeredgecolor='red', 
             markeredgewidth=3)

    # plot transmission curves
    for f in obs['filters']:
        w, t = f.wavelength.copy(), f.transmission.copy()
        t = t / t.max()
        t = 10**(0.2*(np.log10(ymax/ymin)))*t * ymin
        ax3.loglog(w, t, lw=3, color='gray', alpha=0.7)

    ax3.set_xlim([xmin, xmax])
    ax3.set_ylim([ymin, ymax])
    ax3.legend(loc='best', fontsize=11)

    fig3.savefig(adap_dir + 'sedplot_' + str(galaxy_seq) + '.pdf', dpi=200, bbox_inches='tight')

    plt.clf()
    plt.cla()
    plt.close()

    return None

if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
    sys.exit(0)






