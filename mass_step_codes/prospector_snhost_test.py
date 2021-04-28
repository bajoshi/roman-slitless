import time
import sys
import os

import h5py
import numpy as np
import scipy

import matplotlib.pyplot as plt

import fsps
import sedpy
import prospect
import emcee
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

home = os.getenv('HOME')

def build_obs(mags, snr=10, ldist=10.0, **extras):
    """Build a dictionary of observational data.  In this example
    the data consist of photometry for a single nearby dwarf galaxy
    from Johnson et al. 2013.

    :param snr:
        The S/N to assign to the photometry, since none are reported
        in Johnson et al. 2013

    :param ldist:
        The luminosity distance to assume for translating absolute magnitudes
        into apparent magnitudes.

    :returns obs:
        A dictionary of observational data to use in the fit.
    """

    from prospect.utils.obsutils import fix_obs

    # The obs dictionary, empty for now
    obs = {}

    # These are the names of the relevant filters, 
    # in the same order as the photometric data (see below)
    sdss_griz = ['sdss_{0}0'.format(b) for b in ['g','r','i','z']]
    y = ['decam_Y']
    filternames = sdss_griz + y
    # And here we instantiate the `Filter()` objects using methods in `sedpy`,
    # and put the resultinf list of Filter objects in the "filters" key of the `obs` dictionary
    obs["filters"] = sedpy.observate.load_filters(filternames)

    # Now we store the measured fluxes for a single object, **in the same order as "filters"**
    # In this example we use a row of absolute AB magnitudes from Johnson et al. 2013 (NGC4163)
    # We then turn them into apparent magnitudes based on the supplied `ldist` meta-parameter.
    # You could also, e.g. read from a catalog.
    # The units of the fluxes need to be maggies (Jy/3631) so we will do the conversion here too.
    #M_AB = 
    #dm = 25 + 5.0 * np.log10(ldist)
    #mags = np.array([17.41900063, 16.98089981, 16.8614006, 16.48570061, 16.27280045])
    obs["maggies"] = 10**(-0.4*mags)

    # And now we store the uncertainties (again in units of maggies)
    # In this example we are going to fudge the uncertainties based on the supplied `snr` meta-parameter.
    obs["maggies_unc"] = (1./snr) * obs["maggies"]

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

def build_model(object_redshift=None, ldist=10.0, fixed_metallicity=None, add_duste=False, **extras):
    """Build a prospect.models.SedModel object
    
    :param object_redshift: (optional, default: None)
        If given, produce spectra and observed frame photometry appropriate 
        for this redshift. Otherwise, the redshift will be zero.
        
    :param ldist: (optional, default: 10)
        The luminosity distance (in Mpc) for the model.  Spectra and observed 
        frame (apparent) photometry will be appropriate for this luminosity distance.
        
    :param fixed_metallicity: (optional, default: None)
        If given, fix the model metallicity (:math:`log(Z/Z_sun)`) to the given value.
        
    :param add_duste: (optional, default: False)
        If `True`, add dust emission and associated (fixed) parameters to the model.
        
    :returns model:
        An instance of prospect.models.SedModel
    """

    # Get (a copy of) one of the prepackaged model set dictionaries.
    # This is, somewhat confusingly, a dictionary of dictionaries, keyed by parameter name
    model_params = TemplateLibrary["parametric_sfh"]
    
    # Now add the lumdist parameter by hand as another entry in the dictionary.
    # This will control the distance since we are setting the redshift to zero.  
    # In `build_obs` above we used a distance of 10Mpc to convert from absolute to apparent magnitudes, 
    # so we use that here too, since the `maggies` are appropriate for that distance.
    #model_params["lumdist"] = {"N": 1, "isfree": False, "init": ldist, "units":"Mpc"}

    # Seems like zred is fixed by default for this model (and many other models) so undo that
    #model_params["zred"]['isfree'] = True
    
    # Let's make some changes to initial values appropriate for our objects and data
    #model_params["zred"]["init"] = 0.5
    model_params["mass"]["init"] = 1e10
    model_params["logzsol"]["init"] = -0.5
    model_params["dust2"]["init"] = 0.05
    model_params["tage"]["init"] = 10.0
    model_params["tau"]["init"] = 1.0

    #model_params["zred"]["prior"] = priors.TopHat(mini=0.00001, maxi=3.0)
    model_params["dust2"]["prior"] = priors.TopHat(mini=0.0, maxi=2.0)
    model_params["mass"]["prior"] = priors.LogUniform(mini=1e8, maxi=1e12)

    # If we are going to be using emcee, it is useful to provide a 
    # minimum scale for the cloud of walkers (the default is 0.1)
    model_params["mass"]["disp_floor"] = 1e8
    model_params["tau"]["disp_floor"] = 1.0
    model_params["tage"]["disp_floor"] = 1.0
    
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
    from prospect.sources import CSPSpecBasis
    sps = CSPSpecBasis(zcontinuous=zcontinuous)
    return sps

def main():

    vers = (np.__version__, scipy.__version__, h5py.__version__, fsps.__version__, prospect.__version__)
    print("Numpy: {}\nScipy: {}\nH5PY: {}\nFSPS: {}\nProspect: {}".format(*vers))

    # Read in the ghost rubin catalog
    # It has grizy photometry for host galaxies
    cat = np.genfromtxt(home + '/Desktop/ghost_rubin.csv', names=True, dtype=None, \
    	usecols=(0,57,107,157,207,257,319,325,328,329,330), delimiter=',')

    test_idx = 137

    print(cat.dtype.names)
    print(cat[test_idx])
    obj_z = float(cat['TransientRedshift'][test_idx])
    print("Transient Redshift: ", obj_z)

    gmag = cat['gApMag'][test_idx]
    rmag = cat['rApMag'][test_idx]
    imag = cat['iApMag'][test_idx]
    zmag = cat['zApMag'][test_idx]
    ymag = cat['yApMag'][test_idx]

    mags = np.array([gmag, rmag, imag, zmag, ymag])

    # And we will store some meta-parameters that control the input arguments to this method:
    run_params = {}
    run_params["snr"] = 50.0

    # Build the obs dictionary using the meta-parameters
    obs = build_obs(mags, **run_params)

    # Look at the contents of the obs dictionary
    #print("Obs Dictionary Keys:\n\n{}\n".format(obs.keys()))
    #print("--------\nFilter objects:\n")
    #print(obs["filters"])

    # --- Plot the Data ----
    # This is why we stored these...
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
    fig.tight_layout()

    plt.show()

    # Set priors and params
    #mass_param = {"name": "mass",
    #              # The mass parameter here is a scalar, so it has N=1
    #              "N": 1,
    #              # We will be fitting for the mass, so it is a free parameter
    #              "isfree": True,
    #              # This is the initial value. For fixed parameters this is the
    #              # value that will always be used. 
    #              "init": 1e9,
    #              # This sets the prior probability for the parameter
    #              "prior": priors.LogUniform(mini=1e6, maxi=1e12),
    #              # this sets the initial dispersion to use when generating 
    #              # clouds of emcee "walkers".  It is not required, but can be very helpful.
    #              "init_disp": 1e6,
    #              # this sets the minimum dispersion to use when generating 
    #              #clouds of emcee "walkers".  It is not required, but can be useful if 
    #              # burn-in rounds leave the walker distribution too narrow for some reason.
    #              "disp_floor": 1e6,
    #              # This is not required, but can be helpful
    #              "units": "solar masses formed",
    #              }

    # Look at all the prepackaged parameter sets
    #TemplateLibrary.show_contents()

    # Check a specific model. 
    #TemplateLibrary.describe("continuity_sfh")

    run_params["object_redshift"] = obj_z
    run_params["fixed_metallicity"] = None
    run_params["add_duste"] = True

    model = build_model(**run_params)
    print(model)
    print("\nInitial free parameter vector theta:\n  {}\n".format(model.theta))
    print("Initial parameter dictionary:\n{}".format(model.params))

    run_params["zcontinuous"] = 1

    sps = build_sps(**run_params)
    
    print("\n------------------")
    print("Done building SPS.")
    print("------------------\n")

    # Generate the model SED at the initial value of theta
    theta = model.theta.copy()
    initial_spec, initial_phot, initial_mfrac = model.sed(theta, obs=obs, sps=sps)

    """
    title_text = ','.join(["{}={}".format(p, model.params[p][0]) 
                           for p in model.free_params])
    
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

    # establish bounds
    xmin, xmax = np.min(wphot)*0.8, np.max(wphot)/0.8
    temp = np.interp(np.linspace(xmin,xmax,10000), wspec, initial_spec)
    ymin, ymax = temp.min()*0.8, temp.max()/0.4

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)

    # plot model + data
    ax.loglog(wspec, initial_spec, label='Model spectrum', 
           lw=0.7, color='navy', alpha=0.7)
    ax.errorbar(wphot, initial_phot, label='Model photometry', 
             marker='s',markersize=10, alpha=0.8, ls='', lw=3,
             markerfacecolor='none', markeredgecolor='blue', 
             markeredgewidth=3)
    ax.errorbar(wphot, obs['maggies'], yerr=obs['maggies_unc'], 
             label='Observed photometry',
             marker='o', markersize=10, alpha=0.8, ls='', lw=3,
             ecolor='red', markerfacecolor='none', markeredgecolor='red', 
             markeredgewidth=3)
    ax.set_title(title_text)

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
    #ax.set_ylim([ymin, ymax])
    ax.legend(loc='best', fontsize=20)
    fig.tight_layout()

    plt.show()
    """

    verbose = True
    run_params["verbose"] = verbose

    # Here we will run all our building functions
    obs = build_obs(mags, **run_params)
    sps = build_sps(**run_params)
    model = build_model(**run_params)
    
    # For fsps based sources it is useful to 
    # know which stellar isochrone and spectral library
    # we are using
    print(sps.ssp.libraries)

    # --- start minimization ----
    run_params["dynesty"] = False
    run_params["emcee"] = False
    run_params["optimize"] = True
    run_params["min_method"] = 'lm'
    # We'll start minimization from "nmin" separate places, 
    # the first based on the current values of each parameter and the 
    # rest drawn from the prior.  Starting from these extra draws 
    # can guard against local minima, or problems caused by 
    # starting at the edge of a prior (e.g. dust2=0.0)
    run_params["nmin"] = 2
    
    output = fit_model(obs, model, sps, lnprobfn=lnprobfn, **run_params)
    
    print("Done optmization in {}s".format(output["optimization"][1]))

    print(model.theta)
    (results, topt) = output["optimization"]
    # Find which of the minimizations gave the best result, 
    # and use the parameter vector for that minimization
    ind_best = np.argmin([r.cost for r in results])
    print(ind_best)
    theta_best = results[ind_best].x.copy()
    print(theta_best)

    # Set this to False if you don't want to do another optimization
    # before emcee sampling (but note that the "optimization" entry 
    # in the output dictionary will be (None, 0.) in this case)
    # If set to true then another round of optmization will be performed 
    # before sampling begins and the "optmization" entry of the output
    # will be populated.
    run_params["optimize"] = False
    run_params["emcee"] = True
    run_params["dynesty"] = False
    # Number of emcee walkers
    run_params["nwalkers"] = 200
    # Number of iterations of the MCMC sampling
    run_params["niter"] = 1000
    # Number of iterations in each round of burn-in
    # After each round, the walkers are reinitialized based on the 
    # locations of the highest probablity half of the walkers.
    run_params["nburn"] = [16, 32, 64]

    print("Now running with Emcee.")

    output = fit_model(obs, model, sps, lnprobfn=lnprobfn, **run_params)
    print('done emcee in {0}s'.format(output["sampling"][1]))
    
    hfile = "demo_emcee_mcmc.h5"
    writer.write_hdf5(hfile, run_params, model, obs,
                      output["sampling"][0], output["optimization"][0],
                      tsample=output["sampling"][1],
                      toptimize=output["optimization"][1])
    
    print('Finished')


    # -------------------
    """
    print("Now running with Dynesty.")

    run_params["dynesty"] = True
    run_params["optmization"] = False
    run_params["emcee"] = False
    run_params["nested_method"] = "rwalk"
    run_params["nlive_init"] = 400
    run_params["nlive_batch"] = 200
    run_params["nested_dlogz_init"] = 0.05
    run_params["nested_posterior_thresh"] = 0.05
    run_params["nested_maxcall"] = int(1e7)

    output = fit_model(obs, model, sps, lnprobfn=lnprobfn, **run_params)
    print('done dynesty in {0}s'.format(output["sampling"][1]))

    hfile = "demo_dynesty_mcmc.h5"
    writer.write_hdf5(hfile, run_params, model, obs,
                      output["sampling"][0], output["optimization"][0],
                      tsample=output["sampling"][1],
                      toptimize=output["optimization"][1])
    
    print('Finished')
    """

    # -------------------------
    # Visualizing results

    results_type = "emcee"  # "emcee" | "dynesty"
    # grab results (dictionary), the obs dictionary, and our corresponding models
    # When using parameter files set `dangerous=True`
    result, obs, _ = reader.results_from("demo_{}_mcmc.h5".format(results_type), dangerous=False)
    
    #The following commented lines reconstruct the model and sps object, 
    # if a parameter file continaing the `build_*` methods was saved along with the results
    #model = reader.get_model(result)
    #sps = reader.get_sps(result)
    
    # let's look at what's stored in the `result` dictionary
    print(result.keys())

    if results_type == "emcee":
        chosen = np.random.choice(result["run_params"]["nwalkers"], size=10, replace=False)
        tracefig = reader.traceplot(result, figsize=(10,6), chains=chosen)
    else:
        tracefig = reader.traceplot(result, figsize=(10,6))

    # maximum a posteriori (of the locations visited by the MCMC sampler)
    imax = np.argmax(result['lnprobability'])
    if results_type == "emcee":
        i, j = np.unravel_index(imax, result['lnprobability'].shape)
        theta_max = result['chain'][i, j, :].copy()
        thin = 5
    else:
        theta_max = result["chain"][imax, :]
        thin = 1
    
    #print('Optimization value: {}'.format(theta_best))
    #print('MAP value: {}'.format(theta_max))
    cornerfig = reader.subcorner(result, start=0, thin=thin,
                                 fig=plt.subplots(5,5,figsize=(8,8))[0])

    plt.show()

    return None

if __name__=='__main__':
    main()
    sys.exit(0)