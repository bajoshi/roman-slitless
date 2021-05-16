import numpy as np
import pandas
import matplotlib.pyplot as plt

import os
import sys

import corner
import prospect.io.read_results as reader

from prospector_goods_fit import plot_data
from prospect.models.sedmodel import SedModel
from prospect.models.templates import TemplateLibrary

home = os.getenv('HOME')
adap_dir = home + '/Documents/adap2021/'
utils_dir = home + '/Documents/GitHub/roman-slitless/fitting_pipeline/utils/'

sys.path.append(utils_dir)
from convert_to_sci_not import convert_to_sci_not as csn

field = 'South'
galaxy_seq = 3456
nsamp = 500  # random samples from the posterior for plotting

def build_model(object_redshift=None, fixed_metallicity=None, add_duste=False, **extras):

    # Get (a copy of) one of the prepackaged model set dictionaries.
    # This is, somewhat confusingly, a dictionary of dictionaries, keyed by parameter name
    model_params = TemplateLibrary["parametric_sfh"]
    
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
        model_params.update(TemplateLibrary["dust_emission"])
        
    # Now instantiate the model object using this dictionary of parameter specifications
    model = SedModel(model_params)

    return model

def build_sps(zcontinuous=1, **extras):
    from prospect.sources import CSPSpecBasis
    sps = CSPSpecBasis(zcontinuous=zcontinuous)
    return sps

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

    seq = np.array(df['ID'])
    i = int(np.where(seq == galaxy_seq)[0])

elif 'South' in field:
    df = pandas.read_pickle(adap_dir + 'GOODS_South_SNeIa_host_phot.pkl')

    all_filters = ['CTIO_U_FLUX', 'ACS_F435W_FLUX', 'ACS_F606W_FLUX', 'ACS_F775W_FLUX', 
    'ACS_F814W_FLUX', 'ACS_F850LP_FLUX', 'WFC3_F098M_FLUX', 'WFC3_F105W_FLUX', 
    'WFC3_F125W_FLUX', 'WFC3_F160W_FLUX', 'HAWKI_KS_FLUX',
    'IRAC_CH1_FLUX', 'IRAC_CH2_FLUX', 'IRAC_CH3_FLUX', 'IRAC_CH4_FLUX']

    seq = np.array(df['Seq'])
    i = int(np.where(seq == galaxy_seq)[0])

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

fluxes = np.array(fluxes)
fluxes_unc = np.array(fluxes_unc)


# Set params for run
run_params = {}
run_params["object_redshift"] = obj_z
run_params["fixed_metallicity"] = None
run_params["add_duste"] = True

run_params["zcontinuous"] = 1

verbose = True
run_params["verbose"] = verbose

# Here we will run all our building functions
sps = build_sps(**run_params)
model = build_model(**run_params)

#plot_data(obs)
#sys.exit(0)

results_type = 'emcee'

result, obs, _ = reader.results_from(adap_dir + results_type + "_" + \
                 field + "_" + str(galaxy_seq) + ".h5", dangerous=False)

parnames = np.array(result['theta_labels'])
print('Parameters in this model:', parnames)


# ------------------
from prospect.plotting.utils import sample_posterior

wts = result.get("weights", None)
theta = sample_posterior(result["chain"], weights=wts, nsample=nsamp)

"""
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

ax1.set_xlabel(r'$\mathrm{Time\, [Gyr]}$', fontsize=13)
ax1.set_ylabel(r'$\mathrm{SFR\, [M_\odot/yr]}$', fontsize=13)

t = np.arange(0.0, 14.0, 0.001)

tau_arr = theta[:, -1]

mid_tau = np.mean(tau_arr)
min_tau = np.min(tau_arr)
max_tau = np.max(tau_arr)

sfr = t * np.exp(-1 * t/mid_tau)
sfr_min = t * np.exp(-1 * t/min_tau)
sfr_max = t * np.exp(-1 * t/max_tau)

ax1.plot(t, sfr, color='k', lw=2.5)
ax1.fill_between(t, sfr_min, sfr_max, color='gray', alpha=0.5)

fig1.savefig(adap_dir + 'sfh_' + field + str(galaxy_seq) + '.png', dpi=150, bbox_inches='tight')
"""

# -------------
fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(111)

ax.set_xlabel(r'$\mathrm{\lambda\ [\AA]}$', fontsize=15)
ax.set_ylabel(r'$\mathrm{Flux\ Density\ [maggies]}$', fontsize=15)

# generate models
a = 1.0 + obj_z # cosmological redshifting
# photometric effective wavelengths
wphot = obs["phot_wave"]
# spectroscopic wavelengths
if obs["wavelength"] is None:
    # *restframe* spectral wavelengths, since obs["wavelength"] is None
    wspec = sps.wavelengths
    wspec *= a #redshift them
else:
    wspec = obs["wavelength"]

# plot obs data
ax.errorbar(wphot, obs['maggies'], yerr=obs['maggies_unc'], 
         label=r'$\mathrm{Observed\ photometry}$', ecolor='tab:red', 
         marker='o', markersize=10, ls='', lw=1.5, alpha=0.8, 
         markerfacecolor='none', markeredgecolor='tab:red', 
         markeredgewidth=2.5)

# maximum a posteriori (of the locations visited by the MCMC sampler)
pmax = np.argmax(result['lnprobability'])
if results_type == "emcee":
    p, q = np.unravel_index(pmax, result['lnprobability'].shape)
    theta_max = result['chain'][p, q, :].copy()
else:
    theta_max = result["chain"][pmax, :]

mspec_map, mphot_map, _ = model.mean_model(theta_max, obs, sps=sps)

# plot map spec
ax.plot(wspec, mspec_map, lw=1.5, color='k', label=r'$\mathrm{MAP\ model}$', zorder=2)
ax.errorbar(wphot, mphot_map, label=r'$\mathrm{MAP\ model\ photometry}$',
         marker='o', markersize=10, alpha=0.8, ls='', lw=1.5, 
         markerfacecolor='none', markeredgecolor='tab:blue', 
         markeredgewidth=2.5)

# plot filters
ymin, ymax = obs["maggies"].min()*0.75, obs["maggies"].max()/0.1

for f in obs['filters']:
    print(f, f.name)

    w, t = f.wavelength.copy(), f.transmission.copy()
    t = t / t.max()
    t = 10**(0.2*(np.log10(ymax/ymin)))*t * ymin
    ax.loglog(w, t, lw=2.0, color='seagreen', alpha=0.6)

# plot uncertainties by plotting randomly selected samples
for i in range(nsamp):
    spec, phot, mfrac = model.mean_model(theta[i], obs, sps=sps)
    ax.plot(wspec, spec, color='gray', alpha=0.002, zorder=1)

# Put in inferred stellar mass in the plot
# Get the samples in the correct shape first
if results_type == 'emcee':
    trace = result['chain']
    thin = 5
    trace = trace[:, ::thin, :]
    samples = trace.reshape(trace.shape[0] * trace.shape[1], trace.shape[2])
else:
    samples = result['chain']
cq_mass = corner.quantile(samples[:, 0], q=[0.16, 0.5, 0.84])
print("Corner mass:", cq_mass)

low_err = cq_mass[1] - cq_mass[0]
up_err  = cq_mass[2] - cq_mass[1]

mass_str = r'$\mathrm{M_s}$' + r"$ \, =\,$" + csn(cq_mass[1], sigfigs=3) + \
           r"$\substack{+$" + csn(up_err, sigfigs=3) + r"$\\ -$" + \
           csn(low_err, sigfigs=3) + r"$}$"

ax.text(x=0.55, y=0.42, s=mass_str, 
    verticalalignment='top', horizontalalignment='left', 
    transform=ax.transAxes, color='k', size=14)

ax.legend(loc=0, fontsize=12, frameon=False)

ax.set_xlim(3000, 150000)
ax.set_ylim(ymin, ymax)

ax.set_xscale('log')
ax.set_yscale('log')

fig.savefig(adap_dir + field + str(galaxy_seq) + '_allbands_sed.png', dpi=300, bbox_inches='tight')









