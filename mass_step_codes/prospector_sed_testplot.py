import numpy as np
import pandas
import matplotlib.pyplot as plt

import os
import sys

import prospect.io.read_results as reader

home = os.getenv('HOME')
adap_dir = home + '/Documents/adap2021/'

from prospector_goods_fit import plot_data, build_sps, build_model
from prospect.models.sedmodel import SedModel
from prospect.models.templates import TemplateLibrary

field = 'North'
galaxy_seq = 27438
nsamp = 500  # random samples from the posterior for plotting

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

results_type = 'dynesty'

result, obs, _ = reader.results_from(adap_dir + results_type + "_" + \
                 field + "_" + str(galaxy_seq) + ".h5", dangerous=False)

parnames = np.array(result['theta_labels'])
print('Parameters in this model:', parnames)


# ------------------
from prospect.plotting.utils import sample_posterior

wts = result.get("weights", None)
theta = sample_posterior(result["chain"], weights=wts, nsample=nsamp)

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

# plot uncertainties by plotting randomly selected samples
for i in range(nsamp):
    spec, phot, mfrac = model.mean_model(theta[i], obs, sps=sps)
    ax.plot(wspec, spec, color='gray', alpha=0.002, zorder=1)

# plot filters
ymin, ymax = obs["maggies"].min()*0.8, obs["maggies"].max()/0.4

for f in obs['filters']:
    print(f)
    w, t = f.wavelength.copy(), f.transmission.copy()
    t = t / t.max()
    t = 10**(0.2*(np.log10(ymax/ymin)))*t * ymin
    ax.loglog(w, t, lw=2.0, color='seagreen', alpha=0.6)

ax.legend(loc=0, fontsize=12, frameon=False)

ax.set_xlim(2000, 150000)
ax.set_ylim(ymin, ymax)

ax.set_xscale('log')
ax.set_yscale('log')

fig.savefig(adap_dir + field + str(galaxy_seq) + '_allbands_sed.png', dpi=300, bbox_inches='tight')









