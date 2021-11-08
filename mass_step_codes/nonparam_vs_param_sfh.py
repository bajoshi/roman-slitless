import numpy as np
import matplotlib.pyplot as plt

import os
import sys

import corner

import prospect.models.transforms as pt
import prospect.io.read_results as reader

home = os.getenv('HOME')
adap_dir = home + '/Documents/Proposals/ADAP/adap2021/'

field = 'North'
galaxy_seq = 27438
obj_z = 1.557
galaxy_age = 10**4.06

results_type = "dynesty" 
result, obs, _ = reader.results_from(adap_dir + results_type + "_" + \
                 field + "_" + str(galaxy_seq) + ".h5", dangerous=False)

# ----------- non param
nagebins = 6
agebins = np.array([[ 0.        ,  8.        ],
                    [ 8.        ,  8.47712125],
                    [ 8.47712125,  9.        ],
                    [ 9.        ,  9.47712125],
                    [ 9.47712125,  9.77815125],
                    [ 9.77815125, 10.13353891]])

samples = result['chain']

# Get the zfractions from corner quantiles
zf1 = corner.quantile(samples[:, 2], q=[0.16, 0.5, 0.84])
zf2 = corner.quantile(samples[:, 3], q=[0.16, 0.5, 0.84])
zf3 = corner.quantile(samples[:, 4], q=[0.16, 0.5, 0.84])
zf4 = corner.quantile(samples[:, 5], q=[0.16, 0.5, 0.84])
zf5 = corner.quantile(samples[:, 6], q=[0.16, 0.5, 0.84])

zf_arr = np.array([zf1[1], zf2[1], zf3[1], zf4[1], zf5[1]])

cq_mass = corner.quantile(samples[:, 7], q=[0.16, 0.5, 0.84])

new_agebins = pt.zred_to_agebins(zred=obj_z, agebins=agebins)
x_agebins = 10**new_agebins# / 1e9

# now convert to sfh and its errors
sfr = pt.zfrac_to_sfr(total_mass=cq_mass[1], z_fraction=zf_arr, agebins=new_agebins)

# ----------- plot
fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(111)

ax.set_xlabel(r'$\mathrm{Time\, [yr];\, since\ galaxy\ formation}$', fontsize=20)
#ax.xaxis.set_label_coords(x=1.2,y=-0.06)
ax.set_ylabel(r'$\mathrm{SFR\, [M_\odot/yr]}$', fontsize=20)

for a in range(nagebins):
    #ax.plot(10**new_agebins[a], np.ones(len(new_agebins[a])) * sfr[a], color='mediumblue', lw=3.5)

    if a == 0:
        ax.plot(x_agebins[a], np.ones(len(x_agebins[a])) * sfr[a], color='mediumblue', lw=3.0, label='Non-parametric SFH')
    else:
        ax.plot(x_agebins[a], np.ones(len(x_agebins[a])) * sfr[a], color='mediumblue', lw=3.0)

    # put in some poisson errors
    sfr_err = np.ones(len(x_agebins[a])) * np.sqrt(sfr[a])
    sfr_plt = np.ones(len(x_agebins[a])) * sfr[a]
    sfr_low_fill = sfr_plt - sfr_err
    sfr_up_fill = sfr_plt + sfr_err
    ax.fill_between(x_agebins[a], sfr_low_fill, sfr_up_fill, 
        color='gray', alpha=0.6)

#ax.set_xlim(0.0, 10.1335)

# --------- param stuff
del result

results_type = "emcee" 
result, obs, _ = reader.results_from(adap_dir + results_type + "_" + \
                 field + "_" + str(galaxy_seq) + ".h5", dangerous=False)

trace = result['chain']
thin = 5
trace = trace[:, ::thin, :]
samples = trace.reshape(trace.shape[0] * trace.shape[1], trace.shape[2])

logtau = corner.quantile(samples[:, -1], q=[0.16, 0.5, 0.84])
print("Log tau:", logtau)
tau = 10**logtau[1]
print("Tau:", tau)

logt = np.arange(0.001, 9.609, 0.001)
t = 10**logt

A = cq_mass[1] / tau**2
sfr = A * t * np.exp(-1 * t / tau)

ax.plot(t, sfr, color='k', label='Parametric SFH; Delayed tau')

ax.legend(loc='upper right', fontsize=15, frameon=False)

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlim(1e5, 5e9)
ax.set_ylim(5, 2e4)

#plt.show()
fig.savefig(adap_dir +'nonparam_vs_param_sfh.pdf', dpi=300, bbox_inches='tight')





