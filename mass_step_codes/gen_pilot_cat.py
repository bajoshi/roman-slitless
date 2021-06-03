import numpy as np
import pandas

import prospect.io.read_results as reader

import corner

import os
import sys

import matplotlib.pyplot as plt

home = os.getenv('HOME')
adap_dir = home + '/Documents/adap2021/'

def get_sfr(result, ms):

    trace = result['chain']
    thin = 5
    trace = trace[:, ::thin, :]

    samples = trace.reshape(trace.shape[0] * trace.shape[1], trace.shape[2])

    cq_age = corner.quantile(x=samples[:, 3], q=[0.16, 0.5, 0.84])
    cq_tau = corner.quantile(x=samples[:, 4], q=[0.16, 0.5, 0.84])

    age = cq_age[1] * 1e9  # Gyr to years
    logtau = cq_tau[1]
    tau = 10**logtau  # I think this is in years

    const = ms / (1 - np.exp(-1 * age/tau))
    prefac = const / tau

    sfr = prefac * np.exp(-1 * age/tau)

    return sfr

def get_cq_mass(result):

    trace = result['chain']
    thin = 5
    trace = trace[:, ::thin, :]

    samples = trace.reshape(trace.shape[0] * trace.shape[1], trace.shape[2])
    cq_mass = corner.quantile(x=samples[:, 0], q=[0.16, 0.5, 0.84])

    return cq_mass

def main():

    gn = pandas.read_pickle(adap_dir + 'GOODS_North_SNeIa_host_phot.pkl')
    gs = pandas.read_pickle(adap_dir + 'GOODS_South_SNeIa_host_phot.pkl')

    dfs = [gn, gs]
    fields = ['North', 'South']

    # Empty arrays for storage
    z   = []
    ms  = []
    sfr = []

    fc = 0
    for df in dfs:

        field = fields[fc]

        for i in range(len(df)):

            # Now read in the fitting results and get our stellar masses
            current_z = df['zbest'][i]
            z.append(current_z)

            obj_ra = df['RA'][i]
            obj_dec = df['DEC'][i]

            if field == 'North':
                galaxy_seq = df['ID'][i]
                h5file = adap_dir + "goods_param_sfh/all_bands/" + \
                                  "emcee_North_" + str(galaxy_seq) + ".h5"
            elif field == 'South':
                galaxy_seq = df['Seq'][i]
                h5file = adap_dir + "goods_param_sfh/all_bands/" + \
                                  "emcee_South_" + str(galaxy_seq) + ".h5"

            result, obs, _ = reader.results_from(h5file, dangerous=False)

            cq_mass = get_cq_mass(result)
            current_ms = cq_mass[1]
            ms.append(current_ms)

            # Compute current SFR
            current_sfr = get_sfr(result, current_ms)

            if not np.isfinite(current_sfr):
                current_sfr = -99.00

            sfr.append(current_sfr)

            print(str(galaxy_seq), "  ", \
                  "{:.7f}".format(obj_ra), "  ", \
                  "{:.7f}".format(obj_dec), "  ", \
                  "{:.3f}".format(current_z), "  ", \
                  "{:.2e}".format(current_ms), "  ", \
                  "{:.2f}".format(current_sfr))

        fc += 1

    # Plot ms vs z
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'$z$', fontsize=15)
    ax.set_ylabel(r'$\mathrm{log(M_s\, [M_\odot])}$', fontsize=15)

    ax.scatter(z, np.log10(ms), marker='.', s=30, color='k')

    fig.savefig(adap_dir + 'ms_vs_z.pdf', dpi=200, bbox_inches='tight')

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)








