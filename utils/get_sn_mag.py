import numpy as np

from lc_plot import read_lc
from scipy.interpolate import griddata
# from kcorr import get_kcorr_Hogg

import glob
import pickle
import os
import sys
import pdb
from tqdm import tqdm
import matplotlib.pyplot as plt

from astropy.cosmology import FlatLambdaCDM

home = os.getenv('HOME')
dtd_dir = home + '/Documents/GitHub/mass-step/dtd/'

# Define cosmology
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)


# Copies of functions from Lou's control time code
# I'm copying them here to avoid contaminating my namespace
# was having issues with os...
def get_best_rest_filter(dict_keys, ofilter_cen, redshift):
    """
    Will return the filter key closest to deredshifted obs filt.
    Filter keys are assumed to be central wav in nm.
    Assumes given obs filt cen is in nm.

    Modified func by BAJ. THe old way was a one liner:
    min(dict_keys, key=lambda x: abs(x - ofilter_cen/(1+redshift)))

    This func does the same thing but I think is clearer.

    If we need to test the old method and this method do
    the following -- define both as functions, define redshift,
    define the filter dictionary, and run the following for loop:
    for ofilter_cen in range(100, 2000, 100):
        i1 = get_best_rest_filter(filter_dict.keys(), ofilter_cen, redshift)
        i2 = old_way(filter_dict.keys(), ofilter_cen, redshift)
        assert i1 == i2
        print(ofilter_cen, ofilter_cen/(1+redshift), i1, i2)
    """

    diff = np.zeros(len(dict_keys))

    ofilter_rest = ofilter_cen / (1 + redshift)

    for count, key in enumerate(dict_keys):
        diff[count] = np.abs(key - ofilter_rest)

    keylist = list(dict_keys)
    best_rest_filter_cen = keylist[np.argmin(diff)]

    return best_rest_filter_cen


def get_central_wavelength(filter_file):
    """
    Assumes it is given a file name which is a plain text file.
    Assumes two columns in file -- wavelength array in col 0 and
    throughput in col 1.
    Wavelength required in angstroms.

    filter_file should be full path

    Will return central wavelength in nm.
    """

    filter_data = np.loadtxt(filter_file)

    fit_x = np.arange(np.min(filter_data[:, 0]) - 250.,
                      np.max(filter_data[:, 0]) + 250., 50.)

    fit_y = griddata(points=filter_data[:, 0], values=filter_data[:, 1],
                     xi=fit_x)

    elam = int(np.nansum(fit_y*fit_x) / np.nansum(fit_y) / 10.)

    return elam


def rest_frame_Ia_lightcurve(models_dir, throughputs_dir, dstep=3):

    rest_age = np.arange(-20, 120, dstep)
    mag_dict = {}

    for model in glob.glob(models_dir + 'vectors_?.dat'):

        data = np.loadtxt(model)
        yy = griddata(points=data[:, 0], values=data[:, 1], xi=rest_age)
        filt = os.path.basename(model).split('_')[1][0]
        filt_name = throughputs_dir + 'Bessell_' + filt + '.txt'
        elam = get_central_wavelength(filt_name)
        mag_dict[elam] = yy

    return (rest_age, mag_dict)


def kcor(f1, f2, models_used_dict, best_age,
         redshift, vega_spec, extrapolated=True):

    kcor = []

    idx = np.where((vega_spec[:, 0] >= np.min(f1[:, 0])) &
                   (vega_spec[:, 0] <= np.max(f1[:, 0])))
    restf1 = griddata(points=f1[:, 0], values=f1[:, 1],
                      xi=vega_spec[idx][:, 0])

    synth_vega = (np.sum(vega_spec[idx][:, 0]
                         * np.array(restf1) * vega_spec[idx][:, 1]) *
                  np.nanmean(np.diff(vega_spec[idx][:, 0])))

    idx = np.where((vega_spec[:, 0] >= np.min(f2[:, 0])) &
                   (vega_spec[:, 0] <= np.max(f2[:, 0])))
    restf2 = griddata(points=f2[:, 0], values=f2[:, 1],
                      xi=vega_spec[idx][:, 0])

    nearest_vega = (np.sum(vega_spec[idx][:, 0]
                           * np.array(restf2) * vega_spec[idx][:, 1]) *
                    np.nanmean(np.diff(vega_spec[idx][:, 0])))

    # now sn spectrum
    for model in models_used_dict.keys():
        spec = models_used_dict[model]
        idx = np.where((np.abs(spec[:, 0]-best_age) ==
                        np.min(np.abs(spec[:, 0]-best_age))) &
                       (np.abs(spec[:, 0]-best_age) < 3.))
        if (len(idx[0]) == 0.0) or (np.sum(spec[idx][:, 2]) == 0.0):
            continue

        if extrapolated:
            # extrapolated spectrum method
            wave_plus = np.arange(spec[idx][:, 1][-1], 30000., 10.)
            wave_minus = np.arange(1000., spec[idx][:, 1][-1], 10.)
            anchored_x = np.array([1000.]+list(spec[idx][:, 1])+[30000.])
            anchored_y = np.array([0.]+list(spec[idx][:, 2])+[0.])
            counts_plus = griddata(points=anchored_x, values=anchored_y,
                                   xi=wave_plus)
            counts_minus = griddata(points=anchored_x, values=anchored_y,
                                    xi=wave_minus)
            xx = np.array(list(wave_minus) +
                          list(spec[idx][:, 1]) +
                          list(wave_plus))
            yy = np.array(list(counts_minus) +
                          list(spec[idx][:, 2]) +
                          list(counts_plus))
            xx, yy = zip(*sorted(zip(xx, yy)))
            xx, yy = np.array(xx), np.array(yy)

            idx2 = np.where((xx >= np.min(f1[:, 0]/(1+redshift))) &
                            (xx <= np.max(f1[:, 0] / (1+redshift))))
            restf1 = griddata(points=f1[:, 0]/(1+redshift),
                              values=f1[:, 1], xi=xx[idx2])
            synth_obs = (np.sum(xx[idx2] * np.array(restf1)*yy[idx2]) *
                         np.nanmean(np.diff(xx[idx2])))

            idx2 = np.where((xx >= np.min(f2[:, 0])) &
                            (xx <= np.max(f2[:, 0])))
            restf2 = griddata(points=f2[:, 0], values=f2[:, 1], xi=xx[idx2])
            nearest_obs = (np.sum(xx[idx2] * np.array(restf2)*yy[idx2]) *
                           np.nanmean(np.diff(xx[idx2])))

        else:
            # reduce the computation by only working with
            # wavelengths that are defined in filter throughputs
            # this would work fine, except at redshifts where the
            # observed filter does not overlap the template spectra
            idx2 = np.where((spec[idx][:, 1] >=
                             np.min(f1[:, 0]/(1+redshift)))
                            & (spec[idx][:, 1] <=
                               np.max(f1[:, 0]/(1+redshift))))
            restf1 = griddata(points=f1[:, 0]/(1+redshift),
                              values=f1[:, 1], xi=spec[idx][idx2][:, 1])
            synth_obs = (np.sum(spec[idx][idx2][:, 1] * np.array(restf1) *
                                spec[idx][idx2][:, 2]) *
                         np.nanmean(np.diff(spec[idx][idx2][:, 1])))

            idx2 = np.where((spec[idx][:, 1] >= np.min(f2[:, 0])) &
                            (spec[idx][:, 1] <= np.max(f2[:, 0])))
            restf2 = griddata(points=f2[:, 0], values=f2[:, 1],
                              xi=spec[idx][idx2][:, 1])
            nearest_obs = (np.sum(spec[idx][idx2][:, 1] * np.array(restf2)
                                  * spec[idx][idx2][:, 2]) *
                           np.nanmean(np.diff(spec[idx][idx2][:, 1])))

        try:
            kc = -1*(2.5 * np.log10(synth_obs/nearest_obs) -
                     2.5 * np.log10(synth_vega/nearest_vega))
        except:  # noqa
            # pdb.set_trace()
            kc = float('Nan')
        # if synth_obs>0.0:
        #     kc = -1*(2.5 * np.log10(synth_obs/nearest_obs) -
        #              2.5 * np.log10(synth_vega/nearest_vega))
        # else:
        #     kc = float('Nan')
        kcor.append(kc)

    if not kcor:
        result = (float('Nan'), float('Nan'))
    elif len(kcor) == 1 and kcor[0] != kcor[0]:
        result = (float('Nan'), float('Nan'))
    else:
        try:
            result = (np.nanmean(kcor), np.nanstd(kcor))
        except RuntimeWarning:
            print(result)
            pdb.set_trace()

    return result


def get_restframe_mag(phase, band, lc_dir):

    lc_phase, lc_absmag = read_lc(lc_dir, band)

    phase_idx = np.argmin(abs(lc_phase - phase))
    abs_mag = lc_absmag[phase_idx]

    return abs_mag


def get_distance_modulus(redshift):

    dl = cosmo.luminosity_distance(redshift).value  # in Mpc
    dl *= 1e6  # convert to parsecs

    dm = 5 * np.log10(dl/10)

    return dm


def get_sn_mag_F106(phase, redshift, utils_dir=None):

    # -----------
    # Set directories
    if utils_dir is None:
        utils_dir = os.getcwd()

    lc_dir = utils_dir + 'light_curves/'
    # templates_dir = utils_dir + 'templates/'
    throughputs_dir = utils_dir + 'throughputs/'

    # ----------------------------
    # Apply a K correction to get to the apparent
    # magnitude in F106
    # speed_of_light_ang = 3e18  # angstroms per second

    # ---------
    # Read in the SALT2 template
    # Both ways below are identical, i.e., reading the
    # file with genfromtxt or readign the pickle file
    """
    # ------- SN Ia spectrum from Lou
    salt2_spec = np.genfromtxt(templates_dir + "salt2_template_0.txt",
                               dtype=None, names=['day', 'lam', 'llam'],
                               encoding='ascii')

    # Get spectrum corresponding to required phase
    day_idx = np.where(salt2_spec['day'] == int(phase))[0]

    spec_lam = salt2_spec['lam'][day_idx]
    spec_llam = salt2_spec['llam'][day_idx]

    # Convert to l_nu and nu
    spec_nu = speed_of_light_ang / spec_lam
    spec_lnu = spec_lam**2 * spec_llam / speed_of_light_ang
    """

    model_pkl = dtd_dir + 'templates/SEDs_ia.pkl'
    pkl_file = open(model_pkl, 'rb')
    models_used_dict = pickle.load(pkl_file)
    pkl_file.close()

    # ---------
    # Read in the two required filter curves
    # While the column label says transmission
    # it is actually the throughput
    f105 = np.genfromtxt(throughputs_dir + 'F105W_IR_throughput.txt',
                         dtype=None, names=['wav', 'trans'],
                         encoding='ascii')

    bessel_b = np.genfromtxt(throughputs_dir + 'Bessell_B.txt',
                             dtype=None, names=['wav', 'trans'],
                             encoding='ascii')
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(bessel_b['wav'], bessel_b['trans'])
    fig.savefig('bessell_B.png')
    sys.exit(0)
    """

    # ---------
    # Now get the K-correction and distance modulus
    # kcorr = get_kcorr_Hogg(spec_lnu, spec_nu, redshift, bessel_b, f105)
    best_age = phase
    # read in filters
    f105_filter_path = throughputs_dir + 'F105W_IR_throughput.txt'
    ofilter_cen = get_central_wavelength(f105_filter_path)

    filter_dict = {}
    for bessel_filter in glob.glob(throughputs_dir + 'Bessell*.txt'):
        elam = get_central_wavelength(bessel_filter)
        filter_dict[elam] = bessel_filter

    rest_age, rflc = rest_frame_Ia_lightcurve(lc_dir, throughputs_dir)
    best_rest_filter = get_best_rest_filter(rflc.keys(), ofilter_cen,
                                            redshift)

    # f2 is best restframe matched filter
    f1 = np.loadtxt(f105_filter_path)
    f2 = np.loadtxt(filter_dict[best_rest_filter])

    # kcor also needs vega spectrum
    if redshift > 1.5:
        vega_spec = np.loadtxt(dtd_dir + 'templates/vega_model.dat')
    else:
        vega_spec = np.loadtxt(dtd_dir + 'templates/vega_model_mod.dat')

    kcorr = kcor(f1, f2, models_used_dict, best_age,
                 redshift, vega_spec)

    # Get distance modulus based on cosmology
    dist_mod = get_distance_modulus(redshift)
    # print('Distance mod:', dist_mod)

    # Get the absolute magnitude for
    # the chosen band at the given phase
    abs_band_base = os.path.basename(filter_dict[best_rest_filter])
    abs_band = abs_band_base.split('_')[1][0]
    abs_mag = get_restframe_mag(phase, abs_band, lc_dir)

    sn_mag_f106 = abs_mag + dist_mod + kcorr[0]
    # print('\n===========')
    # print(phase)
    # print(abs_mag)
    # print(dist_mod)
    # print(kcorr[0])
    # print(sn_mag_f106)

    # Delete numpy record arrays that have been read in
    # del salt2_spec
    del vega_spec, f1, f2
    del f105, bessel_b

    return sn_mag_f106


if __name__ == '__main__':

    # Run a test on the above function to see if
    # the K-corrections are working correctly
    phase = 0
    z_arr = np.arange(0.01, 3.01, 0.2)

    all_snmag = []

    for redshift in tqdm(z_arr):
        snmag = get_sn_mag_F106(phase, redshift)
        all_snmag.append(snmag)
        # print('{:.3f}'.format(redshift), '{:.3f}'.format(snmag))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(z_arr, all_snmag, color='k', s=2)
    plt.show()
    sys.exit(0)
