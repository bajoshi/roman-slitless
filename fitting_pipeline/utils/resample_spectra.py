import numpy as np

from astropy.io import fits
from astropy import units as u
from specutils.manipulation import FluxConservingResampler
from specutils import Spectrum1D
from scipy.interpolate import griddata

import os
import sys
from tqdm import tqdm

import matplotlib.pyplot as plt

home = os.getenv('HOME')

extdir = "/Volumes/Joshi_external_HDD/Roman/"
ext_spectra_dir = extdir + "roman_slitless_sims_results/"
pylinear_lst_dir = extdir + 'pylinear_lst_files/'
roman_slitless_dir = home + 'Documents/GitHub/roman-slitless/'
fitting_utils = roman_slitless_dir + 'fitting_pipeline/utils/'

sys.path.append(fitting_utils)
from get_template_inputs import get_template_inputs # noqa

# Set pylinear f_lambda scaling factor
pylinear_flam_scale_fac = 1e-17


def resample_spec(wav, flux, ferr, new_wav_grid, plotfig=False):

    # Also resample error
    # This is likely the wrong thing to do
    # TODO: Figure out the correct way.

    # Using griddata because the flux conserving resampler
    # below does something that gives all NaNs here.
    new_err = griddata(points=wav, values=ferr, xi=new_wav_grid)

    # Now resample the flux
    # Get astropy formats
    wav = wav * u.AA
    flux = flux * u.Unit('erg cm-2 s-1 AA-1')
    new_wav_grid = new_wav_grid * u.AA

    input_spec = Spectrum1D(spectral_axis=wav, flux=flux)

    fluxcon = FluxConservingResampler()
    new_spec_fluxcon = fluxcon(input_spec, new_wav_grid)

    resampled_spec = new_spec_fluxcon.flux

    if plotfig:
        fig = plt.figure(figsize=(9, 5))
        ax = fig.add_subplot(111)
        ax.plot(wav, flux, color='royalblue', lw=3.0)
        ax.plot(new_wav_grid, resampled_spec, '--',
                color='crimson', lw=1.0)
        plt.show()
        fig.clear()
        plt.close(fig)

    return resampled_spec, new_err


if __name__ == '__main__':

    print('\nWARNING: check the effective error after resampling.')
    print('Currently resampling error like the flux array.\n')

    exptime1 = '_3600s'
    exptime2 = '_1200s'
    exptime3 = '_400s'

    all_exptimes = [exptime1, exptime2, exptime3]

    # Get new wav grid with the variable dispersion
    # I basically copy pasted the wav column from Jeff Kruk's
    # file. This was copied for the 19th AB mag source but they
    # are all identical.
    prism_wav_grid = [0.72750, 0.72929, 0.73110, 0.73292, 0.73475, 0.73659,
                      0.73845, 0.74032, 0.74220, 0.74410, 0.74601, 0.74794,
                      0.74988, 0.75183, 0.75380, 0.75579, 0.75779, 0.75980,
                      0.76183, 0.76388, 0.76594, 0.76801, 0.77011, 0.77222,
                      0.77434, 0.77648, 0.77864, 0.78082, 0.78301, 0.78523,
                      0.78746, 0.78971, 0.79197, 0.79426, 0.79656, 0.79888,
                      0.80123, 0.80359, 0.80597, 0.80838, 0.81080, 0.81324,
                      0.81571, 0.81820, 0.82070, 0.82323, 0.82579, 0.82836,
                      0.83096, 0.83358, 0.83623, 0.83890, 0.84159, 0.84431,
                      0.84705, 0.84982, 0.85262, 0.85544, 0.85828, 0.86116,
                      0.86406, 0.86699, 0.86994, 0.87293, 0.87594, 0.87899,
                      0.88206, 0.88517, 0.88830, 0.89147, 0.89466, 0.89789,
                      0.90116, 0.90445, 0.90778, 0.91114, 0.91454, 0.91797,
                      0.92144, 0.92494, 0.92848, 0.93206, 0.93568, 0.93933,
                      0.94302, 0.94675, 0.95053, 0.95434, 0.95819, 0.96209,
                      0.96603, 0.97001, 0.97403, 0.97810, 0.98222, 0.98637,
                      0.99058, 0.99483, 0.99913, 1.00348, 1.00788, 1.01232,
                      1.01682, 1.02137, 1.02596, 1.03061, 1.03532, 1.04007,
                      1.04488, 1.04975, 1.05467, 1.05965, 1.06468, 1.06977,
                      1.07492, 1.08012, 1.08539, 1.09071, 1.09610, 1.10154,
                      1.10705, 1.11262, 1.11825, 1.12395, 1.12970, 1.13552,
                      1.14141, 1.14736, 1.15337, 1.15945, 1.16560, 1.17181,
                      1.17809, 1.18443, 1.19084, 1.19732, 1.20386, 1.21047,
                      1.21714, 1.22389, 1.23070, 1.23757, 1.24451, 1.25152,
                      1.25859, 1.26573, 1.27294, 1.28020, 1.28754, 1.29493,
                      1.30239, 1.30991, 1.31749, 1.32513, 1.33283, 1.34059,
                      1.34841, 1.35629, 1.36422, 1.37221, 1.38025, 1.38835,
                      1.39649, 1.40469, 1.41294, 1.42123, 1.42957, 1.43796,
                      1.44639, 1.45487, 1.46338, 1.47194, 1.48053, 1.48916,
                      1.49783, 1.50653, 1.51527, 1.52403, 1.53283, 1.54165,
                      1.55050, 1.55938, 1.56828, 1.57720, 1.58615, 1.59511,
                      1.60409, 1.61309, 1.62211, 1.63114, 1.64019, 1.64924,
                      1.65831, 1.66738, 1.67647, 1.68556, 1.69465, 1.70375,
                      1.71286, 1.72197, 1.73107, 1.74018, 1.74929, 1.75840,
                      1.76750, 1.77660, 1.78570, 1.79479, 1.80387, 1.81295,
                      1.82202, 1.83108, 1.84014, 1.84918]

    # Our spectral range is 7500 to 18720 A
    # so we cut it off to that range
    # Also convert microns to Angstroms
    # first convert to numpy array
    prism_wav_grid = np.asarray(prism_wav_grid)
    prism_wav_grid *= 1e4

    low_wav_idx = np.argmin(np.abs(prism_wav_grid - 7500))
    prism_wav_grid = prism_wav_grid[low_wav_idx:]

    # ---------- Loop over all simulated and extracted SN spectra ---------- #
    # Arrays to loop over
    pointings = np.arange(0, 1)
    detectors = np.arange(1, 19, 1)

    for pt in pointings:
        for det in tqdm(detectors, desc='Detector'):

            img_suffix = 'Y106_' + str(pt) + '_' + str(det)

            # --------------- Read in sed.lst
            sedlst_header = ['segid', 'sed_path']
            sedlst_path = pylinear_lst_dir + 'sed_' + img_suffix + '.lst'
            sedlst = np.genfromtxt(sedlst_path, dtype=None,
                                   names=sedlst_header, encoding='ascii')

            # --------------- loop and find all SN segids
            all_sn_segids = []
            for i in range(len(sedlst)):
                if ('salt' in sedlst['sed_path'][i])\
                        or ('contam' in sedlst['sed_path'][i]):
                    all_sn_segids.append(sedlst['segid'][i])

            # --------------- Loop over all extracted files
            for e in tqdm(range(len(all_exptimes)), desc='Exptimes'):

                exptime = all_exptimes[e]

                # --------------- Read in the extracted spectra
                ext_spec_filename = (ext_spectra_dir + 'romansim_prism_'
                                     + img_suffix + exptime + '_x1d.fits')
                ext_hdu = fits.open(ext_spec_filename)

                # Create a new fits file to save all the spectra
                phdu = fits.PrimaryHDU(header=ext_hdu[0].header)
                hdul = fits.HDUList([phdu])
                resamp_filename = ext_spec_filename.replace('.fits',
                                                            '_resamp.fits')

                # Loop over each SN in x1d file
                for segid in tqdm(all_sn_segids, desc='SN on det ' + str(det)):

                    # ----- Get spectrum
                    wav = ext_hdu[('SOURCE', segid)].data['wavelength']
                    flam = ext_hdu[('SOURCE', segid)].data['flam'] * \
                        pylinear_flam_scale_fac

                    ferr_lo = ext_hdu[('SOURCE', segid)].data['flounc'] * \
                        pylinear_flam_scale_fac
                    ferr_hi = ext_hdu[('SOURCE', segid)].data['fhiunc'] * \
                        pylinear_flam_scale_fac

                    # ----- Get noise level
                    ferr = (ferr_lo + ferr_hi) / 2.0

                    # ----- Now resample
                    resampled_spec, resampled_err = \
                        resample_spec(wav, flam, ferr, prism_wav_grid)

                    # Create table and save
                    c1 = fits.Column(name='wav', array=prism_wav_grid,
                                     format='E')
                    c2 = fits.Column(name='flam', array=resampled_spec,
                                     format='E')
                    c3 = fits.Column(name='ferr', array=resampled_err,
                                     format='E')

                    tab = fits.BinTableHDU.from_columns([c1, c2, c3])

                    hdul.append(tab)

                hdul.writeto(resamp_filename, overwrite=True)

    sys.exit(0)
