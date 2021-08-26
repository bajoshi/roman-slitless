import numpy as np
from tqdm import tqdm
from scipy.interpolate import griddata

import os
import sys

# Assign directories and custom imports
cwd = os.path.dirname(os.path.abspath(__file__))
fitting_utils = cwd + '/utils/'

sys.path.append(fitting_utils)
import dust_utils as du

# Define any required constants/arrays
sn_scalefac = 1.734e40  # see sn_scaling.py 
sn_day_arr = np.arange(-19,51,1)

# Read in SALT2 SN IA file from Lou
salt2_spec = np.genfromtxt(fitting_utils + "salt2_template_0.txt", \
    dtype=None, names=['day', 'lam', 'flam'], encoding='ascii')

# Also load in lookup table for luminosity distance
dl_cat = np.genfromtxt(fitting_utils + 'dl_lookup_table.txt', dtype=None, names=True)
# Get arrays 
dl_z_arr = np.asarray(dl_cat['z'], dtype=np.float64)
dl_cm_arr = np.asarray(dl_cat['dl_cm'], dtype=np.float64)

del dl_cat

# --------------------------------
# --------------------------------
# REdshift array
redshift_arr = np.arange(0.01, 3.01, 0.01)

# Av array
av_arr = np.arange(0.5, 5.5, 0.5)

# -------- Now save all models to an npy array
total_models = len(sn_day_arr) * len(redshift_arr) * len(av_arr)  #  total models
print('Total models:', total_models)

# Get the wavelengths for one of the SN spectra
# They're all the same
sn_lam = salt2_spec['lam'][salt2_spec['day'] == 0]

# For clipping to prism wav grid
x = np.arange(7500.0, 18095.0, 65.0)

# Empty array to write to
allmods = []

for d in tqdm(range(len(sn_day_arr)), desc='SN Phase'):

    day = sn_day_arr[d]

    day_idx = np.where(salt2_spec['day'] == day)[0]
    spec = salt2_spec['flam'][day_idx] * sn_scalefac

    for a in range(len(av_arr)):

        sn_av = av_arr[a]

        sn_dusty_llam = du.get_dust_atten_model(sn_lam, spec, sn_av)

        for r in range(len(redshift_arr)):

            z = redshift_arr[r]

            adiff = np.abs(dl_z_arr - z)
            z_idx = np.argmin(adiff)
            dl = dl_cm_arr[z_idx]

            sn_lam_z = sn_lam * (1 + z)
            spec_redshifted = sn_dusty_llam / (4 * np.pi * dl * dl * (1 + z))
            
            # Clip model to observed wavelength range
            # This must be the same range as the clipped range for the extracted spectra
            # Also make sure the wav sampling is the same
            # Currently the pylinear x1d prism spectra have
            #  np.arange(7500.0, 18030.0, 30.0) # defined above as x
            sn_mod = griddata(points=sn_lam_z, values=spec_redshifted, xi=x)

            allmods.append(sn_mod)

allmods = np.array(allmods)
assert allmods.shape == (total_models, len(x))

savepath = '/Volumes/Joshi_external_HDD/Roman/' + 'allsnmodspec.npy'
np.save(savepath, allmods)

print('Saved all modified SN models to:', savepath)

sys.exit(0)


# ------------------
# Now check the output
# Redefined for plotting
sn_lam = np.arange(7500.0, 18030.0, 30.0)  # NOT THE SAME AS THE sn_lam ARRAY ABOVE

av_arr = np.arange(0.5, 5.5, 0.5)
redshift_arr = np.arange(0.01, 3.01, 0.01)
sn_day_arr = np.arange(-19,51,1)

def retrieve_sn_optpars(big_index):

    av_subidx, z_idx = np.divmod(big_index, len(redshift_arr))
    trash, av_idx    = np.divmod(av_subidx, len(av_arr))
    phase_idx, trash = np.divmod(big_index, len(av_arr)*len(redshift_arr))

    #print(z_idx, av_subidx, phase_idx, trash)

    z = redshift_arr[z_idx]
    av = av_arr[av_idx]
    phase = sn_day_arr[phase_idx]

    del trash

    return z, phase, av

def check_output():

    import matplotlib.pyplot as plt
    import matplotlib as mpl

    import gc

    mpl.rcParams['text.usetex'] = False  # much faster without tex

    a = np.load('/Volumes/Joshi_external_HDD/Roman/allsnmodspec.npy')
    print(a.shape)

    for i in range(a.shape[0]):

        z, phase, av = retrieve_sn_optpars(i)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ax.plot(sn_lam, a[i], color='k')
        
        ax.text(x=0.75, y=0.2,  s='z = ' + '{:.3f}'.format(z), 
            verticalalignment='top', horizontalalignment='left', 
            transform=ax.transAxes, color='royalblue', size=12)
        ax.text(x=0.75, y=0.15, s='Phase = ' + '{:d}'.format(phase),
            verticalalignment='top', horizontalalignment='left', 
            transform=ax.transAxes, color='royalblue', size=12)
        ax.text(x=0.75, y=0.1,  s='Av = ' + '{:.3f}'.format(av), 
            verticalalignment='top', horizontalalignment='left', 
            transform=ax.transAxes, color='royalblue', size=12)

        plt.pause(0.01)

        plt.cla()
        plt.clf()
        #fig.clear()
        plt.close('all')
        plt.close(fig)

        #del fig, ax
        gc.collect()

        # None of the above works for now to release memory
        # python will just keep taking up memory until the
        # process is killed. Can only test in small batches.

    return None

check_output()



