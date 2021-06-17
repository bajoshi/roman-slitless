import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import griddata

import time
import os
import sys
import socket

from functools import reduce
import numba
from numba import jit

print("Make sure that the model_galaxy function here is the same")
print("as that in the fitting pipeline.")
print("-----------------\n")

start = time.time()

# Assign directories and custom imports
cwd = os.path.dirname(os.path.abspath(__file__))
fitting_utils = cwd + '/utils/'

home = os.getenv('HOME')
pears_figs_dir = home + '/Documents/pears_figs_data/'

sys.path.append(fitting_utils)
import dust_utils as du

# Get the dirs correct
if 'plffsn2' in socket.gethostname():
    home = os.getenv('HOME')
    extdir = '/astro/ffsn/Joshi/'
    modeldir = extdir + 'bc03_output_dir/'
    roman_direct_dir = home + '/Documents/roman_direct_sims/sims2021/'
else:
    extdir = '/Volumes/Joshi_external_HDD/Roman/'
    modeldir = extdir + 'bc03_output_dir/m62/'
    roman_direct_dir = extdir + 'roman_direct_sims/sims2021/'

assert os.path.isdir(modeldir)
assert os.path.isdir(roman_direct_dir)

dir_img_part = 'part1'
img_sim_dir = roman_direct_dir + 'K_5degimages_' + dir_img_part + '/'

# Now do the actual reading in
model_lam = np.load(extdir + "bc03_output_dir/bc03_models_wavelengths.npy")
model_ages = np.load(extdir + "bc03_output_dir/bc03_models_ages.npy")

all_m62_models = []
tau_low = 0
tau_high = 20
for t in range(tau_low, tau_high, 1):
    tau_str = "{:.3f}".format(t).replace('.', 'p')
    a = np.load(modeldir + 'bc03_all_tau' + tau_str + '_m62_chab.npy', mmap_mode='r')
    all_m62_models.append(a)
    del a

# load models with large tau separately
all_m62_models.append(np.load(modeldir + 'bc03_all_tau20p000_m62_chab.npy', mmap_mode='r'))

# Read in all models and parameters
model_lam_grid = np.load(pears_figs_dir + 'model_lam_grid_withlines_chabrier.npy')
model_grid = np.load(pears_figs_dir + 'model_comp_spec_llam_withlines_chabrier.npy',
    mmap_mode='r')

mg = np.asarray(model_grid, dtype=np.float64)
ml = np.asarray(model_lam_grid, dtype=np.float64)

log_age_arr = np.load(pears_figs_dir + 'log_age_arr_chab.npy')
metal_arr = np.load(pears_figs_dir + 'metal_arr_chab.npy')
tau_gyr_arr = np.load(pears_figs_dir + 'tau_gyr_arr_chab.npy')
tauv_arr = np.load(pears_figs_dir + 'tauv_arr_chab.npy')

#print('log age range:', np.min(log_age_arr), np.max(log_age_arr))
#print('metals range:', np.min(metal_arr), np.max(metal_arr))
#print('tau gyr range:', np.min(tau_gyr_arr[tau_gyr_arr>0.0]), np.max(tau_gyr_arr))
#print('tau-v range:', np.min(tauv_arr[tauv_arr>0.0]), np.max(tauv_arr))

# Also load in lookup table for luminosity distance
dl_cat = np.genfromtxt(fitting_utils + 'dl_lookup_table.txt', dtype=None, names=True)
# Get arrays 
dl_z_arr = np.asarray(dl_cat['z'], dtype=np.float64)
dl_cm_arr = np.asarray(dl_cat['dl_cm'], dtype=np.float64)
age_gyr_arr = np.asarray(dl_cat['age_gyr'], dtype=np.float64)

del dl_cat

t1 = []
t2 = []
t3 = []
t4 = []
t5 = []
t6 = []

get_dl_at_z_locals_dict = {
    'z': numba.float64,
    'z_idx': numba.int64,
    'dl': numba.float64
}

@jit(nopython=True, parallel=True)#, locals=get_dl_at_z_locals_dict)
def get_dl_at_z(z):

    adiff = np.abs(dl_z_arr - z)
    z_idx = np.argmin(adiff)
    dl = dl_cm_arr[z_idx]

    return dl

#get_dl_at_z(1.0)
#print(get_dl_at_z.inspect_types(pretty=True))
#sys.exit(0)

pi = 3.14159265359

@jit(nopython=True, parallel=True)
def apply_redshift(restframe_wav, restframe_lum, redshift):

    dl = get_dl_at_z(redshift)

    oneplusz = 1 + redshift

    redshifted_wav = restframe_wav * oneplusz
    redshifted_flux = restframe_lum / (4 * pi * dl * dl * oneplusz)

    return redshifted_wav, redshifted_flux

#apply_redshift(np.arange(1000.0, 2000.0), np.linspace(1e-14, 5e-14, 1000), 1.0)
#print(apply_redshift.inspect_types(pretty=True))
#sys.exit(0)

@jit(nopython=True)
def manual_intersect1d(a1, a2, a3, a4):

    # typically age idx is the shortest so 
    # match with that
    for a in a1:
        if (a in a2) and (a in a3) and (a in a4):
            unique_common_idx = a
            break

    return unique_common_idx

@jit(nopython=True, parallel=True)
def get_template(age, tau, tauv, metallicity):

    # First find closest values and then indices corresponding to them
    # It has to be done this way because you typically wont find an exact match
    closest_age_idx = np.argmin(np.abs(log_age_arr - age))
    closest_tau_idx = np.argmin(np.abs(tau_gyr_arr - tau))
    closest_tauv_idx = np.argmin(np.abs(tauv_arr - tauv))

    # Now get indices
    age_idx = np.flatnonzero(log_age_arr == log_age_arr[closest_age_idx])
    tau_idx = np.flatnonzero(tau_gyr_arr == tau_gyr_arr[closest_tau_idx])
    tauv_idx = np.flatnonzero(tauv_arr   ==    tauv_arr[closest_tauv_idx])
    metal_idx = np.flatnonzero(metal_arr == metallicity)

    model_idx = manual_intersect1d(age_idx, tau_idx, tauv_idx, metal_idx)

    model_llam = mg[model_idx]

    return model_llam

#l = get_template(np.log10(6.0 * 1e9), 10.0, 0.5, 0.02)
#print(get_template.inspect_types(pretty=True))
#sys.exit(0)

def model_galaxy(x, z, ms, age, logtau, av):
    """
    Expects to get the following arguments
    x: observed wavelength grid
    z: redshift to apply to template
    ms: log of the stellar mass
    age: age of SED in Gyr
    tau: exponential SFH timescale in Gyr
    metallicity: absolute fraction of metals
    av: visual dust extinction
    """

    # If using hte larger model set with no emission lines
    """
    tau = 10**logtau  # logtau is log of tau in gyr

    #print("log(tau [Gyr]):", logtau)
    #print("Tau [Gyr]:", tau)
    #print("Age [Gyr]:", age)

    if tau < 20.0:

        tau_int_idx = int((tau - int(np.floor(tau))) * 1e3)
        age_idx = np.argmin(abs(model_ages - age*1e9))
        model_idx = tau_int_idx * len(model_ages)  +  age_idx

        models_taurange_idx = np.argmin(abs(np.arange(tau_low, tau_high, 1) - int(np.floor(tau))))
        models_arr = all_m62_models[models_taurange_idx]

    elif tau >= 20.0:

        logtau_arr = np.arange(1.30, 2.01, 0.01)
        logtau_idx = np.argmin(abs(logtau_arr - logtau))

        age_idx = np.argmin(abs(model_ages - age*1e9))
        model_idx = logtau_idx * len(model_ages) + age_idx

        models_arr = all_m62_models[-1]

    # Force to numpy array for numba
    model_llam = np.asarray(models_arr[model_idx], dtype=np.float64)
    """

    t1.append(time.time())

    # Smaller model set with emission lines
    tau = 10**logtau  # logtau is log of tau in gyr
    tauv = av / 1.086
    model_llam = get_template(np.log10(age * 1e9), tau, tauv, 0.02)

    model_llam = np.asarray(model_llam, dtype=np.float64)

    t2.append(time.time())

    # ------ Apply stellar velocity dispersion
    # ------ and dust attenuation
    # assumed for now as a constant 220 km/s
    # TODO: optimize
    # -- This does not have to be done each time the model function
    #    is called because we're assuming a constant vel disp
    #if stellar_vdisp:
    #    sigmav = 220
    #    model_vdisp = add_stellar_vdisp(ml, model_llam, sigmav)

    #    # ------ Apply dust extinction
    #    model_dusty_llam = du.get_dust_atten_model(ml, model_vdisp, av)

    #else:
    # ------ Apply dust extinction
    model_dusty_llam = du.get_dust_atten_model(ml, model_llam, av)

    model_dusty_llam = np.asarray(model_dusty_llam, dtype=np.float64)

    t3.append(time.time())

    # ------ Multiply luminosity by stellar mass
    model_dusty_llam = model_dusty_llam * 10**ms

    # ------ Apply redshift
    model_lam_z, model_flam_z = apply_redshift(ml, model_dusty_llam, z)
    #model_flam_z = Lsol * model_flam_z

    t4.append(time.time())

    # ------ Apply LSF
    model_lsfconv = gaussian_filter1d(input=model_flam_z, sigma=10.0)

    t5.append(time.time())
    
    # ------ Downgrade and regrid to grism resolution
    model_mod = griddata(points=model_lam_z, values=model_lsfconv, xi=x)

    t6.append(time.time())

    return model_mod

# ----- Testing preliminaries
z = np.arange(0.0, 3.0, 0.001)
ms = np.arange(8.0, 11.0, 0.01)
age = np.logspace(-1, 1, 1000)
logtau = np.linspace(-2, 2.0, 1000)
av = np.arange(0.0, 5.0, 0.01)

testiter = 100000
x = np.arange(7500, 18000, 30.0)

t1 = []
t2 = []
t3 = []
t4 = []
t5 = []
t6 = []

# Now call in loop
for i in range(testiter):

    print('Working on iteration:', i, end='\r')

    i_z = np.random.choice(z)
    i_ms = np.random.choice(ms)
    i_age = np.random.choice(age)
    i_logtau = np.random.choice(logtau)
    i_av = np.random.choice(av)

    model_galaxy(x, i_z, i_ms, i_age, i_logtau, i_av)

# --- Timing stuff
# This must be called before plotting because the 
# model func will be called again by the plotting routine.
#print(len(t1), len(t2), len(t3), len(t4), len(t5), len(t6))
t1 = np.asarray(t1)
t2 = np.asarray(t2)
t3 = np.asarray(t3)
t4 = np.asarray(t4)
t5 = np.asarray(t5)
t6 = np.asarray(t6)

print('\n')
print('Timing stats for each iteration of the model_galaxy function.')
print('------------------------')
print('Mean (t2-t1), i.e., time to get template   [seconds]:', "{:.5f}".format(np.mean(t2 - t1)))
print('Mean (t3-t2), i.e., time to apply dust     [seconds]:', "{:.5f}".format(np.mean(t3 - t2)))
print('Mean (t4-t3), i.e., time to apply Ms and z [seconds]:', "{:.5f}".format(np.mean(t4 - t3)))
print('Mean (t5-t4), i.e., time to apply LSF      [seconds]:', "{:.5f}".format(np.mean(t5 - t4)))
print('Mean (t6-t5), i.e., time to grid data      [seconds]:', "{:.5f}".format(np.mean(t6 - t5)))
print('------------------------')
print('Mean total time for model galaxy           [seconds]:', "{:.5f}".format(np.mean(t6 - t1)))









