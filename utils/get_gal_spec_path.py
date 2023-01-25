import numpy as np

import os
import socket

import dust_utils as du
from apply_redshift import apply_redshift

if 'plffsn2' in socket.gethostname():
    extdir = '/astro/ffsn/Joshi/'
    modeldir = extdir + 'bc03_output_dir/'
else:
    extdir = '/Volumes/Joshi_external_HDD/Roman/'
    modeldir = extdir + 'bc03_output_dir/m62/'

roman_sims_seds = extdir + 'roman_slitless_sims_seds/'

assert os.path.isdir(roman_sims_seds)
assert os.path.isdir(modeldir)

# ------------------------
model_lam = np.load(extdir + "bc03_output_dir/bc03_models_wavelengths.npy", mmap_mode='r')
model_ages = np.load(extdir + "bc03_output_dir/bc03_models_ages.npy", mmap_mode='r')

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
# ------------------------

def get_bc03_spec(age, logtau):

    tau = 10**logtau  # logtau is log of tau in gyr

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

    model_llam = models_arr[model_idx]

    return np.array(model_llam, dtype=np.float64)

def get_gal_spec_path(cosmo, redshift, Lsol=3.826e33):
    """
    This function will generate a template SED assuming
    a composite stellar population using BC03. 
    -- SFH is assumed to be exponential.
       -- where tau is in between 0.01 to 15.0 (in Gyr)
    -- Age is dependent on z and only allows for models that are 
       at least 100 Myr younger than the Universe.
    -- Dust is applied assuming a Calzetti form for the dust extinction law.
    -- Metallicity is one of the six options in BC03.
    -- Log of stellar mass/M_sol is between 10.0 to 11.5
    """

    # ---------- Choosing stellar population parameters ----------- #
    # Choose stellar pop parameters at random
    # --------- Stellar mass
    log_stellar_mass_arr = np.linspace(10.0, 11.5, 100)
    log_stellar_mass_chosen = np.random.choice(log_stellar_mass_arr)

    log_stellar_mass_str = "{:.2f}".format(log_stellar_mass_chosen).replace('.', 'p')

    # --------- Age
    age_arr = np.arange(0.1, 13.0, 0.005)  # in Gyr

    # Now choose age consistent with given redshift
    # i.e., make sure model is not older than the Universe
    # Allowing at least 100 Myr for the first galaxies to form after Big Bang
    age_at_z = cosmo.age(redshift).value  # in Gyr
    age_lim = age_at_z - 0.1  # in Gyr

    chosen_age = np.random.choice(age_arr)
    while chosen_age > age_lim:
        chosen_age = np.random.choice(age_arr)

    # --------- SFH
    # Choose SFH form from a few different models
    # and then choose params for the chosen SFH form
    sfh_forms = ['instantaneous', 'constant', 'exponential', 'linearly_declining']

    # choose_sfh(sfh_forms[2])

    tau_arr = np.arange(0.01, 15.0, 0.005)  # in Gyr
    chosen_tau = np.random.choice(tau_arr)

    # --------- Metallicity
    metals_arr = np.array([0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05])
    # While the newer 2016 version has an additional metallicity
    # referred to as "m82", the documentation never specifies the 
    # actual metallicity associated with it. So I'm ignoring that one.
    metals = 0.02 # np.random.choice(metals_arr)

    # Get hte metallicity string
    if metals == 0.0001:
        metallicity = 'm22'
    elif metals == 0.0004:
        metallicity = 'm32'
    elif metals == 0.004:
        metallicity = 'm42'
    elif metals == 0.008:
        metallicity = 'm52'
    elif metals == 0.02:
        metallicity = 'm62'
    elif metals == 0.05:
        metallicity = 'm72'

    # ----------- CALL BC03 -----------
    # The BC03 generated spectra will always be at redshift=0 and dust free.
    # This code will apply dust extinction and redshift effects manually
    #outdir = home + '/Documents/bc03_output_dir/'
    #gen_bc03_spectrum(chosen_tau, metals, outdir)
    logtau = np.log10(chosen_tau)
    bc03_spec_wav = np.array(model_lam, dtype=np.float64)
    bc03_spec_llam = get_bc03_spec(chosen_age, logtau)

    # Apply Calzetti dust extinction depending on av value chosen
    av_arr = np.arange(0.0, 5.0, 0.001)  # in mags
    chosen_av = np.random.choice(av_arr)

    bc03_dusty_llam = du.get_dust_atten_model(bc03_spec_wav, bc03_spec_llam, chosen_av)

    # Multiply flux by stellar mass
    bc03_dusty_llam = bc03_dusty_llam * 10**log_stellar_mass_chosen

    # Convert to physical units
    bc03_dusty_llam *= Lsol

    # Apply redshift
    pc2cm = 3.086e18  # 1 parsec to centimeter
    dl = cosmo.luminosity_distance(redshift).value * 1e6 * pc2cm  # in cm
    bc03_wav_z, bc03_flux = apply_redshift(bc03_spec_wav, bc03_dusty_llam, redshift, dl)

    # Save file
    gal_spec_path = roman_sims_seds + 'bc03_template' + \
    "_z" + "{:.3f}".format(redshift).replace('.', 'p') + \
    "_ms" + log_stellar_mass_str + \
    "_age" + "{:.3f}".format(chosen_age).replace('.', 'p') + \
    "_tau" + "{:.3f}".format(chosen_tau).replace('.', 'p') + \
    "_met" + "{:.4f}".format(metals).replace('.', 'p') + \
    "_av" + "{:.3f}".format(chosen_av).replace('.', 'p') + \
    ".txt"

    # Save individual spectrum file if it doesn't already exist
    if not os.path.isfile(gal_spec_path):

        fh_gal = open(gal_spec_path, 'w')
        fh_gal.write("#  lam  flux")
        fh_gal.write("\n")

        for j in range(len(bc03_flux)):

            fh_gal.write("{:.2f}".format(bc03_wav_z[j]) + " " + str(bc03_flux[j]))
            fh_gal.write("\n")

        fh_gal.close()

    return gal_spec_path

