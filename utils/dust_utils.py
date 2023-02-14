import numpy as np
# from numba import jit


# @jit(nopython=True)
def get_klambda(wav):
    # Calzetti law
    # wavlength has to be in microns

    rv = 4.05

    if wav > 0.63:
        klam = 2.659 * (-1.857 + 1.040/wav) + rv

    elif wav < 0.63:
        klam = 2.659 * (-2.156 + 1.509/wav - 0.198/wav**2 + 0.011/wav**3) + rv

    elif wav == 0.63:
        klam = 3.49
        # Since the curves dont exactly meet at 0.63 micron, I've taken the average of the two.
        # They're close though. One gives me klam=3.5 and the other klam=3.48

    return klam


# @jit(nopython=True)
def get_dust_atten_model(model_wav_arr, model_flux_arr, av):
    """
    This function will apply the Calzetti dust extinction law 
    to the given model using the supplied value of Av.

    It assumes that the model it is being given is dust-free.
    It assumes that the model wavelengths it is given are in Angstroms.

    It returns the dust-attenuated flux array
    at the same wavelengths as before.
    """
    # Now loop over the dust-free SED and generate a new dust-attenuated SED
    # print("Applying dust attenuation...")
    dust_atten_model_flux = np.empty(len(model_wav_arr), np.float64)

    current_wav = model_wav_arr / 1e4  # because this has to be in microns

    for i in range(len(model_wav_arr)):

        cw = current_wav[i]

        # The calzetti law is only valid up to 2.2 micron so beyond 
        # 2.2 micron this function just replaces the old values
        if cw <= 2.2:
            klam = get_klambda(cw)
            alam = klam * av / 4.05

            dust_atten_model_flux[i] = (model_flux_arr[i] *
                                        10**(-1 * 0.4 * alam))
        else:
            dust_atten_model_flux[i] = model_flux_arr[i]

        # print(i, cw, alam, 10**(-1 * 0.4 * alam), \
        #      "{:.2e}".format(model_flux_arr[i]),
        #      "{:.2e}".format(dust_atten_model_flux[i]))

    return dust_atten_model_flux
