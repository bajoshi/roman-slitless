import numpy as np
from astropy.io import fits

import os
import sys

home = os.getenv('HOME')
stacking_utils = home + '/Documents/GitHub/stacking-analysis-pears/util_codes/'

modeldir = "/Volumes/Joshi_external_HDD/Roman/bc03_output_dir/m62/"

sys.path.append(stacking_utils)
from bc03_utils import gen_bc03_spectrum

# This class came from stackoverflow
# SEE: https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-python
class bcolors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def consolidate_fits2npy():

    """
    h = fits.open(modeldir + "bc2003_hr_m22_csp_tau11p000_chab.fits")
    np.save(modeldir + 'bc03_models_wavelengths.npy', h[1].data)
    np.save(modeldir + 'bc03_models_ages.npy', h[2].data)
    h.close()
    sys.exit(0)
    """

    # Set up arrays
    metals = 0.0001

    # First get hte metallicity string
    # Get the isedfile and the metallicity in the format that BC03 needs
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

    for tau_start in range(14, 20, 1):
        
        print("Working on tau starting from:", tau_start)
        tau_arr = np.arange(tau_start, tau_start + 1.000, 0.001)

        # Set up array and loop
        totalwavelengths = 13216
        totalages = 221
        totalspectra = totalages * len(tau_arr)
        npy_savearr = np.zeros((totalspectra, totalwavelengths), dtype=np.float32)

        count = 0

        for tau in tau_arr:

            print("Model number:", count+1, end='\r')

            # Now construct the output ised path
            tau_str = "{:.3f}".format(tau).replace('.', 'p')
            output = modeldir + "bc2003_hr_" + metallicity + "_csp_tau" + tau_str + "_chab.fits"

            # Now read the fits file and add spectra to large numpy array
            if not os.path.isfile(output):
                print("Missing fits file. Generating spec with tau:", tau)
                #if os.path.isfile(output.replace('.fits','.ised')): 
                #    os.remove(output.replace('.fits','.ised'))
                gen_bc03_spectrum(tau, metals, modeldir)

            h = fits.open(output)

            for j in range(totalages):

                spec = h[3+j].data

                # save to npy array
                npy_savearr[count] = spec

                count += 1

            h.close()

        # Now save
        savefile = modeldir + 'bc03_all_tau' + '{:.3f}'.format(tau_arr[0]).replace('.', 'p') + '_' + metallicity + '_chab.npy'
        np.save(savefile, npy_savearr)
        print("Saved:", savefile)

    return None

def main(tau):

    #consolidate_fits2npy()
    #sys.exit(0)

    # Set up arrays
    #tau_arr = np.arange(0.000, 9.001, 0.001)
    metals = 0.02

    # First get hte metallicity string
    # Get the isedfile and the metallicity in the format that BC03 needs
    #if metals == 0.0001:
    #    metallicity = 'm22'
    #elif metals == 0.0004:
    #    metallicity = 'm32'
    #elif metals == 0.004:
    #    metallicity = 'm42'
    #elif metals == 0.008:
    #    metallicity = 'm52'
    #elif metals == 0.02:
    #    metallicity = 'm62'
    #elif metals == 0.05:
    #    metallicity = 'm72'

    metallicity = 'm62'

    # If the ised file exists and has the correct size then don't rerun

    # Now construct the output ised path
    tau_str = "{:.3f}".format(tau).replace('.', 'p')
    output = modeldir + "bc2003_hr_" + metallicity + "_csp_tau" + tau_str + "_chab.ised"

    print("Working on:", output)
    gen_bc03_spectrum(tau, metals, modeldir)

    #if os.path.isfile(output):
    #    s = os.stat(output).st_size
    #    s = s/1e6
    #    # be careful if this is run on anything other than MacOS
    #    # MacOS uses base-10 file size, i.e., 1 MB = 1e6 bytes,
    #    # whereas other OSes will use base-2, i.e., 1 MB = 1024*1024 bytes
    #    if s >= 11.5:  # in MB
    #        print(f"{bcolors.GREEN}File exists and has correct size. Moving to the next model.{bcolors.ENDC}")
    #        return None
    #    else:
    #        os.remove(output)
    #        print(f"{bcolors.FAIL}Removed:", output, "due to file size of", s, "MB", f"{bcolors.ENDC}")
    #        gen_bc03_spectrum(tau, metals, modeldir)
    #else:
    #    gen_bc03_spectrum(tau, metals, modeldir)            

    return None

if __name__ == '__main__':

    tau = int(sys.argv[1])
    tau /= 1000

    main(tau)

    sys.exit(0)





