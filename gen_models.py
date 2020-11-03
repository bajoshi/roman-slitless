import numpy as np

import os
import sys

home = os.getenv('HOME')
stacking_utils = home + '/Documents/GitHub/stacking-analysis-pears/util_codes/'

modeldir = "/Volumes/Joshi_external_HDD/Roman/bc03_output_dir/"

sys.path.append(stacking_utils)
from bc03_utils import get_bc03_spectrum

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

def main():

    # Set up arrays
    tau_arr = np.arange(9.000, 15.001, 0.001)
    metals = 0.0001

    for tau in tau_arr:

        # If the ised file exists and has the correct size then don't rerun

        # First get hte metallicity string
        # Get the isedfile and the metallicity in the format that BC03 needs
        if metals == 0.0001:
            metallicity = 'm22'
            isedfile = home + "/Documents/GALAXEV2016/bc03/src/bc2003_hr_xmiless_m22_chab_ssp.ised"
        elif metals == 0.0004:
            metallicity = 'm32'
            isedfile = home + "/Documents/GALAXEV2016/bc03/src/bc2003_hr_xmiless_m32_chab_ssp.ised"
        elif metals == 0.004:
            metallicity = 'm42'
            isedfile = home + "/Documents/GALAXEV2016/bc03/src/bc2003_hr_xmiless_m42_chab_ssp.ised"
        elif metals == 0.008:
            metallicity = 'm52'
            isedfile = home + "/Documents/GALAXEV2016/bc03/src/bc2003_hr_xmiless_m52_chab_ssp.ised"
        elif metals == 0.02:
            metallicity = 'm62'
            isedfile = home + "/Documents/GALAXEV2016/bc03/src/bc2003_hr_xmiless_m62_chab_ssp.ised"
        elif metals == 0.05:
            metallicity = 'm72'
            isedfile = home + "/Documents/GALAXEV2016/bc03/src/bc2003_hr_xmiless_m72_chab_ssp.ised"

        # Now construct the output ised path
        tau_str = "{:.3f}".format(tau).replace('.', 'p')
        output = modeldir + "bc2003_hr_" + metallicity + "_csp_tau" + tau_str + "_chab.ised"

        print("\nWorking on:", output)

        if os.path.isfile(output):
            s = os.stat(output).st_size
            s = s/1e6
            # be careful if this is run on anything other than MacOS
            # MacOS uses base-10 file size, i.e., 1 MB = 1e6 bytes,
            # whereas other OSes will use base-2, i.e., 1 MB = 1024*1024 bytes
            if s >= 11.5:  # in MB
                print("File exists and has correct size. Moving to the next model.")
                continue
            else:
                os.remove(output)
                print(f"{bcolors.FAIL}Removed:", output, "due to file size of", s, "MB", f"{bcolors.ENDC}")
                model_lam, model_llam = get_bc03_spectrum(1.0, tau, metals, modeldir)
        else:            
            model_lam, model_llam = get_bc03_spectrum(1.0, tau, metals, modeldir)

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)

