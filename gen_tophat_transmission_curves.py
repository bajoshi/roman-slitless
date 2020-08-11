import numpy as np

import os
import sys

home = os.getenv("HOME")
roman_slitless_dir = home + "/Documents/GitHub/roman-slitless/"

def main():

    # Generate grism and prism curves
    g_wav = np.arange(1.0e4, 1.93e4 + 1.0, 1.0)
    p_wav = np.arange(0.75e4, 1.8e4 + 1.0, 1.0)
    tophat_trans = 0.4  # blanket 40 percent transmission
    wav_long = np.arange(0.7e4, 2.0e4 + 1.0, 1.0)

    fh_g = open(roman_slitless_dir + 'grism_tophat_transmission.txt', 'w')
    fh_p = open(roman_slitless_dir + 'prism_tophat_transmission.txt', 'w')

    fh_g.write("#  wavelength_A  transmission_percent" + "\n")
    fh_p.write("#  wavelength_A  transmission_percent" + "\n")

    for i in range(len(wav_long)):

        if (wav_long[i] >= g_wav[0]) and (wav_long[i] <= g_wav[-1]): 
            fh_g.write(str(wav_long[i]) + "  " + str(tophat_trans) + "\n") 
        else: 
            fh_g.write(str(wav_long[i]) + "  " + "0.0" + "\n")

        if (wav_long[i] >= p_wav[0]) and (wav_long[i] <= p_wav[-1]):
            fh_p.write(str(wav_long[i]) + "  " + str(tophat_trans) + "\n") 
        else: 
            fh_p.write(str(wav_long[i]) + "  " + "0.0" + "\n")

    fh_g.close()
    fh_p.close()

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)