import numpy as np
from astroquery.mast import Observations
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u

from tqdm import tqdm
import os
import sys

import matplotlib.pyplot as plt

home = os.getenv('HOME')

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

    # Read in the Pantheon+ catalog from Ben
    adap_dir = home + '/Documents/adap2021/'
    pantheon_datadir = adap_dir + 'pantheon_data/'
    cat = np.genfromtxt(adap_dir + 'pantheon_plus.csv', dtype=None, names=True, delimiter=',', encoding='ascii')

    print("Read in Pantheon+ catalog with the following header names:")
    print(cat.dtype.names)

    num_orig_cols = len(cat.dtype.names)

    # Open a new file to write an updated catalog
    # Adds the following columns 
    # HST data
    # GALEX data
    # if yes to any of the above observatories then give 
    # Inst/Camera field, and filters.
    # if no then leave these cols blank.
    fh = open(adap_dir + 'pantheon_plus_data.csv', 'w')

    # Write header
    fh.write("Serial_num,CID,CIDint,IDSURVEY,zHEL,zHELERR,zCMB,zCMBERR,zHD,zHDERR," +\
        "HOST_LOGMASS,HOST_LOGMASS_ERR,RA,DEC,HOST_RA,HOST_DEC," +\
        "HST_data,Inst/Cam,Filters" + "\n")

    # Loop over all objects in the catalog
    # and search for HST data at the SN and Host location
    for i in range(27, len(cat)): #tqdm(range(len(cat)), desc="Processing SN"):

        # Get coords
        sn_ra = cat['RA'][i]
        sn_dec = cat['DEC'][i]
        host_ra = cat['HOST_RA'][i]
        host_dec = cat['HOST_DEC'][i]

        # Print info
        #print(f"{bcolors.CYAN}")
        #print("SN identifier:", cat['CID'][i], " at:", sn_ra, sn_dec)
        #print("Host galaxy coords:", host_ra, host_dec)
        #print(f"{bcolors.ENDC}")

        # Set up query
        sn_coords = SkyCoord(ra=sn_ra*u.degree, dec=sn_dec*u.degree, frame='icrs')

        print("SN coordinates:", sn_coords)

        obs_table = Observations.query_criteria(coordinates=sn_coords, radius="0.5 arcsec", \
            intentType='science', obs_collection=['HST'])

        #print(obs_table)
        print(obs_table.columns)
        print("\nRows in obs table:", len(obs_table))
        print("HST filters available for this SN:")
        all_instr = np.unique(obs_table['instrument_name'])
        print(all_instr)
        print("--------------------------------------\n")

        sys.exit(0)

        # Download any existing wfc3 data
        for r in range(len(obs_table)):

            instr = obs_table['instrument_name'][r]

            if 'WFC3' in instr:

                data_products = Observations.get_product_list(obs_table[r])
                Observations.download_products(data_products, download_dir=pantheon_datadir, 
                    productType="SCIENCE", mrp_only=True)

        sys.exit(0)

        # Now loop over all the observations
        ra_one = sn_ra
        dec_one = sn_dec

        dist = []
        inst_cam = []
        exptimes = []
        filt = []

        for o in range(len(obs_table)):

            ra_two = obs_table['s_ra'][o]
            dec_two = obs_table['s_dec'][o]

            dist_to_sn = np.arccos(np.cos(dec_one*np.pi/180) * \
                np.cos(dec_two*np.pi/180) * np.cos(ra_one*np.pi/180 - ra_two*np.pi/180) + \
                np.sin(dec_one*np.pi/180) * np.sin(dec_two*np.pi/180))

            # print("{:.7}".format(ra_two), "{:.7}".format(dec_two))
            #print("Distance from SN [arcsec]:", "{:.5}".format(dist_to_sn * (180/np.pi) * 3600), \
            #    "Inst/Cam:", obs_table['instrument_name'][o], "Filter(s):", obs_table['filters'][o], \
            #    "ExpTime:", obs_table['t_exptime'][o])

            dist.append(dist_to_sn * (180/np.pi) * 3600)  # The dist returned by the line above is in radians
            inst_cam.append(obs_table['instrument_name'][o])
            filt.append(obs_table['filters'][o])
            exptimes.append(obs_table['t_exptime'][o])

        # Add to original catalog
        # Need to loop over the original row to do this
        for j in range(num_orig_cols):
            fh.write(str(cat[i][j]) + ',')

        # Now add the new cols
        # First convert to numpy arrays
        inst_cam = np.unique(np.asarray(inst_cam))
        filt = np.unique(np.asarray(filt))

        if len(obs_table) > 0:
            hst_data = True
        else:
            hst_data = False

        fh.write(str(hst_data) + ",")

        if len(inst_cam) > 1:
            for w in range(len(inst_cam)):
                fh.write(str(inst_cam[w]) + ";")
            fh.write(",")
        else:
            fh.write(str(inst_cam) + ",")

        if len(filt) > 1:
            for v in range(len(filt)):
                fh.write(str(filt[v]) + ";")
        else:
            fh.write(str(filt))
        fh.write("\n")

        # Check that the distances are within FoV of the instrument specified
        # dist = np.asarray(dist) * (180/np.pi) * 3600  # radians to degrees to arcseconds

    fh.close()

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)