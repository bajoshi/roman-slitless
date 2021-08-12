import numpy as np

import os
import sys
import tarfile
import glob

import matplotlib.pyplot as plt

home = os.getenv("HOME")
roman_slitless_dir = home + "/Documents/GitHub/roman-slitless/"

def main():
    # Metadata for final saving
    meta = ' SNANA_VERSION: v11_03b' + '\n' + \
           ' Catenated data files: ' + '\n' + \
           '   + /scratch/midway2/rkessler/PIPPIN_OUTPUT/WFIRST_STARTERKIT+RUBIN/' + \
           '5_MERGE/MERGE_WFIRSTfit_SIM_WFIRST_SIMDATA_G10/output/PIP_WFIRST_STARTERKIT+RUBIN_WFIRST_SIMDATA_G10/FITOPT000.FITRES' + '\n' + \
           '   + /scratch/midway2/rkessler/PIPPIN_OUTPUT/WFIRST_STARTERKIT+RUBIN/' + \
           '5_MERGE/MERGE_FOUNDfit_SIM_FOUND_SIMDATA_G10/output/PIP_WFIRST_STARTERKIT+RUBIN_FOUND_SIMDATA_G10/FITOPT000.FITRES' + '\n' + \
           ' Appended columns: PROB* ' + '\n' + \
           ' Dropped columns: TrestMIN TrestMAX OPT_PHOTOZ NFILT_USEFIT SNRMAX_g SNRMAX_r SNRMAX_i  ' + '\n'
    #'#' + '\n'

    # Read in original fitres file
    header = ['VARNAMES:', 'CID', 'CIDint', 'IDSURVEY', 'TYPE', 'FIELD', 'CUTFLAG_SNANA', 
              'ERRFLAG_FIT', 'zHEL', 'zHELERR', 'zCMB', 'zCMBERR', 'zHD', 'zHDERR', 
              'VPEC', 'VPECERR', 'MWEBV', 'HOST_LOGMASS', 'HOST_LOGMASS_ERR', 
              'HOST_sSFR', 'HOST_sSFR_ERR', 'PKMJDINI', 'SNRMAX1', 'SNRMAX2', 
              'SNRMAX3', 'PKMJD', 'PKMJDERR', 'x1', 'x1ERR', 'c', 'cERR', 'mB', 
              'mBERR', 'x0', 'x0ERR', 'COV_x1_c', 'COV_x1_x0', 'COV_c_x0', 'NDOF', 
              'FITCHI2', 'FITPROB', 'SIM_TYPE_INDEX', 'SIM_TEMPLATE_INDEX', 
              'SIM_LIBID', 'SIM_NGEN_LIBID', 'SIM_ZCMB', 'SIM_ZFLAG', 'SIM_VPEC', 
              'SIM_DLMAG', 'SIM_PKMJD', 'SIM_x1', 'SIM_c', 'SIM_alpha', 'SIM_beta', 
              'SIM_x0', 'SIM_mB', 'SIM_AV', 'SIM_RV', 'RA', 'DEC', 'TGAPMAX', 'PROB_FITPROBTEST']
    fitres_path = roman_slitless_dir + 'INPUT_FITOPT000.FITRES' 

    # Construct header to use with numpy savetxt
    full_header = meta
    for h in range(len(header)):
        full_header += header[h] + ' '
    full_header += '\n'

    # set how many modified fitres files to generate
    iterations = 10

    zdiff = []

    # Loop
    for it in range(iterations):

        # Must read the fitres file every time 
        # an iteration is done because it gets 
        # modified each time.
        fitres = np.genfromtxt(fitres_path, dtype=None, names=header, encoding='ascii', skip_header=9)

        new_fitres_name = fitres_path.replace('.FITRES', '_BAJmod_10ltSNRlt20_iter' + str(it) + '.FITRES')

        # Now loop over each line and modify if required
        for i in range(len(fitres)):
            snr = fitres['SNRMAX1'][i]
            if (snr >= 10.0) and (snr <= 20.0):
                z = fitres['zCMB'][i]
                zerr = 0.0023
                znew = np.random.normal(loc=z, scale=zerr, size=None)

                fitres['zHEL'][i] = znew
                fitres['zCMB'][i] = znew
                fitres['zHD'][i]  = znew

                fitres['zCMBERR'][i] = zerr
                fitres['zHELERR'][i] = zerr
                fitres['zHDERR'][i]  = zerr

                zdiff.append(znew - z)

        np.savetxt(new_fitres_name, fitres, delimiter=' ', fmt='%s', 
            header=full_header)

    print('FITRES files modified and saved.')

    # put all iterations in one tar.gz
    flist = glob.glob(roman_slitless_dir + '*iter*.FITRES')
    with tarfile.open(home + '/Documents/bajmod_fitres.tar.gz', 'w:gz') as tar:
        for fl in flist:
            tar.add(fl)

    # Now delete all individual modified FITRES files
    for fl in flist:
        os.remove(fl)

    # Plot differences in redshift
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('znew - z')
    ax.set_ylabel('Number')
    ax.hist(zdiff, 100, range=(-0.01, 0.01), histtype='step')
    fig.savefig('figures/zdiff_for_fitres.pdf', dpi=200, bbox_inches='tight')

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)

