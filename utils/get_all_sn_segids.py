import numpy as np

def get_all_sn_segids(sedlst_path):
    # Read in sed.lst
    sedlst_header = ['segid', 'sed_path']
    sedlst = np.genfromtxt(sedlst_path, dtype=None, names=sedlst_header, encoding='ascii')

    # loop and find all SN segids
    all_sn_segids = []
    for i in range(len(sedlst)):
        if 'salt' in sedlst['sed_path'][i]:
            all_sn_segids.append(sedlst['segid'][i])

    return all_sn_segids