import numpy as np
from astropy.io import fits

#import pylinear

import os
import sys

from inspect_obj_x1d import paperfigure

extdir = '/Volumes/Joshi_external_HDD/Roman/'
results_dir = extdir + 'roman_slitless_sims_results/'
img_sim_dir = extdir + 'roman_direct_sims/sims2021/K_5degimages_part1/'
pylinear_lst_dir = extdir + 'pylinear_lst_files/'
tablespath = results_dir + 'tables/'


def save_segid_2Dspec_region(detector, segid, exptime, SN=False):

    # ---------------------------
    # Open the 2D FLT files for all roll angles
    # for the given detector
    roll1 = 'romansim_prism1_Y106_0_' + str(detector) +\
        '_' + exptime + '_flt.fits'
    roll2 = 'romansim_prism2_Y106_0_' + str(detector) +\
        '_' + exptime + '_flt.fits'
    roll3 = 'romansim_prism3_Y106_0_' + str(detector) +\
        '_' + exptime + '_flt.fits'

    img1 = fits.open(results_dir + roll1)
    img2 = fits.open(results_dir + roll2)
    img3 = fits.open(results_dir + roll3)

    # ---------------------------
    # LST and segmap files needed
    segfile = img_sim_dir + '5deg_Y106_0_' + str(detector) +\
        '_SNadded_segmap.fits'
    obslst = pylinear_lst_dir + 'obs_Y106_0_' + str(detector) + \
        '.lst'
    fltlst = pylinear_lst_dir + 'flt_Y106_0_' + str(detector) + \
        '.lst'

    assert os.path.isfile(segfile)
    assert os.path.isfile(obslst)
    assert os.path.isfile(fltlst)

    print('\nNOTE: You have to edit the OBS and FLT LST paths by hand.')
    print('Copy paste the PLFFSN2 file and edit.')
    print('Note that the exptime info has been removed because the spectrum')
    print('will fall on the same location regardless of the exptime.\n')

    # ---------------------------
    # X1D spectra file
    # x1d_name = results_dir +\
    #     'romansim_prism_Y106_0_' + str(detector) + '_x1d.fits'
    # x1d = fits.open(x1d_name)

    # ---------------------------
    # If given a SN segid then you need
    # to save the combined SN + host galaxy
    # 2D spec region.

    # ---------------------------
    # Save the region
    # ---------------------------
    # Load in sources
    sources = pylinear.source.SourceCollection(segfile, obslst, 
                                               detindex=0,
                                               maglim=30)
    
    # Load in grisms for the sim to test
    grisms = pylinear.grism.GrismCollection(fltlst, observed=True)

    # ---------------------------
    # Loop over grisms and gen regions for requested segid    
    for grism in grisms:
            
        print("Working on:", grism.dataset)
        reg_filename =\
            img_sim_dir + grism.dataset + '_' + str(segid) + '_2dext.reg'

        with open(reg_filename, 'w') as fh:
            
            with pylinear.h5table.H5Table(grism.dataset, path=tablespath,
                                          mode='r') as h5:
                
                device = grism['SCA09']
                
                h5.open_table('SCA09', '+1', 'pdt')
                
                odt = h5.load_from_file(sources[segid], '+1', 'odt')
                ddt = odt.decimate(device.naxis1, device.naxis2)
                
                region_text = ddt.region()
                region_text = region_text.replace('helvetica 12 bold',
                                                  'helvetica 10 bold')

                # Make it a color that shows better with viridis
                # Comment out one or the other along with the segid
                # arg on the command line
                # red for SNe
                # region_text = \
                #     region_text.replace('color=#1f77b4', 'color=#e41a1c')
                # light purple for hosts
                region_text = \
                    region_text.replace('color=#1f77b4', 'color=#fccde5')

                # Remove text string
                txt_str = 'text={' + str(segid) + '}'
                region_text = region_text.replace(txt_str, '')
            
                fh.write(region_text + '\n')

    # ---------------------------
    # CLose all open HDUs
    img1.close()
    img2.close()
    img3.close()
    # x1d.close()

    return None


def oneD_spec_plot_wrapper(detector, segid, host_segid, exptime):
    # ---------------
    # Read in extracted spectra
    x1d = fits.open(results_dir + 'romansim_prism_Y106_0_'
                    + str(detector) + '_' + exptime + '_x1d.fits')

    # ---------------
    # Read in sedlst
    sedlst_path = pylinear_lst_dir + 'sed_Y106_0_' + str(detector) + '.lst'
    sedlst = np.genfromtxt(sedlst_path, dtype=None,
                           names=['segid', 'sed_path'], encoding='ascii')

    # Get SN mag from inserted objects catalog
    insert_npy_path = img_sim_dir + '5deg_Y106_0_' +\
        str(detector) + '_SNadded.npy'
    insert_cat = np.load(insert_npy_path)
    all_inserted_segids = insert_cat[:, -1].astype(np.int64)

    sn_idx = int(np.where(all_inserted_segids == segid)[0])
    matched_segid = int(insert_cat[sn_idx][-1])

    assert matched_segid == segid

    snmag = float(insert_cat[sn_idx][2])

    # ---------------
    paperfigure(x1d, segid, host_segid, snmag, sedlst,
                detector, exptime)

    # ---------------
    x1d.close()

    return None


if __name__ == '__main__':
    """
    This code takes a detector and segid as input
    and plots the 2D and 1D spectra for visualization.

    It will first save a regions file corresponding to
    the segid for the 2D spectrum. Then we use matplotib
    and numpy to plot the 2D spectrum contained within
    the region. This is done per roll angle.
    """
    
    detector = int(sys.argv[1])
    segid = int(sys.argv[2])
    host_segid = int(sys.argv[3])
    exptime = sys.argv[4]

    print('Saving 2D spectrum region and plotting...')
    print('Detector:', detector)
    print('SegID:', segid)
    print('Exposure Time:', exptime)

    # save_segid_2Dspec_region(detector, segid, exptime, SN=True)
    # print('2D spectrum regions saved.')

    # Also plot 1D spectra
    oneD_spec_plot_wrapper(detector, segid, host_segid, exptime)

    sys.exit(0)
