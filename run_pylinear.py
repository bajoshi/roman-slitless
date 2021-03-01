import pylinear
from astropy.io import fits
import numpy as np

import os
import sys
import time
import datetime as dt
import glob
import shutil
import socket

def create_lst_files(machine, lst_dir, img_suffix, exptime):

    return None

def main():

    # ---------------------- Preliminary stuff
    # Get starting time
    start = time.time()
    print("Starting at:", dt.datetime.now())
    
    """
    import subprocess 
    
    lscpu = subprocess.Popen(['lscpu'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    nproc = subprocess.Popen(['vmstat', '-s'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    lscpu_out, lscpu_err = lscpu.communicate()
    nproc_out, nproc_err = nproc.communicate()
    
    print(lscpu_out)
    print(lscpu_err)
    print(nproc_out)
    print(nproc_err)
    
    sys.exit(0)
    """
    
    # Change directory to make sure results go in the right place
    home = os.getenv('HOME')
    
    if 'compute' in socket.gethostname():
        os.chdir(home + '/scratch/roman_slitless_sims_results/')
        # Define directories for imaging and lst files
        pylinear_lst_dir = home + '/scratch/roman-slitless/pylinear_lst_files/'
        direct_img_dir = home + '/scratch/roman_direct_sims/K_akari_rotate_subset/'
    
        path = home + '/scratch/roman_slitless_sims_results/tables'
        
        obsstr = '_marcc'
    
    elif 'plffsn2' in socket.gethostname():
        os.chdir(home + '/Documents/roman_slitless_sims_results/')
        # Define directories for imaging and lst files
        pylinear_lst_dir = home + '/Documents/GitHub/roman-slitless/pylinear_lst_files/'
        direct_img_dir = home + '/Documents/roman_direct_sims/K_akari_rotate_subset/'
    
        path = home + '/Documents/roman_slitless_sims_results/tables'
    
        obsstr = '_plffsn2'
    
    # Figure out the correct filenames depending on which machine is being used
    img_suffix_list = ['Y106_11_1']
    exptime_list = [300, 600, 900, 1800, 3600]
    
    sim_count = 0
    
    for img in img_suffix_list:
    
        img_suffix = img_suffix_list[sim_count]
    
        # Define list files and other preliminary stuff
        segfile = direct_img_dir + 'akari_match_' + img_suffix + '_segmap.fits'
    
        obslst = pylinear_lst_dir + 'obs_' + img_suffix + obsstr + '.lst'
        wcslst = pylinear_lst_dir + 'wcs_' + img_suffix + '.lst'
        sedlst = pylinear_lst_dir + 'sed_' + img_suffix + obsstr + '.lst'
        beam = '+1'
        maglim = 99.0
    
        # make sure the files exist
        assert os.path.isfile(segfile)
        assert os.path.isfile(obslst)
        assert os.path.isfile(sedlst)
        assert os.path.isfile(wcslst)
    
        print("Using the following paths to lst files and segmap:")
        print("Segmentation map:", segfile)
        print("OBS LST:", obslst)
        print("SED LST:", sedlst)
        print("WCS LST:", wcslst)
    
        # ---------------------- Get sources
        sources = pylinear.source.SourceCollection(segfile, obslst, detindex=0, maglim=maglim)
    
        # Set up and tabulate
        grisms = pylinear.grism.GrismCollection(wcslst, observed=False)
        tabulate = pylinear.modules.Tabulate('pdt', ncpu=0) 
        tabnames = tabulate.run(grisms, sources, beam)
    
        ## ---------------------- Simulate
        print("Simulating...")
        simulate = pylinear.modules.Simulate(sedlst, gzip=False, ncpu=0)
        fltnames = simulate.run(grisms, sources, beam)
        print("Simulation done.")
    
        for e in range(len(exptime_list)):
            
            # ---------------------- Add noise
            print("Adding noise...")
            # check Russell's notes in pylinear notebooks
            # also check WFIRST tech report TR1901
            sky = 1.1      # e/s
            npix = 4096 * 4096
            sky /= npix    # e/s/pix
    
            dark = 0.015   # e/s/pix
            read = 10.0    # electrons
    
            exptime = exptime_list[e]  # seconds
    
            #for oldf in glob.glob('*_flt.fits'):
                print("Working on...", oldf)
                print("Putting in an exposure time of:", exptime, "seconds.")
    
                #if e == 0:
                #    # let's save the file in case we want to compare
                #    savefile = oldf.replace('_flt', '_flt_noiseless')
                #    shutil.copyfile(oldf, savefile)
    
                # open the fits file
                with fits.open(oldf) as hdul:
                    sci = hdul[('SCI',1)].data    # the science image
                    size = sci.shape              # dimensionality of the image
    
                    # update the science extension with sky background and dark current
                    signal = (sci + sky + dark)
    
                    # Handling of pixels with negative signal
                    #neg_idx = np.where(sci < 0.0)
                    #sci[neg_idx] = 0.0  # This is wrong but should allow the rest of the program to work for now
    
                    # Multiply the science image with the exptime
                    # sci image originally in electrons/s
                    signal = signal * exptime  # this is now in electrons
    
                    # Randomly vary signal about its mean. Assuming Gaussian distribution
                    # first get the uncertainty
                    variance = signal + read**2
                    sigma = np.sqrt(variance)
                    new_sig = np.random.normal(loc=signal, scale=sigma, size=size)
    
                    # now divide by the exptime and subtract the sky again 
                    # to get back to e/s. LINEAR expects a background subtracted image
                    final_sig = (new_sig / exptime) - sky
    
                    # Assign updated sci image to the first [SCI] extension
                    hdul[('SCI',1)].data = final_sig
    
                    # update the uncertainty extension with the sigma
                    err = np.sqrt(signal) / exptime
    
                    hdul[('ERR',1)].data = err
    
                    # now write to a new file name
                    newfilename = oldf.replace('_flt', '_' + str(exptime) + 's' + '_flt')
                    hdul.writeto(newfilename, overwrite=True)
                    
                print("Written:", newfilename)
            
            print("Noise addition done. Check simulated images.")
            ts = time.time()
            print("Time taken for simulation:", "{:.2f}".format(ts - start), "seconds.")
    
            # ---------------------- Extraction
            fltlst = pylinear_lst_dir + 'flt_' + img_suffix + '_' + str(exptime) + 's' + obsstr + '.lst'
            assert os.path.isfile(fltlst)
            print("FLT LST:", fltlst)
    
            grisms = pylinear.grism.GrismCollection(fltlst, observed=True)
            tabulate = pylinear.modules.Tabulate('pdt', path=path, ncpu=0)
            tabnames = tabulate.run(grisms, sources, beam)
    
            extraction_parameters = grisms.get_default_extraction()
    
            print('\nDefault parameters: range = {lamb0}, {lamb1} A, sampling = {dlamb} A'.format(**extraction_parameters))
    
            # Set extraction params
            sources.update_extraction_parameters(**extraction_parameters)
            method = 'golden'  # golden, grid, or single
            root = 'romansim_' + img_suffix + '_' + str(exptime) + 's'
            logdamp = [-7, -1, 0.1]
    
            print("Extracting...")
            pylinear.modules.extract.extract1d(grisms, sources, beam, logdamp, method, root, path, \
                inverter='lsqr', ncpu=0, group=False)
    
            print("Simulation and extraction done.")
            try:
                te = time.time() - ts
                print("Time taken for extraction:", "{:.2f}".format(te), "seconds.")
            except NameError:
                print("Finished at:", dt.datetime.now())
    
        # Increment simulation counter
        # This only increments after all the exptimes 
        # for a given direct image are simulated
        sim_count += 1
    
    print("Total time taken:", "{:.2f}".format(time.time() - start), "seconds.")

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)




