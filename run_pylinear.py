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

def create_wcs_lst(lst_dir, img_suffix, roll_angle_list, \
    simroot, ra_cen, dec_cen, disp_elem):

    # Format the ra dec
    ra_cen_fmt = "{:.7f}".format(ra_cen)
    dec_cen_fmt = "{:.7f}".format(dec_cen)

    # Write list
    wcs_filename = 'wcs_' + img_suffix + '.lst'

    with open(lst_dir + wcs_filename, 'w') as fh:
        fh.write("# TELESCOPE = Roman" + "\n")
        fh.write("# INSTRUMENT = WFI" + "\n")
        fh.write("# DETECTOR = WFI" + "\n")
        fh.write("# GRISM = " + disp_elem + "\n")
        fh.write("# BLOCKING = " + "\n")

        for r in range(len(roll_angle_list)):

            roll_angle = "{:.1f}".format(roll_angle_list[r])

            str_to_write = "\n" + simroot + str(r+1) + '_' + img_suffix + '  ' + \
            ra_cen_fmt + '  ' + dec_cen_fmt + '  ' + roll_angle + '  ' + disp_elem
            
            fh.write(str_to_write)

    print("Written WCS LST:", wcs_filename)

    return None

def create_obs_lst(lst_dir, dir_img_path, dir_img_filt, dir_img_name, \
    img_suffix, machine):

    # Write list
    obs_filename = 'obs_' + img_suffix + machine + '.lst'

    with open(lst_dir + obs_filename, 'w') as fh:
        fh.write("# Image File name" + "\n")
        fh.write("# Observing band" + "\n")

        fh.write("\n" + dir_img_path + dir_img_name + '  ' + dir_img_filt)

    print("Written OBS LST:", obs_filename)

    return None

def create_sed_lst(lst_dir, seds_path, img_suffix, machine):

    # After a base sed lst has been generated edit the 
    # paths so that they will work on different machines.
    sedlst_basefilename = 'sed_' + img_suffix + '.lst'
    sedlst_filename = lst_dir + 'sed_' + img_suffix + machine + '.lst'

    if not os.path.isfile(sedlst_basefilename):
        print("Cannot find file:", sedlst_basefilename)
        print("First run the gen_sed_lst program to generate")
        print("a base sed lst whose paths can then be changed.")
        print("Exiting.")
        sys.exit(0)

    fh0 = open(sedlst_basefilename, 'r')
    fh = open(sedlst_filename, 'w')

    fh.write("# 1: SEGMENTATION ID" + "\n")
    fh.write("# 2: SED FILE" + "\n")

    for l in fh0.readlines():

        if 'roman_slitless_sims_seds' in l:

            la = l.split('/')
            ln = la[0] + seds_path + la[-1]

            fh.write(ln)

    fh.close()
    fh0.close()

    print("Written SED LST:", 'sed_' + img_suffix + machine + '.lst')

    return None

def create_flt_lst(lst_dir, result_path, simroot, img_suffix, exptime_list, \
    machine, roll_angle_list):

    # There is a unique flt lst for each exptime
    # Also unique to machine and direct image
    # Loop over all exptimes
    for t in exptime_list:

        # Assign name and write list
        flt_filename = 'flt_' + img_suffix + '_' + str(t) + 's' + machine + '.lst'

        with open(lst_dir + flt_filename, 'w') as fh:
            fh.write("# Path to each flt image" + "\n")
            fh.write("# This has to be a simulated or observed dispersed image" + "\n")

            for r in range(len(roll_angle_list)):

                str_to_write = '\n' + result_path + simroot + str(r+1) + '_' + img_suffix + '_' + str(t) + 's_flt.fits'
                fh.write(str_to_write)

        print("Written FLT LST:", flt_filename)

    return None

def create_lst_files(machine, lst_dir, img_suffix, roll_angle_list, \
    dir_img_path, dir_img_filt, dir_img_name, seds_path, result_path, \
    exptime_list, simroot):
    """
    This function creates the lst files needed as pylinear inputs.
    It requires the following args --
      
      machine: string to identify machine that pylinear is being run on. If you're
      running pylinear only on one machine this can be left blank.

      lst_dir: path to store lst files.

      img_suffix: direct and dispersed image identifier.

      roll_angle_list: list of roll angles that will be simulated. Floats in list.

      dir_img_path: Directory for direct image.

      dir_img_filt: Observation filter for direct image.

      dir_img_name: basename for direct image.

      seds_path: Path to model templates.

      result_path: Path where results are stored.

      exptime_list: List of exposure times. ints in list.

      simroot: string prefix for simulated images.
    """

    # Get some preliminary stuff first
    h = fits.open(dir_img_path + dir_img_name)
    ra_cen = float(h[0].header['CRVAL1'])
    dec_cen = float(h[0].header['CRVAL2'])

    # OBS LST
    create_obs_lst(lst_dir, dir_img_path, dir_img_filt, dir_img_name, img_suffix, machine)

    # WCS LST
    create_wcs_lst(lst_dir, img_suffix, roll_angle_list, simroot, ra_cen, dec_cen, 'G150')

    # FLT LST
    create_flt_lst(lst_dir, result_path, simroot, img_suffix, exptime_list, machine, roll_angle_list)

    # SED LST
    create_sed_lst(lst_dir, seds_path, img_suffix, machine)

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
        # Define path for results and change to that directory
        result_path = home + '/scratch/roman_slitless_sims_results/'
        
        # Define directories for imaging and lst files
        pylinear_lst_dir = home + '/scratch/roman-slitless/pylinear_lst_files/'
        dir_img_dir = home + '/scratch/roman_direct_sims/K_akari_rotate_subset/'

        # Define paths for tables
        tablespath = home + '/scratch/roman_slitless_sims_results/tables/'

        # Define path for SEDs
        seds_path = home + '/scratch/roman_slitless_sims_seds/'
        
        # Define identifier for machine
        obsstr = '_marcc'
    
    elif 'plffsn2' in socket.gethostname():
        # Define path for results and change to that directory
        result_path = home + '/Documents/roman_slitless_sims_results/'
        
        # Define directories for imaging and lst files
        pylinear_lst_dir = home + '/Documents/GitHub/roman-slitless/pylinear_lst_files/'
        dir_img_dir = home + '/Documents/roman_direct_sims/K_akari_rotate_subset/'

        # Define paths for tables
        tablespath = home + '/Documents/roman_slitless_sims_results/tables/'

        # Define path for SEDs
        seds_path = home + '/Documents/roman_slitless_sims_seds/'
        
        # Define identifier for machine
        obsstr = '_plffsn2'
    
    # Figure out the correct filenames depending on which machine is being used
    img_suffix_list = ['Y106_11_1', 'Y106_11_2']
    exptime_list = [300, 600, 900, 1800, 3600]
    roll_angle_ll = [[165.0, 210.0, 260.0], [70.0, 130.0, 190.0]]

    dir_img_filt = 'hst_wfc3_f105w'
    simroot = 'romansim'
    
    sim_count = 0
    
    for img in img_suffix_list:
    
        img_suffix = img_suffix_list[sim_count]
        roll_angle_list = roll_angle_ll[sim_count]

        dir_img_name = 'akari_match_' + img_suffix + '_sci_counts.fits'

        # Leave commented out # Do not delete
        # Calling sequence for testing on laptop
        # create_lst_files('_plffsn2', '/Users/baj/Documents/GitHub/roman-slitless/pylinear_lst_files/', \
        #     img_suffix, roll_angle_list, \
        #     '/Users/baj/Documents/roman_direct_sims/K_akari_rotate_subset/', dir_img_filt, \
        #     'akari_match_Y106_11_1_sci_counts.fits', home + '/Documents/roman_slitless_sims_seds/', \
        #     '/Users/baj/Documents/roman_slitless_sims_results/', exptime_list, simroot)

        create_lst_files(obsstr, pylinear_lst_dir, img_suffix, roll_angle_list, \
            dir_img_dir, dir_img_filt, dir_img_name, seds_path, result_path, \
            exptime_list, simroot)

        # Change directory to where the simulation results will go
        # This MUST be done after creating lst files otherwise
        # sed lst generation will fail.
        os.chdir(result_path)

        # Define list files and other preliminary stuff
        segfile = dir_img_dir + 'akari_match_' + img_suffix + '_segmap.fits'
    
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
    
            for i in range(len(roll_angle_list)):
                oldf = simroot + str(i+1) + '_' + img_suffix + '_flt.fits'
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
            tabulate = pylinear.modules.Tabulate('pdt', path=tablespath, ncpu=0)
            tabnames = tabulate.run(grisms, sources, beam)
    
            extraction_parameters = grisms.get_default_extraction()
    
            print('\nDefault parameters: range = {lamb0}, {lamb1} A, sampling = {dlamb} A'.format(**extraction_parameters))
    
            # Set extraction params
            sources.update_extraction_parameters(**extraction_parameters)
            method = 'golden'  # golden, grid, or single
            extroot = 'romansim_' + img_suffix + '_' + str(exptime) + 's'
            logdamp = [-7, -1, 0.1]
    
            print("Extracting...")
            pylinear.modules.extract.extract1d(grisms, sources, beam, logdamp, method, extroot, tablespath, \
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

        print("Finished with first set of sims. Check results. Exiting.")
        sys.exit(0)
    
    print("Total time taken:", "{:.2f}".format(time.time() - start), "seconds.")

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)




