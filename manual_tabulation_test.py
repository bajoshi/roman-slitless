import numpy as np
import pylinear
import polyclip

import os
import sys

import matplotlib.pyplot as plt

# specific pylinear imports
from pylinear.h5table import h5table, h5tablebase

# ---------------
# My goal with this code is to understand the point 
# of and behaviour of the subsampling freq, nsub.

# ---------------
# Will load the sources and grisms in the default way
# and then do the tabulation process manually here.
# The required tabulation functions are copy pasted 
# hrere from pylinear codebase.
# Done this way so I don't have to reinstall pylinear
# if I modify something while this test is being done.

# This code is relying on the prep in sensitivity_test.py 
# being done before this code is run.
# Run it to generate the img and seg files. Save 
# the model template and create lst files.

def prep_run_pdt(grisms, sources, beam):

    for grism in grisms:
        print('\nWorking on dataset:', grism.dataset)

        for device in grism:

            # load the config file
            beamconf=device.load_beam(beam)

            # get the center of the device
            xc,yc=device.naxis1/2.,device.naxis2/2.

            # Get lambda step and contruct wav array
            dwav=device.defaults['dlamb']
            wav=np.arange(device.defaults['lamb0'],
                          device.defaults['lamb1']+dwav,dwav)

            print('dlam from instruments.xml file:', dwav)
            #print('Wavelength array for this device:', wav)

            for src in sources:
                gen_pdt(src, wav, beamconf, device)

            sys.exit(0)

    return None

def drizzle(beamconf, xd, yd, wav, band, pixfrac=1.0):

    # disperse each polygon vertix
    print('Dispersing:')
    print('X:', xd)
    print('Y:', yd)
    print('Wav:', wav)

    #print('NAXIS:', beamconf.naxis)
    #print('XRANGE:', beamconf.xr)
    #print('YRANGE:', beamconf.yr)

    #xg, yg = beamconf.disperse(xd, yd, wav, band=band)

    nx = len(xd)
    nw = len(wav)
    xg = np.empty((nw,nx))
    yg = np.empty((nw,nx))
    
    for i,xyd in enumerate(zip(xd,yd)):
        print(i, xyd)
        t = beamconf.displ.invert(xyd, wav)
        #print(t, len(t))
        #print(type(beamconf.displ))
        #print(dir(beamconf.displ))
        #print('\n')
        #print(beamconf.displ.polys[0].coefs)
        #print(beamconf.displ.polys[0].order)
        #print(beamconf.displ.polys[1].coefs)
        #print(beamconf.displ.polys[1].order)
        #print(type(beamconf.dispx))

        xg[:,i] = beamconf.dispx.evaluate(xyd,t) + xyd[0]
        yg[:,i] = beamconf.dispy.evaluate(xyd,t) + xyd[1]
    
    print('Grism image coords:')
    print(xg)
    print(yg)
    print(xg.shape, yg.shape)

    # apply clipping
    xg = np.clip(xg, 0, beamconf.naxis[0])
    yg = np.clip(yg, 0, beamconf.naxis[1])

    # Figure and print statement below will show
    # where each of the four corners of a pixel 
    # will disperse
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.plot(np.append(xd, xd[0]), np.append(yd, yd[0]), lw=4.0, color='r')
    #xg_plt = []
    #yg_plt = []
    #for j in range(4):
    #    print('\n--------------------------')
    #    for i in range(len(xg)):
    #        xg_plt.append(xg[i, j])
    #        yg_plt.append(yg[i, j])
    #        print(xg[i, j], yg[i, j])
    #ax.plot(xg[:, 0], yg[:, 0], lw=2.0, color='b')
    #ax.plot(xg[:, 1], yg[:, 1], lw=2.0, color='b')
    #ax.plot(xg[:, 2], yg[:, 2], lw=2.0, color='b')
    #ax.plot(xg[:, 3], yg[:, 3], lw=2.0, color='b')
    #fig.savefig(testdir + 'pix_disp.pdf')

    # clip against a pixel grid
    clip = polyclip.Polyclip(beamconf.naxis)
    x, y, area, indices = clip(xg, yg)
    
    print(len(x))
    
    #print(area)
    #print(indices)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(len(area)), area, lw=0.2)
    fig.savefig(testdir + 'pix_area_clip.pdf')

    i0 = indices[0:-1]
    i1 = indices[1:]
    gg = np.where(i1 != i0)[0]
    lam = np.empty(len(x), dtype=np.uint16)
    for g,a,b in zip(gg,i0[gg],i1[gg]):
        lam[a:b] = g

    print(lam)

    return x, y, lam, area

def disperse_mod(xd,yd,wav, beamconf):
    
    nx=len(xd)
    nw=len(wav)
    xg=np.empty((nw,nx))
    yg=np.empty((nw,nx))

    for i,xyd in enumerate(zip(xd,yd)):
        t = displ.invert(xyd,wav)
        xg[:,i] = dispx.evaluate(xyd,t)+xyd[0]
        yg[:,i] = dispy.evaluate(xyd,t)+xyd[1]

    # implement the wedge offset
    #if band is not None and band in self.wedge:
    #    # NOR HAS THIS A NEGATIVE SIGN
    #    xg-=self.wedge[band][0]
    #    yg-=self.wedge[band][1]
        
    return xg,yg

def gen_pdt(src, wav, beamconf, device):

    dwav = wav[1] - wav[0]  # Should be the exact same as dwav above
    #filtname = sources.obscat.detband.name  # filtname hardcoded for now
    filtname = 'F105W'

    DX = np.array([-0.5,-0.5,+0.5,+0.5], dtype=np.float64)
    DY = np.array([-0.5,+0.5,+0.5,-0.5], dtype=np.float64)

    # default nsub
    nsub = 10
    
    pixfrac = 1.0    # DO NOT CHANGE THIS VALUE

    # Empty list of pdts for each source
    pdts = []

    # compute ratio of pixel area between the FLT and the source
    pixrat = device.pixelarea / (pixfrac*src.pixelarea)

    #print('Source pix area (what does this mean?):', src.pixelarea)

    # process each pixel in the source
    pixcount = 0
    for xd,yd,wd in src:

        # The x,y here follows ds9 x,y convention
        # bottom left is 0,0
        # Can confirm by pulling segmap up in ds9
        # It is NOT image x,y coords
        # Does not seem they are relative to image center either?? 
        print('\n' + '{:d}'.format(xd), 
                     '{:d}'.format(yd), 
                     '{:.4f}'.format(wd))
        pixcount += 1

        # convert the corners of the direct image to the
        # corresponding grism image
        xg,yg = src.xy2xy(xd+DX, yd+DY, device)
        # These above grism coordinates correspond to 
        # image coordinates. The above func basically
        # figures out where the four corners of each pixel
        # will be in the dispersed image.
        with np.printoptions(precision=2):
            print(xg, yg)

        print(beamconf.wedge)

        # drizzle these corners
        x,y,l,v = drizzle(beamconf, xg, yg, wav, filtname)

        sys.exit(0)

        #print(x, y, l, v)
        # These are x, y, wavelength index (relative to wav 
        # array above) and value from the spectrum. I do not
        # think that the value has been scaled yet from the 
        # spectrum to match the broadband mag.

        #print(v)
        #print(v*pixrat*dwav)

        # Check by plotting the pixel areas 
        # This should not look weird
        # it will look wrong if any of the stuff in the conf file is wrong
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.arange(len(v)), v, lw=0.5)
        fig.savefig(testdir + 'pix_' + str(xd) + '_' + str(yd) + '.pdf')

        sys.exit(0)

        # define the pixel coordinate
        pix = (int(xd-src.ltv[0]), int(yd-src.ltv[1]))
        print(pix)

        # create the table
        pdt = h5table.PDT(pix,x,y,l,v*pixrat*dwav)

    print('\nTotal pixels in source:', pixcount)

    return None

# END OF FUNCTION DEFS
########################################

testdir = '/Volumes/Joshi_external_HDD/Roman/sensitivity_test/'

segfile = testdir + 'seg.fits'
imgfile = testdir + 'img.fits'

# Pylinear settings 
maglim = 30.0
roll_angles = [0.0, 5.0, 10.0]
obslst = 'obs.lst'
wcslst = 'wcs.lst'
sedlst = 'sed.lst'
fltlst = 'flt.lst'
beam = '+1'

# cd for pylinear run
os.chdir(testdir)

# Load sources
sources = pylinear.source.SourceCollection(segfile, obslst, detindex=0, maglim=maglim)

# Load grisms
grisms = pylinear.grism.GrismCollection(wcslst, observed=False)

# Initialize directory 
path = 'tables'
if not os.path.isdir(path):
    os.mkdir(path)

# Prep and run the pdt func
prep_run_pdt(grisms, sources, beam)









