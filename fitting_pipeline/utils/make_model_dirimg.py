import numpy as np
from astropy.io import fits


def gen_model_img(dirimg_path, segmap_path, save=False, data_ext=0):

    # Open file
    dhdu = fits.open(dirimg_path)
    shdu = fits.open(segmap_path)

    # Get direct image and segmap data
    ddat = dhdu[data_ext].data
    dhdr = dhdu[data_ext].header
    sdat = shdu[0].data

    # Close HDUs
    dhdu.close()
    shdu.close()

    # Get indices where no sources are detected
    backidx = np.where(sdat == 0)

    # Estimate background.
    # Just to see the value. Not actually used.
    backest = np.mean(ddat[backidx])
    print('Estimated (mean) background:', backest)

    # Force pix at less than 5 times backest
    # to exactly zero
    model_img = ddat
    # zeroidx = np.where(model_img < 5 * backest)
    model_img[backidx] = 0.0

    if save:
        new_hdu = fits.PrimaryHDU(data=model_img, header=dhdr)
        new_flname = dirimg_path.replace('.fits', '_model.fits')
        new_hdu.writeto(new_flname, overwrite=True)
        print('Saved:', new_flname)

        return None

    else:
        return model_img
