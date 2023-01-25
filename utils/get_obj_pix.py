import numpy as np
from astropy.io import fits

def get_obj_pix(xc, yc, dir_img_name):

    # Get the data and the central value
    dat = fits.getdata(dir_img_name)

    xc = int(xc)
    yc = int(yc)

    row = yc - 1
    col = xc - 1

    central_val = dat[row, col]  # this may or may not be the largest value

    # now in a box centered on the obj 
    # find all pix that are within 10% of hte max
    boxsize = 3
    box = dat[row-boxsize:row+boxsize+1, col-boxsize:col+boxsize+1]

    box_max = np.max(box)
    box_thresh = 0.25 * box_max

    box_invalid = np.ma.masked_where(box >= box_thresh, box)
    box_invalid_idx = np.ma.getmask(box_invalid)

    # Convert back to full array indices
    valid_rows = []
    valid_cols = []

    for i in range(2*boxsize+1):
        for j in range(2*boxsize+1):

            if box_invalid_idx[i,j]:
                valid_rows.append(row - boxsize + i)
                valid_cols.append(col - boxsize + j)

    valid_rows = np.array(valid_rows)
    valid_cols = np.array(valid_cols)

    obj_pix = np.array([valid_rows, valid_cols])

    return obj_pix