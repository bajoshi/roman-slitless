import numpy as np


def create_regions(ra, dec, reg_file, color='green', width=1, text=None):

    with open(reg_file, 'w') as fh:

        fh.write("# Region file format: DS9 version 4.1" + "\n")
        fh.write("global ")
        fh.write("color=" + color)
        fh.write(" dashlist=8 3 ")
        fh.write("width=" + str(width))
        fh.write(" font=\"helvetica 10 normal roman\" ")
        fh.write("select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 ")
        fh.write("delete=1 include=1 source=1" + "\n")
        fh.write("fk5" + "\n")

        for i in range(len(ra)):
            # region radius is 1 arcsec written in degrees
            if isinstance(text, np.ndarray) or isinstance(text, list):

                fh.write("circle(" +
                         "{:.7f}".format(ra[i]) + "," +
                         "{:.7f}".format(dec[i]) + "," +
                         "0.0002778" + ") # color=" + color +
                         " text={" + str(text[i]) + "}" +
                         " width=" + str(width) + "\n")

            else:
                fh.write("circle(" +
                         "{:.7f}".format(ra[i]) + "," +
                         "{:.7f}".format(dec[i]) + "," +
                         "0.0002778" + ") # color=" + color +
                         " width=" + str(width) + "\n")

    print("Written:", reg_file)

    return None
