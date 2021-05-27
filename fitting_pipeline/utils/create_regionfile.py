import numpy as np

def create_regions(ra, dec, reg_file, text=None):

    with open(reg_file, 'w') as fh:

        fh.write("# Region file format: DS9 version 4.1" + "\n")
        fh.write("global color=green dashlist=8 3 width=1 font=\"helvetica 10 normal roman\" ")
        fh.write("select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 ")
        fh.write("delete=1 include=1 source=1" + "\n")
        fh.write("fk5" + "\n")

        for i in range(len(ra)):
            # region size is 1 arcsec written in degrees
            if isinstance(text, np.ndarray) or isinstance(text, list):

                fh.write("circle(" + \
                         "{:.7f}".format(ra[i])  + "," + \
                         "{:.7f}".format(dec[i]) + "," + \
                         "0.0002778" + "\") # color=green" + \
                         " text={" + str(text[i]) + "}" + " width=2" + "\n")

            else:
                fh.write("circle(" + \
                         "{:.7f}".format(ra[i])  + "," + \
                         "{:.7f}".format(dec[i]) + "," + \
                         "0.0002778" + "\") # color=green" + \
                         " width=2" + "\n")
            
    
    print("Written:", reg_file)

    return None