def get_total_extensions(fitsfile):
    """
    This function will return the number of extensions in a fits file.
    It does not count the 0th extension.

    It takes the opened fits header data unit as its only argument.
    """

    nexten = 0 # this is the total number of extensions
    while 1:
        try:
            if fitsfile[nexten+1]:
                nexten += 1
        except IndexError:
            break

    return nexten
