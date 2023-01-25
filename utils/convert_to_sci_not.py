def convert_to_sci_not(n, sigfigs=2):

    # convert to python string with sci notation
    sigfig_str = "{:." + str(sigfigs) + "e}"
    n_str = sigfig_str.format(n)

    # split string and assign parts
    n_splt = n_str.split('e')
    decimal = n_splt[0]
    exponent = n_splt[1]

    # strip leading zeros in exponent
    if float(exponent) < 0:
        exponent = exponent.split('-')[1]
        exponent = '-' + exponent.lstrip('0')
    elif float(exponent) > 0:
        exponent = exponent.lstrip('+')  # also remove the + sign that's not required
        exponent = exponent.lstrip('0')
    
    # create final string with proper TeX sci notation and return
    if float(exponent) == 0.0:
        sci_str = decimal
    else:
        sci_str = decimal + r'$\times$' + r'$10^{' + exponent + r'}$'

    return sci_str