# Galsim demo13 script edited by BAJ
# Edited to be viewed side-by-side with
# the original demo13.py script. 
# Editing to learn more about GalSim working.

import numpy as np

import galsim
import galsim.roman as roman

import argparse
import os
import sys


def parse_args(argv):
    parser = argparse.ArgumentParser(prog='demo13', add_help=True)
    parser.add_argument('-f', '--filters', type=str, default='Y', 
                        action='store',
                        help='Which filters to simulate (default = "Y")')
    parser.add_argument('-o', '--outpath', type=str, default='output',
                        help='Which directory to put the output files')
    parser.add_argument('-n', '--nobj', type=int, default=1500,
                        help='How many objects to draw')
    parser.add_argument('--seed', type=int, default=12345,
                        help='Initial seed for random numbers')
    parser.add_argument('-s', '--sca', type=int, default=1, 
                        choices=range(1, 19),
                        help='Which SCA to simulate '
                        + '(default is arbitrarily SCA 1)')
    parser.add_argument('-t', '--test', action='store_const', 
                        default=False, const=True,
                        help='Whether to use the smaller test sample, '
                        + 'rather than the full COSMOS samples')
    parser.add_argument('-v', '--verbosity', type=int, 
                        default=2, choices=range(0, 4),
                        help='Verbosity level')

    args = parser.parse_args(argv)
    return args


def get_galsim_stellar_SED():

    pickles_lib_path = '/Volumes/Joshi_external_HDD/' + \
        'Pickles_stellar_library/for_romansim/'

    # Randomly choses a stellar spectrum
    all_stars = ['a5v', 'b5iii', 'f0v', 'g2v', 'k3i', 'm4v', 'o5v']
    star_chosen = np.random.choice(all_stars)

    pickles_spec_path = pickles_lib_path + 'uk' + star_chosen + '.dat'



    return galsim_SED


if __name__ == '__main__':

    # ---------------------------------------------------
    # READ IN ARGS AND OTHER PREP
    # ---------------------------------------------------

    argv = sys.argv[1:]

    args = parse_args(argv)
    use_filters = args.filters
    outpath = args.outpath
    nobj = args.nobj
    seed = args.seed
    use_SCA = args.sca

    print('Default GalSim values changed in this edited program.')
    print('Default exposure time in GalSim Roman module:', roman.exptime, '\n')

    # Make output directory if not already present.
    if not os.path.isdir(outpath):
        os.mkdir(outpath)

    # ---------------------------------------------------
    # FILTER RELATED STUFF
    # ---------------------------------------------------
    roman_filters = roman.getBandpasses(AB_zeropoint='AB')
    # The ZP in the args above is the mag system ZP, I think...

    print(type(roman_filters))
    print(roman_filters.keys())
    print(roman_filters['Y106'], '\n')

    filters = []
    for filter_name in roman_filters:
        if filter_name[0] in use_filters:
            filters.append(filter_name)

    # We'll use this one for our flux normalization 
    # of stars, so we'll need this regardless of
    # which bandpass we are simulating.
    y_bandpass = roman_filters['Y106']

    # ---------------------------------------------------
    # Generate fake image
    # ---------------------------------------------------
    # I would like to do this by passing a numpy array
    # because the 2D images from pylinear which will be
    # noised will be numpy arrays.
    img_arr = np.ones((roman.n_pix, roman.n_pix))

    image = galsim.Image(img_arr)

    # print(dir(image))

    # Write to test files and check with ds9
    outdir = os.getcwd() + '/galsim_test_out/'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    image.write('galsim_test_out/base_image.fits')

    # Now noise the image and write that too
    roman.allDetectorEffects(image)
    image.write('galsim_test_out/noised_image.fits')

    sys.exit(0)
