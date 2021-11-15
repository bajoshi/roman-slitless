import os
import glob
import socket
import sys

# -----------------
if 'plffsn2' in socket.gethostname():
    extdir = '/astro/ffsn/Joshi/'
    
    roman_sims_seds = extdir + 'roman_slitless_sims_seds/'
    pylinear_lst_dir = extdir + 'pylinear_lst_files/'
    roman_direct_dir = extdir + 'roman_direct_sims/sims2021/'

    roman_slitless_dir = extdir + "GitHub/roman-slitless/"
    fitting_utils = roman_slitless_dir + "fitting_pipeline/utils/"

else:
    extdir = '/Volumes/Joshi_external_HDD/Roman/'
    
    roman_sims_seds = "/Volumes/Joshi_external_HDD/Roman/roman_slitless_sims_seds/"
    pylinear_lst_dir = "/Volumes/Joshi_external_HDD/Roman/pylinear_lst_files/"
    roman_direct_dir = extdir + 'roman_direct_sims/sims2021/'

    home = os.getenv("HOME")
    roman_slitless_dir = home + "/Documents/GitHub/roman-slitless/"
    fitting_utils = roman_slitless_dir + "fitting_pipeline/utils/"

assert os.path.isdir(roman_sims_seds)
assert os.path.isdir(pylinear_lst_dir)
assert os.path.isdir(roman_direct_dir)

if __name__ == '__main__':
    # 1. Remove lst files
    # 2. Remove all sed txt files
    # 3. Remove all segmap, SNadded, npy, reg, and cat files
    # --- the segmap and cat for the reference img: 5deg_Y106_0_6
    # have been moved one level up to avoid deleting them.
    # 4. Remove all txt files in surver/ folder.

    dir_img_part = 'part1'
    img_sim_dir = roman_direct_dir + 'K_5degimages_' + dir_img_part + '/'
    survey_dir = roman_slitless_dir + 'survey/'

    # ---- LST files
    print('Working on deleting LST files.')
    for fl in glob.glob(pylinear_lst_dir + '*.lst'):
        os.remove(fl)

    # ---- SED txt files
    print('Working on deleting sed txt files.')
    for fl in glob.glob(roman_sims_seds + '*.txt'):
        os.remove(fl)

    # ---- Files associated with SN insertion and SExtractor
    flext = ['_segmap.fits', '_SNadded.fits', '.npy', '.reg', '.cat']
    for ext in flext:
        print('Working on deleting:', ext)
        for fl in glob.glob(img_sim_dir + '*' + ext):
            os.remove(fl)

    # ---- Txt files with inserted SN props in survey
    os.chdir(survey_dir)
    for fl in glob.glob('inserted_sn_*.txt'):
        os.remove(fl)

    print('Finished cleanup.')

    sys.exit(0)