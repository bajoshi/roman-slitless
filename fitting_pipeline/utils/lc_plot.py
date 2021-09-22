import numpy as np

import matplotlib.pyplot as plt

import os
home = os.getenv('HOME')

roman_slitless_dir = home + '/Documents/GitHub/roman-slitless/'

def read_lc(band):

    lc = np.genfromtxt('light_curves/vectors_' + band.upper() + '.dat',
        dtype=None, names=['phase', 'absmag'], usecols=(0,1))

    return lc['phase'], lc['absmag']

if __name__ == '__main__':
    
    lc_u_phase, lc_u_absmag = read_lc('U')
    lc_b_phase, lc_b_absmag = read_lc('B')
    lc_v_phase, lc_v_absmag = read_lc('V')
    lc_r_phase, lc_r_absmag = read_lc('R')
    lc_i_phase, lc_i_absmag = read_lc('I')

    phase_idx = np.where((lc_u_phase >= -10) & (lc_u_phase <= 20))[0]

    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111)

    ax.set_xlabel('Phase', fontsize=15)
    ax.set_ylabel('Abs Mag', fontsize=15)

    ax.plot(lc_u_phase[phase_idx], lc_u_absmag[phase_idx], lw=2.5, color='violet', label='U-band')
    ax.plot(lc_b_phase[phase_idx], lc_b_absmag[phase_idx], lw=2.5, color='royalblue', label='B-band')
    ax.plot(lc_v_phase[phase_idx], lc_v_absmag[phase_idx], lw=2.5, color='seagreen', label='V-band')
    ax.plot(lc_r_phase[phase_idx], lc_r_absmag[phase_idx], lw=2.5, color='crimson', label='R-band')
    ax.plot(lc_i_phase[phase_idx], lc_i_absmag[phase_idx], lw=2.5, color='goldenrod', label='I-band')

    ax.legend(loc=0, fontsize=14, frameon=False)

    ax.set_xlim(-10, 20)

    fig.savefig(roman_slitless_dir + 'figures/mlcs_lightcurves.pdf', dpi=200, bbox_inches='tight')
    