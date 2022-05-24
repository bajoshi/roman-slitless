import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
import warnings
import os


def expx(x, *p):
    a, b = p
    return(a*np.exp(b*x))


def twosidebell(x, y, splits=1, check=False):
    y = abs(y)
    #bins = stats.mstats.mquantiles(x,np.arange(0,1+splits, splits).tolist())
    bins = np.arange(0, np.nanmax(x), splits)
    marks = []
    for i in range(len(bins)):
        if i==0: continue
        idx = np.where((x >bins[i-1]) & (x <= bins[i]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mark = np.nanstd(y.loc[idx])
            iidx = np.where(y > 5.*mark) ## some 5-sigma clipping
            mark = np.nanstd(y.loc[set(idx[0])-set(iidx[0])])
            if check: import pdb; pdb.set_trace()
        marks.append(mark)
    marks = np.array(marks)
    idx= np.where(marks==marks)
    p0 = [1.,-1.]
    popt,pcov = curve_fit(expx, bins[1:][idx]-0.5*splits, marks[idx], p0=p0, check_finite=False)
    print(popt)
    
    if check: import pdb; pdb.set_trace()
    xx =  bins-0.5*splits
    yy = expx(xx, *popt)
    return(xx,yy)


home = os.getenv('HOME')
extdir = "/Volumes/Joshi_external_HDD/Roman/"

roman_slitless_dir = home + '/Documents/GitHub/roman-slitless/'
ext_spectra_dir = extdir + "roman_slitless_sims_results/"
fitting_resdir = ext_spectra_dir + 'fitting_results/'

resfile = fitting_resdir + 'zrecovery_pylinear_sims_pt0.txt'
cat = np.genfromtxt(resfile, dtype=None, names=True, encoding='ascii')

###########################################
# PREP
# -------------------
# Colors from colorbrewer
exp_400s_color = '#7570b3'
exp_1200s_color = '#d95f02'
exp_3600s_color = '#1b9e77'

# True params
ztrue = cat['z_true']
phase_true = cat['phase_true']
av_true = cat['Av_true']

# SNR
snr400 = cat['SNR400']
snr1200 = cat['SNR1200']
snr3600 = cat['SNR3600']

# Remove invalid measures
z400 = cat['z400']
z400[z400 == -9999.0] = np.nan

z1200 = cat['z1200']
z1200[z1200 == -9999.0] = np.nan

z3600 = cat['z3600']
z3600[z3600 == -9999.0] = np.nan

# ---
phase400 = cat['phase400']
phase400[phase400 == -9999.0] = np.nan

phase1200 = cat['phase1200']
phase1200[phase1200 == -9999.0] = np.nan

phase3600 = cat['phase3600']
phase3600[phase3600 == -9999.0] = np.nan

# ---
av400 = cat['Av400']
av400[av400 == -9999.0] = np.nan

av1200 = cat['Av1200']
av1200[av1200 == -9999.0] = np.nan

av3600 = cat['Av3600']
av3600[av3600 == -9999.0] = np.nan

# --------------
# Make these plots only for the uncontaminated SNe
overlap_idx = cat['overlap']

z400[overlap_idx] = np.nan
z1200[overlap_idx] = np.nan
z3600[overlap_idx] = np.nan

phase400[overlap_idx] = np.nan
phase1200[overlap_idx] = np.nan
phase3600[overlap_idx] = np.nan

av400[overlap_idx] = np.nan
av1200[overlap_idx] = np.nan
av3600[overlap_idx] = np.nan


### preserving the data
df = pd.DataFrame()
cnt = 0
df.insert(cnt, 'snr400', snr400, True)
df.insert(cnt+1, 'snr1200', snr1200, True)
df.insert(cnt+1, 'snr3600', snr3600, True)

df.insert(cnt+1, 'z400', z400, True)
df.insert(cnt+1, 'z1200', z1200, True)
df.insert(cnt+1, 'z3600', z3600, True)
df.insert(cnt+1, 'ztrue', ztrue, True)

df.insert(cnt+1, 'phase400', phase400, True)
df.insert(cnt+1, 'phase1200', phase1200, True)
df.insert(cnt+1, 'phase3600', phase3600, True)
df.insert(cnt+1, 'phase_true', phase_true, True)

df.insert(cnt+1, 'av400', av400, True)
df.insert(cnt+1, 'av1200', av1200, True)
df.insert(cnt+1, 'av3600', av3600, True)
df.insert(cnt+1, 'av_true', av_true, True)



# --------------------
# Thin all points to some fraction of total points.
# This helps to see the distributions better.
# We will use np.random.choice to pick the indices
# that will be plotted.
select_frac = 0.2
size = int(len(cat) * select_frac)
print('Plotting', size, 'points.')
idx_toplot = np.random.choice(np.arange(len(cat)), size=size)

ztrue = ztrue[idx_toplot]
phase_true = phase_true[idx_toplot]
av_true = av_true[idx_toplot]

z400 = z400[idx_toplot]
z1200 = z1200[idx_toplot]
z3600 = z3600[idx_toplot]
phase400 = phase400[idx_toplot]
phase1200 = phase1200[idx_toplot]
phase3600 = phase3600[idx_toplot]
av400 = av400[idx_toplot]
av1200 = av1200[idx_toplot]
av3600 = av3600[idx_toplot]

snr400 = snr400[idx_toplot]
snr1200 = snr1200[idx_toplot]
snr3600 = snr3600[idx_toplot]

###########################################
# -------------------- Plot SNR vs % accuracy
# get accuracy
z400acc = (z400 - ztrue) / (1 + ztrue)
z1200acc = (z1200 - ztrue) / (1 + ztrue)
z3600acc = (z3600 - ztrue) / (1 + ztrue)

fig = plt.figure(figsize=(9, 5))

gs = fig.add_gridspec(nrows=12, ncols=1, left=0.15, right=0.95, wspace=0.1)

ax1 = fig.add_subplot(gs[:4])
ax2 = fig.add_subplot(gs[4:8])
ax3 = fig.add_subplot(gs[8:])

# Axis labels
ax1.set_ylabel(r'$\frac{z_ - z_\mathrm{true}}{1 + z_\mathrm{true}}$',
               fontsize=15)
ax2.set_ylabel(r'$\Delta \mathrm{Phase}$', fontsize=20)
ax3.set_ylabel(r'$\Delta \mathrm{A_v}$', fontsize=20)
ax3.set_xlabel(r'$\mathrm{S/N}$', fontsize=20)

# Plotting
# z
ax1.axhline(y=0.0, ls='--', lw=2.0, color='gray', zorder=1)

ax1.scatter(snr400, z400acc, s=23, color=exp_400s_color,
            facecolors='None', zorder=20)
ax1.scatter(snr1200, z1200acc, s=15, color=exp_1200s_color,
            facecolors='None', zorder=20)
ax1.scatter(snr3600, z3600acc, s=5, color=exp_3600s_color,
            facecolors='None', zorder=20)

xx, yy = twosidebell(df['snr400'], (df['z400']-df['ztrue'])/(1 + df['ztrue']))
ax1.fill_between(xx, -yy, yy, color=exp_400s_color, zorder=2, alpha=0.2)


xx, yy = twosidebell(df['snr1200'], (df['z1200']-df['ztrue'])/(1 + df['ztrue']))
ax1.fill_between(xx, -yy, yy, color=exp_1200s_color, zorder=1, alpha=0.2)

xx, yy = twosidebell(df['snr3600'], (df['z3600']-df['ztrue'])/(1 + df['ztrue']))
ax1.fill_between(xx, -yy, yy, color=exp_3600s_color, zorder=0, alpha=0.2)

# Phase
ax2.axhline(y=0.0, ls='--', lw=2.0, color='gray', zorder=1)

ax2.scatter(snr400, phase400 - phase_true,
            s=23, color=exp_400s_color,
            facecolors='None', zorder=20)
ax2.scatter(snr1200, phase1200 - phase_true,
            s=15, color=exp_1200s_color,
            facecolors='None', zorder=20)
ax2.scatter(snr3600, phase3600 - phase_true,
            s=5, color=exp_3600s_color,
            facecolors='None', zorder=20)

xx, yy = twosidebell(df['snr400'], df['phase400']-df['phase_true'])
ax2.fill_between(xx, -yy, yy, color=exp_400s_color, zorder=2, alpha=0.2)


xx, yy = twosidebell(df['snr1200'], df['phase1200']-df['phase_true'])
ax2.fill_between(xx, -yy, yy, color=exp_1200s_color, zorder=1, alpha=0.2)

xx, yy = twosidebell(df['snr3600'], df['phase3600']-df['phase_true'])
ax2.fill_between(xx, -yy, yy, color=exp_3600s_color, zorder=0, alpha=0.2)


# Av
ax3.axhline(y=0.0, ls='--', lw=2.0, color='gray', zorder=1)

ax3.scatter(snr400, av400 - av_true,
            s=23, color=exp_400s_color,
            facecolors='None', zorder=2, label='20m')
ax3.scatter(snr1200, av1200 - av_true,
            s=15, color=exp_1200s_color,
            facecolors='None', zorder=2, label='1h')
ax3.scatter(snr3600, av3600 - av_true,
            s=5, color=exp_3600s_color,
            facecolors='None', zorder=2, label='3h')

xx, yy = twosidebell(df['snr400'], df['av400']-df['av_true'])
ax3.fill_between(xx, -yy, yy, color=exp_400s_color, zorder=2, alpha=0.2)


xx, yy = twosidebell(df['snr1200'], df['av1200']-df['av_true'])
ax3.fill_between(xx, -yy, yy, color=exp_1200s_color, zorder=1, alpha=0.2)

xx, yy = twosidebell(df['snr3600'], df['av3600']-df['av_true'])
ax3.fill_between(xx, -yy, yy, color=exp_3600s_color, zorder=0, alpha=0.2)

# SNR limits
ax1.set_xlim(2, 25)
ax2.set_xlim(2, 25)
ax3.set_xlim(2, 25)

# Limit based on consideration of contam/uncontam sne
ax1.set_ylim(-0.002, 0.002)
ax2.set_ylim(-1.5, 1.5)
ax3.set_ylim(-0.15, 0.15)

ax3.legend(ncol=3, frameon=False, loc=1)
fig.savefig(roman_slitless_dir + 'figures/pylinearrecovery_snr.pdf',
            dpi=200, bbox_inches='tight')
