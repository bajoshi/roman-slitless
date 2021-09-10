import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    # Read in spectra from Justin
    # Both have the same length and phases
    highv = np.genfromtxt('highv_Ia.dat', dtype=None, names=['phase','lam','flam'], encoding='ascii')
    lowv  = np.genfromtxt('lowv_Ia.dat',  dtype=None, names=['phase','lam','flam'], encoding='ascii')

    hday0_idx = np.where(highv['phase'] == 0)[0]
    lday0_idx = np.where(lowv['phase'] == 0)[0]

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(highv['lam'][hday0_idx], highv['flam'][hday0_idx], color='tab:red')
    ax.plot(lowv['lam'][lday0_idx], lowv['flam'][lday0_idx], color='tab:blue')

    ax.set_xlim(2000, 10000)

    plt.show()
