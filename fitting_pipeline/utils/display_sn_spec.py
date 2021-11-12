import matplotlib.pyplot as plt

def display_sn_spec(ext_hdu, sn_segid):
    # Get extracted 1d spec 
    wav = ext_hdu[('SOURCE', sn_segid)].data['wavelength']
    flam = ext_hdu[('SOURCE', sn_segid)].data['flam'] * 1e-17
    ferr_lo = ext_hdu[('SOURCE', sn_segid)].data['flounc'] * 1e-17
    ferr_hi = ext_hdu[('SOURCE', sn_segid)].data['fhiunc'] * 1e-17

    # figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Wavelength [A]', fontsize=15)
    ax.set_ylabel('F-lambda [cgs]', fontsize=15)
    ax.plot(wav, flam, color='k', lw=1.5, label='pyLINEAR x1d spec')
    ax.fill_between(wav, flam - ferr_lo, flam + ferr_hi, color='gray')
    plt.show()

    return None