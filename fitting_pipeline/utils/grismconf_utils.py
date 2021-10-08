import numpy as np

def get_all_pq(m):

    allp = np.arange(m+1)
    allq = np.arange(m+1)

    pq = []

    for em in range(m+1):
        for p in allp:
            for q in allq:
                if p + q == em:
                    pq.append((p,q))

    return pq

def grismconf_polynomial_latex(n, m):
    """
    Paste the output of this function along with the
    following in a tex document and compile to see 
    what the polynomial looks like.

    \\documentclass[12pt]{article}
    \\usepackage{amsmath, amssymb}
    \\begin{document}
    \\begin{multline*}

    \\end{multline*}
    \\end{document}

    """
    
    counter_max = int((m+1)*(m+2)/2)

    # Now get all p, q unique combinations 
    # such that p+q belongs to the set {0,1,...,m}.
    # The func below returns a list of 2-tuples
    # where each tuple corresponds to a (p,q) 
    # combination satisfying the above condition.
    pq = get_all_pq(m)

    assert len(pq) == counter_max

    # outer loop with t
    for i in range(n+1):

        if i==1:
            print('t(')        

        if i > 1:
            print('t^{i}'.format(i=i), end='')
            print('(')

        # inner loop with x, y
        for c in range(counter_max):
            p = pq[c][0]
            q = pq[c][1]

            # --------- print coeff
            print('a_{', end='')
            print('{i}'.format(i=i), end='')
            print(',',  end='')
            print('{c}'.format(c=c),  end='')
            print('}', end=' ')

            # --------- formatting xy string
            if p==0 and q==0:
                xy_str = ''
            elif p==0 and q==1:
                xy_str = 'x'
            elif p==1 and q==0:
                xy_str = 'y'
            elif p==1 and q==1:
                xy_str = 'xy'
            elif p==0 and q>1:
                xy_str = 'x^{q}'.format(q=q)
            elif p>1 and q==0:
                xy_str = 'y^{p}'.format(p=p)
            elif p==1 and q>1:
                xy_str = 'x^{q}y'.format(q=q)
            elif p>1 and q==1:
                xy_str = 'xy^{p}'.format(p=p)
            else:
                xy_str = 'x^{q} y^{p}'.format(q=q, p=p)

            # --------- final printing
            if i == n and c == counter_max-1:
                print(xy_str)
            else:
                print(xy_str, '+')

        if i>=1:
            print(')')

    print('----- Move the + sign manually if n>=2. Also need to add newlines manually.\n')

    return None

def grismconf_polynomial(n, m, a, x0, y0, t):

    pnm = 0

    counter_max = int((m+1)*(m+2)/2)

    # Now get all p, q unique combinations 
    # such that p+q belongs to the set {0,1,...,m}.
    # The func below returns a list of 2-tuples
    # where each tuple corresponds to a (p,q) 
    # combination satisfying the above condition.
    pq = get_all_pq(m)

    assert len(pq) == counter_max

    # outer loop with t
    for i in range(n+1):

        inner_sum = 0

        # inner loop with x0,y0
        for j in range(counter_max):
            p = pq[j][0]
            q = pq[j][1]

            inner_sum += a[i, j] * x0**q * y0**p

        if i > 0:
            pnm += inner_sum * t**i
        elif i == 0:
            pnm = inner_sum

    return pnm

def check_with_grismconf():
    """
    See examples from Nor on the GRISMCONF page:
    https://github.com/npirzkal/GRISMCONF
    """
    import grismconf as gc
    from tqdm import tqdm

    g102_conf_path = '/Users/baj/Documents/pylinear_ref_files/pylinear_config/WFC3IR/g102.conf'

    C = gc.Config(g102_conf_path)

    # ------- Inputs for the closed form function above
    # I've simply copy pasted the required coeffs corresponding
    # to the function I want to test against.
    coeff_arr = np.array([[0.22124288639552142, -0.0005296596699611924, -0.00194267296229834, 
                           7.090784982811698e-8, 8.606706636991972e-7, 1.7723344437142464e-7], 
                          [2.926894451867832, -0.0012376481548631484, -0.0003952881061447967, 
                          -2.2206943788145118e-7, 2.6074632924170137e-6, -1.4638486314051208e-7]])
    cshp = coeff_arr.shape

    # SOlve for n and m
    n = cshp[0] - 1

    # solving for m using the quadratic formula
    b2_4ac = np.sqrt(9 + 8 * (cshp[1] - 1))
    mpos = int((-3 + b2_4ac)/2)
    mneg = int((-3 - b2_4ac)/2)
    if   mpos > 0: m = mpos
    elif mneg > 0: m = mneg
    else: 
        print('Incorrect value for shape.')

    # ---------------------------
    # Evaluate the polynomials using the GRISMCONF package
    # and also using hte above function. Must match for
    # any given x, y, t
    print('Now checking GRISMCONF polynomial eval against closed form eval...')
    for x in tqdm(np.arange(100.0, 1000.0, 5.0)):
        for y in np.arange(100.0, 1000.0, 5.0):

            for t in np.arange(0.0, 1.0, 0.1):

                # Evaluate using the GRISMCONF function
                x0 = x
                y0 = y
                d = C.DISPY('+1', x0, y0, t)

                # Evaluate using the closed form function written above
                pnm = grismconf_polynomial(n, m, coeff_arr, x0, y0, t)

                assert np.isclose(d, pnm)

    return None

def roman_prism_dispersion():
    """
    This function will solve for the DISPX and DISPL coefficients 
    that correctly match the strong wavelength dependence 
    of dispersion in the Roman prism by comparing to the 
    wavelength sampling in the spectra from Jeff Kruk.

    For the Roman prism conf I'm assuming all DISPY 
    coefficients to be zero.

    Typically, i.e., for HST WFC3/IR grisms,
    DISPL coeffs are set manually (see notes) since
    those coeffs are simply a (2,1) array usually so
    n=1,m=0 and therefore the polynomial is a straight
    line ---  lambda = a00 + t*a10.
    So we just need to set the coeffs such that the 
    wavelength range is returned back, i.e., at t=0
    lambda = lambda_min so a00 is lambda_min, and at t=1
    lambda = lambda_max so a10 is lambda_max - a00.

    However, this isn't the case for the Roman prism.

    # ------------ DISPX form:
    p10 = a00_x + a10_x * t 
    NOT possible to get a variable dispersion with this form 
    IF DISPL is also linear with t, which it isn't (see below).
    Although, DISPL is linear with t for HST grisms this CANNOT
    be the case for the Roman prism.

    For the Roman prism,
    DISPL cannot be linear in t because looking at the dispersion
    from Jeff Kruk's spectrum it looks like a third deg polynomial.
    The dispersion (dl/dx) is proportional to (dl/dt)/(dx/dt) and 
    therefore if dl/dx is a third deg polynomial then the most 
    straightforward way to arrive at this is to have lambda~t^4 
    and x~t.

    # ------------ DISPL form:
    Assuming DISPL form to be:
    lambda =   a00_lam 
             + a10_lam * t 
             + a20_lam * t^2 
             + a30_lam * t^3
             + a40_lam * t^4 
    
    and a00_lam is fixed at 7500.0
    a00_lam is the same regardless of what the rest of the expression
    because at t=0 i.e., start of spectrum, the wavelength should be
    lambda_min which is 7500.0 for the Roman prism.
    """

    genplot = True

    # Read in manually copy pasted parts from Jeff Kruk's file
    mag = 23.0
    datadir = '/Volumes/Joshi_external_HDD/Roman/sensitivity_files/'
    s = np.genfromtxt(datadir + 'abmag' + str(int(mag)) + '_prism_sens_kruk.txt', 
        dtype=None, names=['wav'], 
        usecols=(0), skip_header=3, encoding='ascii')
    
    wav = s['wav'] * 1e4  # convert microns to angstroms
    print('Number of x pix for whole wav arrray:', len(wav))
    wav_idx = np.where((wav >= 7500) & (wav <= 18000))[0]
    print('Number of x pix for 7500 <= wav <= 18000:', len(wav_idx))

    # ---------
    tarr = np.arange(0.0, 1.01, 0.002)

    lam_min = wav[0]
    a00_lam = lam_min
    lam_max = wav[-1]
    dlam = lam_max - lam_min
    print('Wav range:', lam_min, lam_max)

    # First fit a polynomial directly to the dispersion
    # We will assume that this is a good fit and then
    # solve for coeffs that can reproduce the dispersion
    # given by the numpy polynomial.
    disp = wav[1:] - wav[:-1]
    wavmean = (wav[1:] + wav[:-1])/2
    pp = np.polyfit(x=wavmean, y=disp, deg=2)
    pol = np.poly1d(pp)

    # Numpy polyfit returns polynomial coeffs with highest power first
    print(pol)
    pol_0 = pp[0]  # goes with highest power of x variable in fit
    pol_1 = pp[1]
    pol_2 = pp[2]

    print('Polynomial coeffs:')
    print(pol_0)
    print(pol_1)
    print(pol_2)
    print('--------------')

    A = pol_1
    B = pol_0

    # Now solve for coeffs that go in the CONF file
    a10_x = (lam_max - a00_lam) / (A + B/2) # (A + B/2 + C/3 + D/4)
    print('Copy-paste the following into the conf file:')
    print('-----')
    print('DISPX_+1_1', '{:.5f}'.format(a10_x))
    print('-----')
    
    a10_lam = A * a10_x
    a20_lam = B * a10_x / 2
    #a30_lam = C * a10_x / 3
    #a40_lam = D * a10_x / 4

    print('DISPL_+1_0', '{:.5f}'.format(a00_lam))
    print('DISPL_+1_1', '{:.5f}'.format(a10_lam))
    print('DISPL_+1_2', '{:.5f}'.format(a20_lam))
    #print('DISPL_+1_3', '{:.5f}'.format(a30_lam))
    #print('DISPL_+1_4', '{:.5f}'.format(a40_lam))
    print('-----')

    # ------ Now confirm that you get the correct dispersion back
    # by reading in the new updated Roman prism conf file through
    # GRISMCONF
    import grismconf as gc
    roman_prism_conf_path = '/Users/baj/Documents/pylinear_ref_files/pylinear_config/Roman/Roman_WFI_P127_grismconf.v1.0.conf'
    conf = gc.Config(roman_prism_conf_path)

    prism_disp = np.zeros(len(tarr))
    prism_disp_trans = np.zeros(len(tarr))
    new_wav = np.zeros(len(tarr))

    # Using GRISMCONF # From the docs
    for tcount, t in enumerate(tarr):

        # Compute dispersion
        prism_disp[tcount] = conf.DDISPL('+1',0,0,t)/conf.DDISPX('+1',0,0,t)

        # Also compute new wavelengths at t according to DISPL
        new_wav[tcount] = conf.DISPL('+1',0,0,t)

        # Transform again to wav space
        prism_disp_trans[tcount] = A + (B/dlam)*(new_wav[tcount] - lam_min)

        # Manually computing derivative
        manual_deriv = a10_lam + 2*a20_lam*t# + 3*a30_lam*t**2 + 4*a40_lam*t**3

        print(tcount, '{:.3f}'.format(t), ' ',
            '{:.3f}'.format(prism_disp[tcount]), ' ',
            '{:.3f}'.format(new_wav[tcount]))#, '      ',
        #    '{:.3f}'.format(conf.DDISPL('+1',0,0,t)), ' ',
        #    '{:.3f}'.format(conf.DDISPX('+1',0,0,t)), ' ',
        #    '{:.3f}'.format(manual_deriv))

    if genplot:

        import matplotlib.pyplot as plt
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('Wavelength [Angstoms]', fontsize=13)
        ax.set_ylabel('Dispersion [Angstoms per pixel]', fontsize=13)

        ax.plot(wavmean, disp, color='k', lw=1.5, label='Orig prism disp')
        ax.plot(wavmean, pol(wavmean), color='seagreen', lw=1.5, label='np polyfit to orig disp')
        ax.plot(new_wav, prism_disp, color='crimson', lw=2.0, label='GRISMCONF disp from new conf file')

        ax.legend(fontsize=12)

        plt.show()

    return None

if __name__ == '__main__':
    
    # ----- printing in latex
    #grismconf_polynomial_latex(3, 3)

    # ----- Quick evalution of polynomial 
    # for a given n, m, x0, y0, and t
    # User must also supply coeff array
    n = 1
    m = 2

    x0 = 1000.0
    y0 = 1000.0
    t = 0

    # From DISPX, DISPY, or DISPL
    coeff_arr = np.array([[2.78644810e+01, -1.05405727e-02, -9.67048003e-04,  6.93139575e-06,  4.04962658e-06, -6.87672404e-07], 
                          [2.08466649e+02,  8.53323515e-03, -1.29802612e-02, -4.92687932e-06, -5.11233288e-06,  8.59590592e-07]])

    # --------- END USER INPUTS

    # Check shape and t input
    col_size = int((m+1)*(m+2)/2)
    shp_errmsg = 'Wrong shape for coefficient array. Expected (' + str(n+1) + ',' + str(col_size) + ')' 
    cshp = coeff_arr.shape
    assert cshp == (n+1, col_size), shp_errmsg
    assert 0 <= t <= 1, 'Wrong value of t given.'

    # Evaluate polynomial
    pnm = grismconf_polynomial(n, m, coeff_arr, x0, y0, t)
    #print('Polynomial evaluation:', pnm)

    # Check closed form eval against grismconf eval
    #check_with_grismconf()

    # Now figure out the coefficients you need for Roman
    roman_prism_dispersion()

