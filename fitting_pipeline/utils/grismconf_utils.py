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
    coeff_arr = np.array([[0.22124288639552142, -0.0005296596699611924, -0.00194267296229834, 7.090784982811698e-8, 8.606706636991972e-7, 1.7723344437142464e-7], 
                          [2.926894451867832, -0.0012376481548631484, -0.0003952881061447967, -2.2206943788145118e-7, 2.6074632924170137e-6, -1.4638486314051208e-7]])
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
                dx = C.DISPY('+1', x0, y0, t)

                # Evaluate using the closed form function written above
                pnm = grismconf_polynomial(n, m, coeff_arr, x0, y0, t)

                assert np.isclose(dx, pnm)

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
    print('Polynomial evaluation:', pnm)

    check_with_grismconf()

