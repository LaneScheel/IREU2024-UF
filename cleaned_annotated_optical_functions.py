import numpy as np
import math

#-------------------------------------------------------------------------------------------------
def get_waist(q, lam=1064e-9):
    '''
    Get waist size from q parameter.

    q   - beam parameter
    lam - wavelength
    zr  - Rayleigh Range
    w0  - beam waist
    '''
    zr = np.imag(q)
    w0 = np.sqrt(zr * lam / np.pi)
    return w0
#-------------------------------------------------------------------------------------------------
def get_width(q, lam=1064e-9):
    '''
    Get beam size from q parameter.

    q   - beam parameter
    lam - wavelength
    zr  - Rayleigh Range
    w0  - beam waist
    w   - beam width
    '''
    w0 = get_waist(q, lam=lam)
    zr = np.imag(q)
    w = w0 * np.abs(q)/zr
    return w
#-------------------------------------------------------------------------------------------------
def prop_beam_param(q1, M, n1=1, n2=1):
    '''
    Propagate a q parameter through an ABCD matrix

    q1  - input beam parameter
    q2  - output beam parameter
    M   - component ABCD metrix
    n1  - index of refraction 1
    n2  - index of refraciton 2
    '''
    A,B,C,D = M.ravel()
    q2 = n2*(A*q1/n1 + B)/(C*q1/n1 + D)
    return q2
#-------------------------------------------------------------------------------------------------
def calc_accum_gouy_n(q, M, n=0):
    '''
    1D accumulated Gouy phase of a mode of order n and beam parameter q through
    abcd matrix M.

    q   - input beam parameter
    M   - component ABCD matrix
    n   - modes wished to be calculated
    psi - accumulated general gouy phase 
    '''
    n = np.array(n)
    A,B,C,D = M.ravel()
    psi = (A+B/np.conj(q))/np.abs(A+B/q)
    return psi**(n+1/2)
#-------------------------------------------------------------------------------------------------
def calc_accum_gouy_nm(qx, Mx, qy=None, My=None, n=0, m=0):
    '''
    2D accumulated Gouy phase of a mode of order n, m and beam parameter qx, qy 
    through abcd matrix Mx, My.

    qx    - input beam parameter in x
    qy    - input beam parameter in y
    Mx    - component ABCD matrix in x
    My    - component ABCD matrix in y
    n     - n indice of modes wished to be calculated
    m     - m indice of modes wished to be calculated
    psi_x - accumulated general gouy phase in x 
    psi_y - accumulated general gouy phase in y
    '''
    n = np.array(n)
    m = np.array(m)
    if My is None:
        My = Mx
    if qy is None:
        qy = qx

    psi_x = calc_accum_gouy_n(qx, Mx, n=n)
    psi_y = calc_accum_gouy_n(qy, My, n=m)
    return psi_x * psi_y
#-------------------------------------------------------------------------------------------------
def set_space(d, n=1):
    '''
    ABCD matrix for free space of d meters

    d - distance or length of space
    n - index of refraction
    M - ABCD matrix for free space
    '''
    M = np.array([[1, d/n],[0, 1]])
    return M
#-------------------------------------------------------------------------------------------------
def set_lens(p):
    '''
    ABCD matrix for a lens with focal power of p diopters

    p - focal power in 1/m
    '''
    M = np.array([[1, 0],[-p, 1]])
    return M
#-------------------------------------------------------------------------------------------------
def set_mirror(R):
    '''
    ABCD matrix for a mirror with radius of curvature R

    R - radius of curvature
    '''
    return set_lens(2/R)
#-------------------------------------------------------------------------------------------------
def calc_q_eig(M):
    '''
    Computes both positive and negative solutions to the quadratic eigenmode equation
    -c*q**2 + (a-d)*q + b = 0

    M - component ABCD matrix
    '''
    A,B,C,D = M.flatten()
    root_term = np.lib.scimath.sqrt(4*B*C + (A-D)**2)
    q1 = ((A-D) + root_term)/(2*C)
    q2 = ((A-D) - root_term)/(2*C)
    if np.imag(q1) > 0:
        q = q1
    else:
        q = q2
    return q
#-------------------------------------------------------------------------------------------------
def get_1D_field_amp_n(x, q, n=0, lam=1064e-9):
    '''
    1D HG electric field amplitude taken from Siegmann eq 16.54.

    x   - array of x positions in meters
    q   - beam parameter (input or output depending on where you want to do the calculation)
    lam - wavelength in meters
    zr  - Rayleigh Range
    w0  - beam waist in meters
    w   - beam width in meters
    k   - wave number
    n   - n indice of modes to calculate for
    c   - hermite polynomial coefficients 
    t1  - normalization constant 1
    t2  - normalization constant 2
    t3  - normalization constant 3
    u   - computed field amplitudes
    E   - normalized field amplitudes
    '''

    def herm(n,x):
        c = np.zeros(n+1)
        c[-1] = 1
        return np.polynomial.hermite.hermval(x,c)

    w0 = get_waist(q, lam=lam)
    w = get_width(q, lam=lam)
    k = 2*np.pi/lam
    
    t1 = np.sqrt(np.sqrt(2/np.pi))
    t2 = np.sqrt(1.0/(2.0**n*math.factorial(n)*w0))
    t3 = np.sqrt(w0/w)
    norm = t1*t2*t3
    u = herm(n, np.divide.outer(np.sqrt(2)*x,w)) * np.exp(-1j*k*np.divide.outer(x**2,(2*q)))

    E = norm * u  
    return E
#-------------------------------------------------------------------------------------------------
def get_2D_field_amp_nm(x, y=None, qx=1j, qy=None, n=0, m=0, lam=1064e-9):
    '''
    2D HG electric field amplitude.

    x   - array of x positions
    y   - array of y positions
    qx  - beam parameter in x
    qy  - beam parameter in y
    ux  - calculated field amplitude in x
    uy  - calculated field amplitude in y
    '''
    if y is None:
        y = x
    if qy is None:
        qy = qx
    uy = get_1D_field_amp_n(y, qy, n, lam=lam)
    ux = get_1D_field_amp_n(x, qx, m, lam=lam)
    return np.outer(uy, ux)      
#-------------------------------------------------------------------------------------------------