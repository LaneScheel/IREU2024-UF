import numpy as np
import cleaned_annotated_optical_functions as of

#-------------------------------------------------------------------------------------------------
def DLCT(x1s, x2s=None, M_abcd=None, lam=1064e-9):
    '''
    Calculate the linear operator L for the DLCT

    lam    - wavelength
    M_abcd - component ABCD matrix
    x1s    - array of x values
    x2s    - will be the same as x1s unless you're doing something wrong
    L      - linear operator
    dx1    - delta x
    '''
    A,B,C,D = M_abcd.ravel()

    if x2s is None:
        x2s = x1s
    
    if np.isclose(B, 0):
        L = np.diag(np.exp(-1j*np.pi * C/lam * x1s**2))     
    else:
        dx1 = x1s[1] - x1s[0]
        x1g, x2g = np.meshgrid(x1s,x2s)

        arg = (A*x1g**2 - 2*np.outer(x1s,x2s) + D*x2g**2)/(B*lam)
        L = dx1*np.sqrt(1j/(B*lam)) * np.exp(-1j*np.pi*arg)
    return L
#-------------------------------------------------------------------------------------------------
def LCT_1D_cav_scan(D_rt, u_inc, r, phi=0):
    '''
    Perform a 1D 'cavity scan' utilizing the LCT.

    D-rt   - round trip distance
    u-inc  - array of incident field amplitudes
    r      - ?
    phi    - ?
    N      - length of the incident array
    g      - ?
    I      - identity matrix of size N
    u_circ - array of circulating field amplitudes
    '''
    N = len(u_inc)
    g = np.exp(-1j*phi/90*np.pi)
    I = np.eye(N)
    u_circ = np.linalg.solve(I-r*g*D_rt, u_inc)
    return u_circ
#-------------------------------------------------------------------------------------------------
def xft(x, axis=0, norm=True):
    '''
    A fast way to compute the centered DFT by using the FFT and the Fourier shift 
    theorem to center the FFT kernel.
    '''
    # parameters for expanding the Fourier shift mask for broadcasting
    # to work when taking 1D FFT of N-D arrays
    x_shape = np.shape(x)
    n_dim = len(x_shape)
    N = x_shape[axis]
    new_dims = np.arange(n_dim-1)
    new_dims[axis:] += 1

    # Fourier shift mask
    n = np.arange(N)
    a0 = np.exp(-1j*np.pi*(N-1)**2/2/N)
    S = np.exp(1j*np.pi*(N-1)*n/N)
    S = np.expand_dims(S, new_dims.tolist())

    # compute the XFT
    X = a0*S*np.fft.fft(S*x, axis=axis)
    if norm:
        X /= np.sqrt(N)
    return X
#-------------------------------------------------------------------------------------------------
def CM_kernel(xs, C, lam=1064e-9, diag=False):
    '''
    Generate a Chirp multiplication kernel for a mirror.
    This accounts for the wavefronts interaction with 
    the curve mirror surface.
    
    '''
    C = C/lam
    d = np.exp(-1j*np.pi * C * xs**2)
    if not diag:
        d = np.diag(d)
    return d
#-------------------------------------------------------------------------------------------------
def DLCT_lpl(xs, M_abcd, lam=1064e-9):
    '''
    A formulation of the CM-CC-CM LCT kernel using purely CM_kernels and DFT matrices
    by carefully applying the appropriate scaling factors that result from commuting out
    the DFT related scaling operators. 
    '''
    A,B,C,D = M_abcd.ravel()
    N = len(xs)
    dx = xs[1] - xs[0]
    F = xft(np.eye(N))
    iF = np.conj(F).T
    scale = -lam/(N*dx**2)

    if B == 0:
        out = CM_kernel(xs, C, lam=lam)
    else: 
        Q3 = CM_kernel(xs, (D-1)/B, lam=lam)
        Q2 = CM_kernel(xs, -B*scale**2, lam=lam)
        Q1 = CM_kernel(xs, (A-1)/B, lam=lam)
        out = Q3@iF@Q2@F@Q1
    return out
#-------------------------------------------------------------------------------------------------