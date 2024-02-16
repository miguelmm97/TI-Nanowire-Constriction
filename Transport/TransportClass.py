from dataclasses import dataclass, field
from numpy import pi, ComplexWarning
from numpy.linalg import inv
from scipy.linalg import expm, ishermitian
import numpy as np
from numpy.fft import ifftshift, ifft, fftshift, ifft2, fft2
import time
import logging


# Constants
hbar = 1e-34                # Planck's constant in Js
nm = 1e-9                   # Conversion from nm to m
ams = 1e-10                 # Conversion from Å to m
e = 1.6e-19                 # Electron charge in C
phi0 = 2 * pi * hbar / e    # Quantum of flux

# Pauli matrices
sigma_0 = np.eye(2, dtype=np.complex128)
sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

# Geometry
def step(x1, x2, smoothing=None):

    if smoothing is None:
        return np.heaviside(x1 - x2, 1)
    else:
        return 0.5 + (1 / pi) * np.arctan(smoothing * (x1 - x2))

def geom_nc(x, x1, x2, r1, r2, sigma=None):
    return r1 + (r2 - r1) * step(x, x2, sigma) + ((r2 - r1) / (x2 - x1)) * (x - x1) * (step(x, x1, sigma) - step(x, x2, sigma))

def geom_cons(x, x1, x2, x3, r1, r2, sigma):
    return geom_nc(x, x1, x2, r1, r2, sigma) + geom_nc(-x + x2 + x3, x1, x2, r1, r2, sigma) - r2


# Transfer and scattering
def M_Ax(modes, dx, w, h, B_perp=0.):

    if w is None or h is None:
        return 0
    else:
        perimeter = 2 * (w + h)
        aspect_ratio = w / (w + h)
        C = - 1j * (nm ** 2 / hbar) * e * B_perp * perimeter / pi ** 2
        M = np.zeros(shape=(len(modes), len(modes)), dtype=np.complex128)

        if B_perp != 0:
            i = 0
            for n1 in modes:
                j = 0
                for n2 in modes:
                    if (n1 - n2) % 2 != 0:
                        m = n1 - n2
                        M[i, j] = C * ((-1) ** ((m + 1) / 2)) * np.sin(m * pi * aspect_ratio / 2) / m ** 2
                    j += 1
                i += 1

        return np.kron(sigma_0, M) * dx

def M_theta(modes, dx, R, dR, w=None, h=None, B_par=0.):
    C = 0.5 * (nm ** 2) * e * B_par / hbar
    A_theta = C * R ** 2 if (w is None or h is None) else C * (w * h / pi)  # A_theta = eBa²/2hbar
    M = (np.sqrt(1 + dR ** 2) / R) * np.diag(modes - 0.5 + A_theta)         # 1/R (n-1/2 + eBa²/2hbar) term

    return np.kron(sigma_x, M) * dx

def M_EV(modes, dx, dR, E, vf, Vnm=None):

    if Vnm is None:
        Vnm = np.zeros(len(modes))
    else:
        if Vnm.shape != (len(modes), len(modes)):
            raise AssertionError('Vnm must be the Fourier transform matrix of V')

    M = (1j / vf) * np.sqrt(1 + dR ** 2) * (E * np.eye(len(modes)) - Vnm)  # i ( E delta_nm + V_nm) / vf term
    return np.kron(sigma_z, M) * dx

def transfer_to_scattering(T, debug=False):

    n_modes = int(T.shape[0] / 2)
    if debug:
        if not np.allclose(T.T.conj() @ np.kron(sigma_z, np.eye(n_modes)) @ T, np.kron(sigma_z, np.eye(n_modes)), atol=1e-13):
            raise ValueError('Current flow not preserved by the transfer matrix!')

    # Divide transfer matrix
    inv_t = T[0: n_modes, 0: n_modes]
    inv_tp = T[n_modes:, n_modes:]
    inv_r = T[n_modes:, 0: n_modes]
    inv_rp = T[0: n_modes, n_modes:]

    if debug:
        if np.linalg.cond(inv_t) > 10 or np.linalg.cond(inv_tp) > 10:
            raise ValueError('Non-invertible matrix encountered in the transfer matrix')

    # Transform to scattering
    t = np.linalg.inv(inv_t).T.conj()
    tp = np.linalg.inv(inv_tp)
    r = - tp @ inv_r
    rp = inv_rp @ tp
    S = np.block([[r, tp], [t, rp]])

    if debug:
        if not np.allclose(S.T.conj() @ S, np.eye(len(S)), atol=1e-13):
            raise ValueError('Unitarity of the scattering matrix not preserved!')

    return S

def scattering_to_transfer(S, debug=False):

    n_modes = int(S.shape[0] / 2)
    if debug:
        if not np.allclose(S.T.conj() @ S, np.eye(len(S)), atol=1e-13):
            raise ValueError('Unitarity of the scattering matrix not preserved!')

    # Divide scattering matrix
    r = S[0: n_modes, 0: n_modes]
    rp = S[n_modes:, n_modes:]
    t = S[n_modes:, 0: n_modes]
    tp = S[0: n_modes, n_modes:]

    if debug:
        print('Condition number for t: ', np.linalg.cond(t), ', det(t):', np.linalg.det(t))
        print('Condition number for tp: ', np.linalg.cond(tp), ', det(tp):', np.linalg.det(tp))
        if np.linalg.cond(t) > 10 or np.linalg.cond(tp) > 10:
            raise ValueError('Non-invertible matrix encountered in the scattering matrix')

    # Transform to transfer matrix
    inv_tp = np.linalg.inv(tp)
    T_00 = np.linalg.inv(t.T.conj())
    T_01 = rp @ inv_tp
    T_10 = - inv_tp @ r
    T_11 = inv_tp
    T = np.block([[T_00, T_01], [T_10, T_11]])

    if debug:
        if not np.allclose(T.T.conj() @ np.kron(sigma_z, np.eye(n_modes)) @ T, np.kron(sigma_z, np.eye(n_modes)), atol=1e-8):
            raise ValueError('Current flow not preserved by the transfer matrix!')

    return T

def scat_product(s1, s2, debug=False):

    if debug:
        if s1.shape != s2.shape:
            raise ValueError(" Different size for scattering matrices")
        if not np.allclose(s1.T.conj() @ s1, np.eye(len(s1)), atol=1e-8):
            raise ValueError('Unitarity of the scattering matrix not preserved!')
        if not np.allclose(s2.T.conj() @ s2, np.eye(len(s2)), atol=1e-8):
            raise ValueError('Unitarity of the scattering matrix not preserved!')

    # Divide scattering matrix
    n_modes = int(s1.shape[0] / 2)
    r1, r2 = s1[0: n_modes, 0: n_modes], s2[0: n_modes, 0: n_modes]
    r1p, r2p = s1[n_modes:, n_modes:], s2[n_modes:, n_modes:]
    t1, t2 = s1[n_modes:, 0: n_modes], s2[n_modes:, 0: n_modes]
    t1p, t2p = s1[0: n_modes, n_modes:], s2[0: n_modes, n_modes:]

    Id = np.eye(n_modes)
    if debug:
        if np.linalg.cond(Id - r1p @ r2) > 10 or np.linalg.cond(Id - r2 @ r1p) > 10:
            raise ValueError('Non-invertible matrix encountered in the product of scattering matrices')

    # Product of S1 S2
    inv_r1pr2 = inv(Id - r1p @ r2)
    r = r1 + t1p @ r2 @ inv_r1pr2 @ t1
    rp = r2p + t2 @ inv_r1pr2 @ r1p @ t2p
    t = t2 @ inv_r1pr2 @ t1
    tp = t1p @ inv(Id - r2 @ r1p) @ t2p
    scat_matrix = np.block([[r, tp], [t, rp]])

    if debug:
        if not np.allclose(scat_matrix.T.conj() @ scat_matrix, np.eye(len(scat_matrix)), atol=1e-8):
            raise ValueError('Unitarity of the scattering matrix not preserved!')

    return scat_matrix

def transport_mode(x, theta, r, n, E, vf, spin='up', lead=True):

    k = (E / vf) ** 2 - (1 / r ** 2) * (n - 0.5) ** 2
    norm  = 1  / np.sqrt(2 * pi * r)
    transverse_part = np.exp(1j * ((n - 0.5) * theta))

    if spin == 'up':
        longitudinal_part = np.exp(1j * k * x)
    elif spin == 'down':
        longitudinal_part = np.exp(- 1j * k * x)
    else:
        raise ValueError('Unspecified spin')

    if lead:
        return norm * transverse_part * longitudinal_part
    else:
        return norm * transverse_part


# Gaussian correlated potential
def gaussian_correlated_potential_1D_FFT(L, Nx, strength, xi, vf):
    """
    Generates a sample of a gaussian correlated potential V(x) with strength,
    a certain correlation length and Nx points in momentum space. It uses the
    FFT algorithm from numpy.

    Params:
    ------
    L:                            {np.float}  Physical length of the system
    Nx:                             {np.int}  NUmber of Fourier modes (it must be odd)
    strength:                     {np.float}  Strength of the potential in units of (hbar vf /corr_length)^2
    xi:                           {np.float}  Correlation length of the distribution in nm
    vf:                           {np.float}  Fermi velocity in nm meV

    Returns:
    -------
    Vgauss:                {np.array(float)}  Gaussian correlated potential sample
    """

    if Nx % 2 == 0: raise ValueError('Nx must be odd')

    # Definitions for the transform
    dx = L / Nx
    fmax = 2 * pi / dx;
    df = 2 * pi / L
    f = np.linspace(0, fmax / 2, num=int(Nx / 2) + 1, endpoint=False)

    # Correlations and fourier modes
    scale = strength * (vf ** 2) / xi
    Vn = np.abs(np.random.normal(scale=np.sqrt(scale) * np.exp(- 0.25 * (xi * f) ** 2)))
    V = np.concatenate((Vn, Vn[1:][::-1]))
    phi = np.random.uniform(0, 2 * pi, size=int((Nx - 1) / 2))
    phases = np.concatenate(([0], phi, -phi[::-1]))
    FT_V = V * np.exp(1j * phases)

    # Convert the product of the two functions back to real space
    Vgauss = np.sqrt(2 * pi) * ifft(FT_V) / dx / np.sqrt(df)

    return Vgauss

def gaussian_correlated_potential_2D_FFT(L, r, Nx, Ny, strength, xi, vf):
    """
    Generates a sample of a gaussian correlated potential V(x) with strength,
    a certain correlation length and Nx points in momentum space. It uses the
    FFT algorithm from numpy.

    Params:
    ------
    L:                            {np.float}  Physical length of the system
    r:                            {np.float}  Radius of the wire
    Nx:                             {np.int}  Number of Fourier modes (it must be odd)
    Ntheta:                         {np.int}  Number of Fourier modes (it must be odd)
    strength:                     {np.float}  Strength of the potential in units of (hbar vf /corr_length)^2
    correlation_length:           {np.float}  Correlation length of the distribution in nm
    vf:                           {np.float}  Fermi velocity in nm meV

    Returns:
    -------
    Vgauss:                {np.array(float)}  Gaussian correlated potential sample
    """

    if (Nx % 2 == 0) or (Ny % 2 == 0): raise ValueError('Nx and Ny must be odd')

    # Definitions for the transform
    dx = L / Nx;
    dy = 2 * pi * r / Ny
    nx = int(Nx / 2);
    ny = int(Ny / 2)
    fmax_x = 2 * pi / dx;
    df_x = 2 * pi / L
    fmax_y = 2 * pi / dy;
    df_y = 1 / r
    fx = np.linspace(0, fmax_x / 2, num=nx + 1, endpoint=False)
    fy = np.linspace(0, fmax_y / 2, num=ny + 1, endpoint=False)
    FX, FY = np.meshgrid(fx, fy)

    # Correlations and fourier modes
    scale = strength * (vf ** 2)
    V = np.zeros((Ny, Nx))
    V_x0 = np.random.normal(scale=np.sqrt(scale) * np.exp(- 0.25 * (xi ** 2) * (FX[0, 1:] ** 2 + FY[0, 1:] ** 2)))
    V_0y = np.random.normal(scale=np.sqrt(scale) * np.exp(- 0.25 * (xi ** 2) * (FX[1:, 0] ** 2 + FY[1:, 0] ** 2)))
    V[0, 1:] = np.concatenate((V_x0, V_x0[::-1]))
    V[1:, 0] = np.concatenate((V_0y, V_0y[::-1]))
    V[0, 0] = np.random.normal(scale=np.sqrt(scale) * np.exp(- 0.25 * (xi ** 2) * (FX[0, 0] ** 2 + FY[0, 0] ** 2)))
    V[1: ny + 1, 1: nx + 1] = np.random.normal(
        scale=np.sqrt(scale) * np.exp(- 0.25 * (xi ** 2) * (FX[1:, 1:] ** 2 + FY[1:, 1:] ** 2)))
    V[ny + 1:, 1: nx + 1] = np.random.normal(
        scale=np.sqrt(scale) * np.exp(- 0.25 * (xi ** 2) * (FX[1:, 1:][::-1, :] ** 2 + FY[1:, 1:][::-1, :] ** 2)))
    V[1: ny + 1, nx + 1:] = V[ny + 1:, 1: nx + 1][::-1, ::-1]
    V[ny + 1:, nx + 1:] = V[1: ny + 1, 1: nx + 1][::-1, ::-1]

    phases = np.zeros((Ny, Nx))
    phi_x0 = np.random.uniform(0, 2 * pi, size=int(Nx / 2))
    phi_0y = np.random.uniform(0, 2 * pi, size=int(Ny / 2))
    phases[0, 0] = 0
    phases[1: ny + 1, 1: nx + 1] = np.random.uniform(0, 2 * pi, size=(ny, nx))
    phases[ny + 1:, 1: nx + 1] = np.random.uniform(0, 2 * pi, size=(ny, nx))
    phases[1: ny + 1, nx + 1:] = - phases[ny + 1:, 1: nx + 1][::-1, ::-1]
    phases[ny + 1:, nx + 1:] = - phases[1: ny + 1, 1: nx + 1][::-1, ::-1]
    phases[0, 1:] = np.concatenate((phi_x0, -phi_x0[::-1]))
    phases[1:, 0] = np.concatenate((phi_0y, -phi_0y[::-1]))
    FT_V = np.abs(V) * np.exp(1j * phases)

    # Convert the product of the two functions back to real space
    Vgauss = (2 * np.pi) * ifft2(FT_V) / (dx * dy) / (np.sqrt(df_x) * np.sqrt(df_y))
    V_iFFTx = np.sqrt(2 * np.pi) * ifft(FT_V, axis=1) / (dx) / np.sqrt(df_x)

    # return V_iFFTx * (2 * pi * np.sqrt(r) / Ny), Vgauss
    return V_iFFTx * (1 / np.sqrt(2 * pi * r)), Vgauss


# Code work
def get_fileID(file_list):
    expID = 0
    for file in file_list:
        if file.startswith('Experiment') and file.endswith('.h5'):
            stringID = file.split('Experiment')[1].split('.h5')[0]
            ID = int(stringID)
            expID = max(ID, expID)
    return expID + 1

def code_testing(L, Nx, Ntheta, dis_strength, corr_length, vf, check=0):

    # Simple check: No potential in 1d
    if check == 0:
        V_fft = np.zeros((Nx,))
        V_real = V_fft

    # Simple check: Constant potential in 1d
    if check == 1:
        V_fft = 10 * np.ones((Nx,))
        V_real = V_fft

    # Simple check: Constant potential in 2d
    elif check == 2:
        V_real = 10 * np.ones((Ntheta, Nx))
        V_1 = fft2(V_real) * np.sqrt(L * 2 * pi * 20) / (Nx * Ntheta)
        V_fft = ifft(V_1, axis=1) * (Nx / np.sqrt(L)) * (1 / np.sqrt(2 * pi * 20))


    # Simple check: V periodic function of theta
    elif check == 3:
        V_real = np.zeros((Ntheta, Nx))
        theta_sample = np.linspace(0, 2 * pi, Ntheta, endpoint=False)
        for i in range(Nx): V_real[:, i] = np.sin(theta_sample)
        V_1 = fft2(V_real) * np.sqrt(L * 2 * pi * 20) / (Nx * Ntheta)
        V_fft = ifft(V_1, axis=1) * (Nx / np.sqrt(L)) * (1 / np.sqrt(2 * pi * 20))


    # Simple check: Comparison 1d vs 2d with the same gaussian correlated potential
    elif check == 4:
        V_real = gaussian_correlated_potential_1D_FFT(L, Nx, dis_strength, corr_length, vf);
        V_fft = V_real
        V_real2 = np.zeros((Ntheta, Nx))
        for i in range(Nx): V_real2[:, i] = np.ones((Ntheta,)) * V_real[i]
        V_1 = fft2(V_real2) * np.sqrt(L * 2 * pi * 20) / (Nx * Ntheta)
        V_fft = ifft(V_1, axis=1) * (Nx / np.sqrt(L)) * (1 / np.sqrt(2 * pi * 20))

    return V_fft, V_real

def check_imaginary(array):
    for x in np.nditer(array):
        if not np.imag(x) < 1e-15:
            raise ValueError('Imaginary part is not negligible!')


@dataclass
class transport:
    """ Transport calculations on 3dTI nanostructures based on their effective surface theory."""

    L:          float       # Total length of the nanowire in nm
    rad:        float       # Radius of the nanowire (only meaningful if constant radius)
    vf:         float       # Fermi velocity in meV nm
    B_perp:     float       # Magnetic field perpendicular to the axis of the nanostructure
    B_par:      float       # Magnetic field parallel to the axis of the nanostructure
    l_cutoff:   int         # Cutoff in the number of angular momentum modes
    geometry    = {}
    n_regions   = 0
    S_tot       = None
    Nmodes:     int = field(init=False)
    modes:      np.ndarray = field(init=False)

    def __post_init__(self):
        self.modes = np.arange(-self.l_cutoff, self.l_cutoff + 1)
        self.Nmodes = len(self.modes)

    # Methods for creating the geometry of the transport region
    def add_nw(self, x0, xf, Vnm=None, n_points=None, r=None, w=None, h=None):

        if self.n_regions != 0 and x0 != self.geometry[self.n_regions - 1]['xf']:
            raise ValueError('Regions dont match')
        if r is None and (w is None or h is None):
            raise ValueError('Need to specify r or (w, h)')

        self.geometry[self.n_regions] = {
            'region_type': 'nw',
            'x0': x0,
            'xf': xf,
            'n_points': n_points,
            'V': Vnm,
            'w': w,
            'h': h,
            'r': (w + h) / pi if r is None else r,
            'dx': 100 if n_points is None else abs(xf - x0) / n_points,
        }

        self.n_regions += 1

    def add_nc(self, x0, xf, n_points, Vnm=None, sigma=None, r1=None, r2=None, w1=None, w2=None, h1=None, h2=None):

        if self.n_regions != 0 and x0 != self.geometry[self.n_regions - 1]['xf']:
            raise ValueError('Regions dont match')
        if r1 is None and (w1 is None or h1 is None):
            raise ValueError('Need to specify r or (w, h)')
        if r2 is None and (w2 is None or h2 is None):
            raise ValueError('Need to specify r or (w, h)')


        r1 = (w1 + h1) / pi if r1 is None else r1
        r2 = (w2 + h2) / pi if r2 is None else r2
        x = np.linspace(x0, xf, n_points)
        self.geometry[self.n_regions] = {
            'region_type': 'nw',
            'x0': x0,
            'xf': xf,
            'dx': abs(xf - x0) / n_points,
            'n_points': n_points,
            'V': Vnm,
            'r': geom_nc(x, x0, xf, r1, r2, sigma),
            'w': None if w1 is None or w2 is None else geom_nc(x, x0, xf, w1, w2, sigma),
            'h': None if h1 is None or h2 is None else geom_nc(x, x0, xf, h1, h2, sigma)
        }

        self.n_regions += 1

    def build_geometry(self, r_vec, x_vec, V_vec, n_vec=None, sigma_vec=None):
        """
        Builds the geometry of the nanostructure from specifying the radius, length and potential distribution of the
        different sections. Once this distribution has been given, it creates a nanocone for consecutive points x, x+1
        differing in radius, and a nanowire if they have the same radius.

        The idea is that the vectors r, x, and V give a discretisation of radii and potentials, so that they act as a
        discretisation in which between every two points the potential is constant and the radius changes infinitesimally
        With this we can build any geometry.

        Furthermore, the Npoints vector allows us to perform the transport calculation by further discretising that region
        when calculating the transfer matrix.

        OBS!: The potential V can be both rotationally symmetric or non-rotationally symmetric. In the former case we
        just have to include a discretised vector, each entry corresponding to V(x). In the latter, for each x entry we
        have to include a vector along the radial direction. The entries of this vector are the FFT of V(theta, x), for
        fixed x. We do it in this way because then we need to do the FT of V(theta) to get Vnm in the transfer matrix.

        Params:
        ------

        r:         {np.array(floats)}   Discretisation of the radius of the nanostructure as a function o x
        x:         {np.array(floats)}   Discretisation of x
        V:         {np.array(floats)}   Discretisation of the potential as a function of x.
        n_vec:     {np.array(floats)}   Number of points in the transport calculation for each individual region
        sigma_vec: {np.array(floats)}   Smoothing factor for each individual region


        Return:
        -------
        None, but updates self. geometry

        """

        if sigma_vec is None: sigma_vec = np.repeat(None, x_vec.shape[0])
        if n_vec is None: n_vec = np.repeat(None, x_vec.shape[0])

        if len(V_vec.shape) == 1:
            for i, (r, x, V, n, sigma) in enumerate(
                    zip(r_vec[:-1], x_vec[:-1], V_vec[:-1], n_vec[:-1], sigma_vec[:-1])):
                if r != r_vec[i + 1]:
                    self.add_nc(x, x_vec[i + 1], n, Vnm=self.get_potential_matrix(V), sigma=sigma, r1=r, r2=r[i + 1])
                else:
                    self.add_nw(x, x_vec[i + 1], Vnm=self.get_potential_matrix(V), n_points=n, r=r)

        elif len(V_vec.shape) == 2:
            for i, (r, x, V, n, sigma) in enumerate(
                    zip(r_vec[:-1], x_vec[:-1], V_vec[:, :-1].T, n_vec[:-1], sigma_vec[:-1])):
                if r != r_vec[i + 1]:
                    self.add_nc(x, x_vec[i + 1], n, Vnm=self.get_potential_matrix(V), sigma=sigma, r1=r, r2=r[i + 1])
                else:
                    self.add_nw(x, x_vec[i + 1], Vnm=self.get_potential_matrix(V), n_points=n, r=r)

    # Methods for calculating transport-related quantities
    def get_transfer_matrix(self, E, x0, xf, dx, V, r, w, h, T=None, backwards=False, debug=False, **kwargs):

        Id  = np.eye(self.Nmodes)
        M = M_EV(self.modes, dx, 0, E, self.vf, V)
        M += M_theta(self.modes, dx, r, 0, w, h, B_par=self.B_par)
        M += M_Ax(self.modes, dx, w, h, B_perp=self.B_perp)
        if backwards:
            T = expm(M * (xf - x0) / dx) if T is None else expm(M * (xf - x0) / dx) @ T
            # T = np.kron(sigma_x, Id) @ expm(M * (x0 - xf) / dx) if T is None else np.kron(sigma_x, Id) @ expm(M * (xf - x0) / dx) @ T
        else:
            T = expm(M * (xf - x0) / dx) if T is None else expm(M * (xf - x0) / dx) @ T

        if debug:
            n_modes = int(T.shape[0] / 2)
            if not np.allclose(T.T.conj() @ np.kron(sigma_z, np.eye(n_modes)) @ T, np.kron(sigma_z, np.eye(n_modes)), atol=1e-13):
                raise ValueError('Current flow not preserved by the transfer matrix!')

        return T

    def get_scattering_matrix(self, E, region_type, x0, xf, dx, n_points, V, r, w, h, S=None, backwards=False, debug=False):

        # For nanowires
        if region_type == 'nw':

            # No need for discretisation
            if n_points is None:
                T = self.get_transfer_matrix(E, x0, xf, dx, V, r, w, h, backwards=backwards)
                if S is None:
                    S = transfer_to_scattering(T)
                elif backwards:
                    S = scat_product(transfer_to_scattering(T), S)
                else:
                    S = scat_product(S, transfer_to_scattering(T))

                if debug:
                    if not np.allclose(T, scattering_to_transfer(S)):
                        raise ValueError('Scattering to transfer not working!')
                    if np.isnan(S).any():
                        raise OverflowError('Need for discretisation, overflow in S!')

            else:
                raise NotImplementedError('Part of the code for discretising nanowires needs to be implemented')
        else:
            raise NotImplementedError('Part of the code for nanocodes needs to be implemented')

        return S

    def get_Landauer_conductance(self, E, save_S=False, debug=False):

        S = None
        for i in range(0, self.n_regions):
            S = self.get_scattering_matrix(E, **self.geometry[i], S=S)

        if debug:
            if np.abs(E) < 1 and self.L < 150:
                t_analytic = np.diag(1 / np.cosh(self.L * (self.modes - 0.5) / self.rad))
                r_analytic = np.diag(- np.sinh(self.L * (self.modes - 0.5) / self.rad) / np.cosh(self.L * (self.modes - 0.5) / self.rad))
                if not np.allclose(S[0: self.Nmodes, 0: self.Nmodes], r_analytic):
                    raise ValueError('Analytic check failed for reflection matrix!')
                if not np.allclose(S[self.Nmodes:, 0: self.Nmodes], t_analytic):
                    raise ValueError('Analytic check failed for transmission matrix!')

        if save_S: self.S_tot = S
        t = S[self.Nmodes:, 0: self.Nmodes]
        G = np.trace(t.T.conj() @ t)

        if debug:
            check_imaginary(G)

        return np.real(G)

    def get_transmission_eigenvalues(self, E, get_max=False, debug=False):

        # Full scattering matrix
        S = None
        for i in range(0, self.n_regions):
            S = self.get_scattering_matrix(E, **self.geometry[i], S=S)

        # Transmission eigenvalues
        t = S[self.Nmodes:, 0: self.Nmodes]
        tt = t.T.conj() @ t
        eigval_tt, eigvec_tt = np.linalg.eigh(tt)

        if debug:
            if not ishermitian(tt, atol=1e-15):
                raise ValueError('Transmission matrix not hermitian!')
            if not np.allclose(eigval_tt, np.linalg.eig(t @ t.T.conj())[0]):
                raise ValueError('Transmission eigenvalues different for t^\dagger t and t t^\dagger!')
            if not np.allclose(np.sum(eigval_tt), np.linal.trace(tt)):
                raise ValueError('Transmission eigenvalues do not amount for the conductance at this energy!')
            if np.abs(E) < 1:
                tt_analytic = (1 / np.cosh(self.L * (self.modes - 0.5) / self.rad)) ** 2
                if not np.allclose(tt_analytic, eigval_tt):
                    raise ValueError('Failed analytical test for transmission eigenvalues.')

            # Compare eigenvalues and vectors of t and T
            eigval_t, eigvec_t = np.linalg.eig(t)
            print('T is normal: ', np.allclose(tt, t @ t.T.conj()))
            for i in range(len(eigval_t)):
                print('Eigenvalue: ', i, ', Scalar product between v_t and V_T: ', np.dot(eigvec_t[:, i], eigvec_tt[:, i]))
            for i in range(len(eigval_t)):
                print('Eigenvalue t ^2: ', np.sort(eigval_t * eigval_t.conj())[i], ', Eigenvalue T: ', np.sort(eigval_tt)[i])

        # Return
        if get_max:
            index = np.where(eigval_tt == np.max(eigval_tt))[0][0]
            return eigval_tt, eigvec_tt, np.max(eigval_tt), eigvec_tt[:, index]
        else:
            return eigval_tt, eigvec_tt

    def get_transmitted_state(self, E, state=0, debug=False):

        # Full scattering matrix
        S = None
        for i in range(0, self.n_regions):
            S = self.get_scattering_matrix(E, **self.geometry[i], S=S)

        # Transmission matrix SVD
        t = S[self.Nmodes:, 0: self.Nmodes]
        u_t, sing_val_t, v_t = np.linalg.svd(t)

        # Ordering the singular values squared from max to min
        idx = np.flip((sing_val_t ** 2).argsort())
        sing_val_t = sing_val_t[idx]
        u_t = u_t[:, idx]
        v_t_dagger = v_t.T.conj()[:, idx]

        # Getting the particular state associated
        phi_outR = sing_val_t[state] * u_t[:, state]
        phi_inL = v_t_dagger[:, state]

        if debug:
            tt = t.T.conj() @ t
            u_tt, sing_val_tt, v_tt = np.linalg.svd(tt)
            tt_eigval = np.linalg.eigvalsh(tt)
            if not np.allclose(np.sort(sing_val_t ** 2), np.sort(sing_val_tt)):
                raise ValueError('Singular values of t^\dagger t do not coincide with singular values of t squared!')
            if not np.allclose(np.sort(sing_val_tt), np.sort(tt_eigval)):
                raise ValueError('Singular values of t^\dagger t do not coincide with eigenvalues!')

        return phi_inL, phi_outR

    def get_scattering_states_back_forth_method(self, E, theta_vec, initial_state=0, debug=False):

        #  Preallocation
        psi_scatt_up = np.zeros((len(theta_vec), self.n_regions), dtype=np.complex128)
        psi_scatt_down = np.zeros((len(theta_vec), self.n_regions), dtype=np.complex128)
        S_forward_storage = np.zeros((2 * self.Nmodes, 2 * self.Nmodes, self.n_regions), dtype=np.complex128)
        S_backwards_storage = np.zeros((2 * self.Nmodes, 2 * self.Nmodes, self.n_regions), dtype=np.complex128)

        # Full scattering matrix
        print('------Calculating forward and backwards scattering matrices...')
        S1, S2 = None, None
        for i in range(0, self.n_regions):
            S1 = self.get_scattering_matrix(E, **self.geometry[i], S=S1)
            S2 = self.get_scattering_matrix(E, **self.geometry[self.n_regions - 1 - i], S=S2, backwards=True)
            S_forward_storage[:, :, i] = S1
            S_backwards_storage[:, :, i] = S2

        if debug:
            if np.abs(E) < 1 and self.L < 150:
                t_analytic = np.diag(1 / np.cosh(self.L * (self.modes - 0.5) / self.rad))
                r_analytic = np.diag(- np.sinh(self.L * (self.modes - 0.5) / self.rad) / np.cosh(
                    self.L * (self.modes - 0.5) / self.rad))
                if not np.allclose(S1[0: self.Nmodes, 0: self.Nmodes], r_analytic):
                    raise ValueError('Analytic check failed for reflection matrix 1!')
                if not np.allclose(S1[self.Nmodes:, 0: self.Nmodes], t_analytic):
                    raise ValueError('Analytic check failed for transmission matrix 1!')
                if not np.allclose(S2[0: self.Nmodes, 0: self.Nmodes], r_analytic):
                    raise ValueError('Analytic check failed for reflection matrix 2!')
                if not np.allclose(S2[self.Nmodes:, 0: self.Nmodes], t_analytic):
                    raise ValueError('Analytic check failed for transmission matrix 2!')
            if not np.allclose(S_forward_storage[:, :, -1], S_backwards_storage[:, :, -1]):
                raise ValueError('Forward and backwards scattering matrices do not coincide!')

        # State at the leads
        phi_inL_lead, phi_outR_lead = self.get_transmitted_state(E, state=initial_state, debug=True)

        print('------Calculating distribution of scattering states...')
        n_modes = len(phi_inL_lead)
        for i in range(0, self.n_regions):
            start_iter = time.time()

            # Scattering forward and backwards
            t1 = S_forward_storage[:, :, i][n_modes:, 0: n_modes]
            rp1 = S_forward_storage[:, :, i][n_modes:, n_modes:]
            r2 = S_backwards_storage[:, :, self.n_regions - 1 - i][0: n_modes, 0: n_modes]

            # Scattering states at position x
            phi_x_rightmover = (np.linalg.inv(np.eye(n_modes) - rp1 @ r2) @ t1) @ phi_inL_lead
            phi_x_leftmover = r2 @ phi_x_rightmover
            phi = np.concatenate((phi_x_rightmover, phi_x_leftmover))

            print('------------Region: {}/{}, x:{:.2f} nm, iter time: {:.3f} s'.format(i, self.n_regions, self.geometry[i]['x0'], time.time() - start_iter))
            if debug:
                print('---------------------------------------')
                print('Iter time: {:.2e}'.format(time.time() - start_iter))
                print('Condition number for 1 - rp1r2: ', np.linalg.cond(np.eye(n_modes) - rp1 @ r2), ', det(1-rp1r2):', np.abs(np.linalg.det(np.eye(n_modes) - rp1 @ r2)))
                print('det(inv(1-rp1r2)): ', np.abs(np.linalg.det(np.linalg.inv(np.eye(n_modes) - rp1 @ r2))))
                print('det(inv(1-rp1r2)t1): ', np.abs(np.linalg.det(np.linalg.inv(np.eye(n_modes) - rp1 @ r2) @ t1)))
                print('max eigenvalue for inv(1-rp1r2)t1: ', np.max(np.abs(np.linalg.eig(np.linalg.inv(np.eye(n_modes) - rp1 @ r2) @ t1)[0])))
                print('max value phi_x+: ', np.max(np.abs(phi_x_rightmover)))
                print('max value phi_x-: ', np.max(np.abs(phi_x_leftmover)))
                print('---------------------------------------')

            # Scattering states for the slab
            for j, theta in enumerate(theta_vec):
                trans_mode = transport_mode(self.geometry[i]['x0'], theta, self.geometry[i]['r'], self.modes, E, self.vf, lead=False)
                psi_scatt_up[j, i] = np.dot(trans_mode, phi[: self.Nmodes])
                psi_scatt_down[j, i] = np.dot(trans_mode, phi[self.Nmodes:])

        if debug:
            phi1 = phi_outR_lead / np.linalg.norm(phi_outR_lead)
            phi2 = phi[:self.Nmodes].conj() / np.linalg.norm(phi[:self.Nmodes])
            if np.abs(np.dot(phi1, phi2)) < 0.9:
                raise ValueError('Transferred and transmitted states do not coincide! Overlap:', np.abs(np.dot(phi1, phi2)))

        norm_scatt = np.sqrt(np.sum(psi_scatt_up * psi_scatt_up.conj() + psi_scatt_down * psi_scatt_down.conj()))
        psi_scatt_up, psi_scatt_down = psi_scatt_up / norm_scatt, psi_scatt_down / norm_scatt
        return psi_scatt_up, psi_scatt_down

    def get_bands_nw(self, region, k_range):
        """
        Calculates the spectrum of the region indicated. It must be a nanowire, and it assumes translation invariance.

        Params:
        ------
        region:       {int} Number of the region of the geometry in which we want to calculate the bands
        k_range: {np.array} Range of momenta within which we calculate the bands

        Returns:
        -------
        E: {np.array(len(2Nmodes, len(k)} Energy bands
        V:        {np.array(len(2Nmodes)} Bottom of the bands i.e. centrifugal potential
        """

        # Geometry
        if self.geometry[region]['type'] != 'nw': raise ValueError('Can only calculate spectrum in a nanowire!')
        w = self.geometry[region]['w']  # Width
        h = self.geometry[region]['h']  # Height
        r = self.geometry[region]['r']  # Radius
        P = 2 * (w + h) if r is None else 2 * pi * r  # Perimeter

        # Parallel gauge field: hbar vf 2pi/P (n-1/2 + A) * sigma_y
        A_theta = 0
        if self.B_par != 0:
            Ctheta = 0.5 * (nm ** 2) * e * self.B_par / hbar
            A_theta = Ctheta * r ** 2 if (w is None or h is None) else Ctheta * (w * h) / pi
        Mtheta = np.diag((2 * pi / P) * (self.modes - (1 / 2) + A_theta))
        Hxtheta = self.vf * np.kron(Mtheta, sigma_y)

        # Perpendicular gauge field: e vf < n | A_x | m > * sigma_x
        if self.B_perp != 0:
            r_aspect = w / (w + h)
            Cx = (nm ** 2 / hbar) * e * self.B_perp * P / pi ** 2
            Ax = np.zeros((self.Nmodes, self.Nmodes), dtype=np.complex128)
            i = 0
            for n1 in self.modes:
                j = 0
                for n2 in self.modes:
                    if (n1 - n2) % 2 != 0:
                        m = n1 - n2
                        Ax[i, j] = Cx * ((-1) ** ((m + 1) / 2)) * np.sin(m * pi * r_aspect / 2) / m ** 2
                    j += 1
                i += 1
            Hxtheta += self.vf * np.kron(Ax, sigma_x)

        # Hamiltonian and energy bands
        i = 0
        E = np.zeros((2 * self.Nmodes, len(k_range)))
        for k in k_range:
            Mk = (self.vf * k).repeat(self.Nmodes)  # hbar vf k
            Hk = np.kron(np.diag(Mk), sigma_x)  # hbar vf k * sigma_x
            H = Hk + Hxtheta  # H(k)
            E[:, i] = np.linalg.eigvalsh(H)  # E(k)
            idx = E[:, i].argsort()  # Ordering the energy bands at k
            E[:, i] = E[idx, i]  # Ordered E(k)
            i += 1

        # Bottom of the bands (centrifugal potential)
        V = np.linalg.eigvalsh(Hxtheta)
        idx = V.argsort()
        V = V[idx]

        return E, V

    def get_potential_matrix(self, V):
        """
        Computes the potential matrix used in the transfer matrix approach (part M_EV in the transfer matrix).

        If V is a number (rotational symmetry) the potential matrix becomes trivial. If V is a vector, it assumes the
        different entries are the FFT of V(theta, x) for fixed x.

        Params:
        ------
        V:         {np.array(floats)}   Potential at the point x (either number or vector with angular components.)

        Return:
        -------
        Vnm:       {np.array(floats)}   Potential matrix that is used in the transfer matrix approach.

        """

        try:
            return V * np.eye(2 * self.l_cutoff + 1)
        except ValueError:
            Vnm = np.zeros((self.modes.shape[0], self.modes.shape[0]), dtype='complex128')
            for i in range(self.modes.shape[0]):
                if i == 0:
                    Vnm += np.diag(np.repeat(V[0], self.modes.shape[0]), 0)
                else:
                    Vnm += np.diag(np.repeat(V[-i], self.modes.shape[0] - i), - i)
                    Vnm += np.diag(np.repeat(V[i], self.modes.shape[0] - i), i)

            return Vnm


# Old functions
def SVD_scattering(S, tol=1e-8):

    # Check unitarity
    n_modes = int(S.shape[0] / 2)
    if not np.allclose(S.T.conj() @ S, np.eye(len(S)), atol=1e-13):
        # raise ValueError('Unitarity of the scattering matrix not preserved!')
        print('Unitarity of the scattering matrix not preserved!')

    # Matrices to invert when going from transfer to scattering
    t = +S[n_modes:, 0: n_modes]
    tp = +S[0: n_modes, n_modes:]

    print('----------------------------')
    print('SVD info:')
    # SVD and replacement of small eigenvalues for t
    U, s, V = np.linalg.svd(t)
    sp = np.delete(s, np.where(s < tol)[0])
    Up = np.delete(U, np.where(s < tol)[0], axis=1)
    Vp = np.delete(V, np.where(s < tol)[0], axis=0)
    S[n_modes:, 0: n_modes] = Up * sp @ Vp

    # Checks
    t2 = Up * sp @ Vp
    norm = np.linalg.norm(s)
    print('Number of singular values dropped for t:', len(s) - len(sp))
    print('Max singular value for t: ', np.max(s))
    print('Min singular value for t: ', np.min(s))
    print('Norm of the singular values:', norm)
    print('SVD changes t: ', not np.allclose(t, t2, atol=1e-16))


    # SVD and replacement of small eigenvalues for tp
    U, s, V = np.linalg.svd(tp)
    sp = np.delete(s, np.where(s < tol)[0])
    Up = np.delete(U, np.where(s < tol)[0], axis=1)
    Vp = np.delete(V, np.where(s < tol)[0], axis=0)
    S[0: n_modes, n_modes:] = Up * sp @ Vp

    # Checks
    t2p = Up * sp @ Vp
    norm = np.linalg.norm(s)
    print('Number of singular values dropped for tp:', len(s) - len(sp))
    print('Max singular value for tp: ', np.max(s))
    print('Min singular value for tp: ', np.min(s))
    print('Norm of the singular values:', norm)
    print('SVD changes tp: ', not np.allclose(tp, t2p, atol=1e-16))

    # Check unitarity
    if not np.allclose(S.T.conj() @ S, np.eye(len(S)), atol=1e-8):
        # raise ValueError('Unitarity of the scattering matrix not preserved!')
        print('Unitarity of the scattering matrix not preserved!')

    return S

def stabilise_transfer(T, debug=False):

    # Initial checks
    if np.isnan(T).any():
        raise OverflowError('Too large T entries!')

    # Main function
    T_new = np.zeros(T.shape, dtype='complex128')
    T_new[:, 0] = T[:, 0] / np.linalg.norm(T[:, 0])
    for i in range(1, T.shape[1]):
        aux_sum = np.sum(np.tile(T[:, :i].T @ T[:, i], T.shape[0]).reshape(T.shape[0], i) * T[:, :i], axis=1)
        T_new[:, i] = T[:, i] - aux_sum
        T_new[:, i] = T_new[:, i] / np.linalg.norm(T_new[:, i])

    # Final checks
    n_modes = int(T.shape[0] / 2)
    T_check = T_new.T.conj() @ np.kron(sigma_z, np.eye(n_modes)) @ T_new
    # if not np.allclose(T_new.T.conj() @ np.kron(sigma_z, np.eye(n_modes)) @ T_new, np.kron(sigma_z, np.eye(n_modes))):
    #     raise ValueError('Current flow not preserved by the transfer matrix!')

    # Debug checks
    if debug:
        T_new2 = np.zeros(T.shape, dtype='complex128')
        T_new2[:, 0] = T[:, 0] / np.linalg.norm(T[:, 0])

        for i in range(1, T.shape[1]):
            T_new2[:, i] = T[:, i]
            for j in range(0, i):
                T_new2[:, i] += - np.dot(T[:, j], T[:, i]) * T[:, j]
            T_new2[:, i] = T_new2[:, i] / np.linalg.norm(T_new2[:, i])

        if not np.allclose(T_new, T_new2):
            raise ValueError('Method for stabilising T not being consistent with naive calculation')

    return T_new

def gaussian_correlated_potential_1D(x_vec, strength, correlation_length, vf, Nq, Vn=None, phi_n=None):
    """
    Generates a sample of a gaussian correlated potential V(x) with strength,
    a certain correlation length and Nq points in momentum space.

    Params:
    ------
    x_vec:                 {np.array(float)}  Discretisation of the position grid (Equally spaced!!!!!!!!!!)
    strength:                     {np.float}  Strength of the potential in units of (hbar vf /corr_length)^2
    correlation_length:           {np.float}  Correlation length of the distribution in nm
    vf:                           {np.float}  Fermi velocity in nm meV
    Nq:                             {np.int}  Number of points to work in momentum space
    Vn:                    {np.array(float)}  Ready calculated distribution of momentum modes (length Nq)
    phi_n:                 {np.array(float)}  Ready calculated distribution of random phases (length Nq)

    Returns:
    -------
    Vgauss:                {np.array(float)}  Gaussian correlated potential sample
    Vn:                    {np.array(float)}  Distribution of momentum modes (length Nq)
    phi_n:                 {np.array(float)}  Distribution of random phases (length Nq)
    """

    # Checks and defs
    L = x_vec[-1] - x_vec[0];
    dx = x_vec[1] - x_vec[0]
    qmax = 2 * pi * Nq / L
    if not isinstance(Nq, int): raise ValueError('Nq must be an integer!')
    if dx < 2 * pi / qmax: raise ValueError('dx must be larger than pi/qmax for FT to work')
    Vgauss = np.zeros(x_vec.shape)
    n = np.arange(1, Nq + 1)

    # Generate modes of the potential
    if Vn is None and phi_n is None:
        std_n = np.sqrt(
            strength * (vf ** 2 / correlation_length) * np.exp(-0.5 * (correlation_length * 2 * pi * n / L) ** 2))
        phi_n = np.random.uniform(0, 2 * pi, size=Nq)
        Vn = np.random.normal(scale=std_n)
        V0 = np.random.normal(scale=np.sqrt(strength * (vf ** 2 / correlation_length)))
    elif Vn is not None and phi_n is not None:
        pass
    else:
        raise ValueError('Vqn and phi_qn must both be determined or undetermined at the same time')

    # Translate to real space
    for i, x in enumerate(x_vec):  Vgauss[i] = V0 / np.sqrt(L) + (2 / np.sqrt(L)) * np.dot(Vn, np.cos(
        2 * pi * n * x / L + phi_n))
    return Vgauss, Vn, phi_n

def gaussian_correlated_potential_2D(x_vec, theta_vec, r, strength, correlation_length, vf, Nq, Nl, Vnm=None, phi_nm=None):
    """
    Generates a sample of a gaussian correlated potential V(x, theta) with strength,
    a certain correlation length and Nq, Ntheta points in momentum space.

    Params:
    ------
    x_vec:                 {np.array(float)}  Discretisation of the position grid (Equally spaced!!!!!!!!!!)
    theta_vec:             {np.array(float)}  Discretisation of the angular grid (Equally spaced!!!!!!!!!!)
    r:                            {np.float}  Radius of the geometry (CONSTANT!!!!!!!!!)
    strength:                     {np.float}  Strength of the potential in units of (hbar vf /corr_length)^2
    correlation_length:           {np.float}  Correlation length of the distribution in nm
    vf:                           {np.float}  Fermi velocity in nm meV
    Nq:                             {np.int}  Number of points in momentum space x
    Ntheta:                         {np.int}  Number of points in momentum space theta
    Vnm:                   {np.array(float)}  Ready calculated distribution of momentum modes (length Nq, Ntheta)
    phi_nm:                {np.array(float)}  Ready calculated distribution of random phases (length Nq, Ntheta)

    Returns:
    -------
    Vgauss:                 {np.array(float)}  Gaussian correlated potential sample of shape(x_vec, theta_vec)
    Vnm:                    {np.array(float)}  Distribution of momentum modes (length Nq)
    phi_nm:                 {np.array(float)}  Distribution of random phases (length Nq)
    """

    # Checks and defs
    L = x_vec[-1] - x_vec[0]
    dx = x_vec[1] - x_vec[0];
    dtheta = theta_vec[1] - theta_vec[0]
    qmax = 2 * pi * Nq / L;
    lmax = Nl / r
    if not isinstance(Nq, int): raise TypeError('Nq must be an integer!')
    if not isinstance(Nl, int): raise TypeError('Ntheta must be an integer!')
    if dx < 2 * pi / qmax: raise ValueError('dx must be larger than pi / qmax for FT to work')
    if dtheta < 2 * pi / (r * lmax): raise ValueError('dtheta must be larger than pi / (r * lmax) for FT to work')
    # N = np.tile(np.arange(0, Nq, dtype=float), Nl).reshape((Nq, Nl))
    # M = np.repeat(np.arange(0, Nl, dtype=float), Nq).reshape((Nq, Nl))
    N = np.tile(np.arange(-Nq, Nq, dtype=float), 2 * Nl).reshape((2 * Nq, 2 * Nl))
    M = np.repeat(np.arange(-Nl, Nl, dtype=float), 2 * Nq).reshape((2 * Nq, 2 * Nl))

    # Generate modes of the potential
    if Vnm is None and phi_nm is None:
        scale_std = np.sqrt(strength) * vf
        func_std = np.exp(-0.25 * (correlation_length ** 2) * ((2. * pi * N / L) ** 2 + (M / r) ** 2))
        # phi_nm = np.random.uniform(0., 2 * pi, size=(Nq, Ntheta))
        # Vnm = np.random.normal(scale=scale_std * func_std)
        phi_aux = np.random.uniform(0., 2 * pi, size=(2 * Nq - 1, Nl - 1))
        # phi2_aux
        # V =
    elif Vnm is not None and phi_nm is not None:
        pass
    else:
        raise ValueError('Vnm and phi_nm must both be determined or undetermined at the same time')

    # Transforming to real space
    Vgauss = np.zeros((x_vec.shape[0], theta_vec.shape[0]))
    for i, x in enumerate(x_vec):
        for j, theta in enumerate(theta_vec):
            # Vgauss[i, j]  = (4 / np.sqrt(L * 2 * pi * r)) * np.sum(Vnm[1:, 1:] * np.cos(2 * pi * N[1:, 1:] * x / L) * np.cos(M[1:, 1:] * theta + phi_nm[1:, 1:]))
            # Vgauss[i, j] += (2 / np.sqrt(L * 2 * pi * r)) * np.dot(Vnm[1:, 0], np.cos(M[1:, 0] * theta + phi_nm[1:, 0]))
            # Vgauss[i, j] += (2 / np.sqrt(L * 2 * pi * r)) * np.dot(Vnm[0, 1:], np.cos(2 * pi * N[0, 1:] * x / L + phi_nm[0, 1:]))
            # Vgauss[i, j] += (1 / np.sqrt(L * 2 * pi * r)) * Vnm[0, 0]
            Vgauss[i, j] = (1. / np.sqrt(L * 2. * pi * r)) * np.sum()

    return Vgauss



  # def get_scattering_states_interval_method(self, E, theta_vec, debug=False):
  #
  #       #  Preallocation
  #       psi_scatt_up = np.zeros((len(theta_vec), self.n_regions), 'complex128')
  #       psi_scatt_down = np.zeros((len(theta_vec), self.n_regions), 'complex128')
  #
  #       # Full scattering matrix
  #       S = None
  #       for i in range(0, self.n_regions):
  #           S = self.get_scattering_matrix(E, **self.geometry[i], S=S)
  #
  #       if debug:
  #           if np.abs(E) < 1:
  #               t_analytic = np.diag(1 / np.cosh(self.L * (self.modes - 0.5) / self.rad))
  #               r_analytic = np.diag(- np.sinh(self.L * (self.modes - 0.5) / self.rad) / np.cosh(
  #                   self.L * (self.modes - 0.5) / self.rad))
  #               if not np.allclose(S[0: self.Nmodes, 0: self.Nmodes], r_analytic):
  #                   raise ValueError('Analytic check failed for reflection matrix!')
  #               if not np.allclose(S[self.Nmodes:, 0: self.Nmodes], t_analytic):
  #                   raise ValueError('Analytic check failed for transmission matrix!')
  #
  #       # State at the leads
  #       phi_inL_lead, phi_outR_lead = self.get_max_transmitted_state(E, debug=False)
  #       phi_outL_lead = S[0: self.Nmodes, 0: self.Nmodes] @ phi_inL_lead
  #
  #       # Transfer to position x
  #       n_modes = len(phi_inL_lead)
  #       phi_inL = phi_inL_lead
  #       phi_outL = phi_outL_lead
  #       for i in range(0, self.n_regions):
  #
  #           # r, r', t, t' for the slab dx
  #           S = self.get_scattering_matrix(E, **self.geometry[i])
  #           r = S[0: n_modes, 0: n_modes]
  #           rp = S[n_modes:, n_modes:]
  #           t = S[n_modes:, 0: n_modes]
  #           tp = S[0: n_modes, n_modes:]
  #
  #           if debug:
  #               print('---------------------------------------')
  #               print('Region: {}/{}'.format(i, self.n_regions))
  #               print('max eigenvalue inv(tp): ', np.max(np.abs(np.linalg.eig(np.linalg.inv(tp))[0])))
  #               print('max eigenvalue r: ', np.max(np.abs(np.linalg.eig(r)[0])))
  #               print('max eigenvalue t: ', np.max(np.abs(np.linalg.eig(t)[0])))
  #               print('max eigenvalue rp: ', np.max(np.abs(np.linalg.eig(rp)[0])))
  #               print('max value phi_outL: ', np.max(np.abs(phi_outL)))
  #               print('max value phi_inL: ', np.max(np.abs(phi_inL)))
  #               print('Condition number for tp: ', np.linalg.cond(tp), ', det(tp):', np.abs(np.linalg.det(tp)))
  #               if np.linalg.cond(tp) > 10:
  #                   raise ValueError('Non-invertible matrix encountered in the scattering matrix')
  #               print('---------------------------------------')
  #
  #
  #           # State at the right part of the slab
  #           phi_inR = np.linalg.inv(tp) @ (phi_outL - r @ phi_inL)
  #           phi_outR = t @ phi_inL + rp @ phi_inR
  #
  #           # States at the left part of the new slab
  #           phi_inL = phi_outR
  #           phi_outL = phi_inR
  #           phi = np.concatenate((phi_inL, phi_outL))
  #           print('Norm of phi: ', np.linalg.norm(phi))
  #
  #           # Scattering states for the slab
  #           for j, theta in enumerate(theta_vec):
  #               trans_mode = transport_mode(self.geometry[i]['x0'], theta, self.geometry[i]['r'], self.modes, E, self.vf, lead=False)
  #               psi_scatt_up[j, i] = np.dot(trans_mode, phi[: self.Nmodes])
  #               psi_scatt_down[j, i] = np.dot(trans_mode, phi[self.Nmodes:])
  #
  #       # if debug:
  #       #     if not np.allclose(phi_outR_lead, phi_outR):
  #       #         raise ValueError('Transferred and transmitted states do not coincide!')
  #
  #       norm_scatt = np.sqrt(np.sum(psi_scatt_up * psi_scatt_up.conj() + psi_scatt_down * psi_scatt_down.conj()))
  #       psi_scatt_up, psi_scatt_down = psi_scatt_up / norm_scatt, psi_scatt_down / norm_scatt
  #       return psi_scatt_up, psi_scatt_down
