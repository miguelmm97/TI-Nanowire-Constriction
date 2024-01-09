from dataclasses import dataclass, field
from numpy import pi
from numpy.linalg import inv
from scipy.linalg import expm
import numpy as np
from numpy.fft import ifftshift, ifft, fftshift, ifft2, fft2
import numba as nb
from numba import njit  # "nopython just in time (compiler)"

# nb.set_num_threads(1)

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
def step(x1, x2, sigma=None):
    # Smoothened step function theta(x1-x2)
    if sigma is None:
        return np.heaviside(x1 - x2, 1)
    else:
        return 0.5 + (1 / pi) * np.arctan(sigma * (x1 - x2))

def geom_nc(x, x1, x2, r1, r2, sigma=None):
    # x1, r1: Initial widest part
    # x2, r2: Final narrow part
    return r1 + (r2 - r1) * step(x, x2, sigma) + ((r2 - r1) / (x2 - x1)) * (x - x1) * (
            step(x, x1, sigma) - step(x, x2, sigma))

def geom_cons(x, x1, x2, x3, r1, r2, sigma):
    # x1, r1: Initial lead
    # x2, r2: Constriction
    # x3: Start of the second cone
    return geom_nc(x, x1, x2, r1, r2, sigma) + geom_nc(-x + x2 + x3, x1, x2, r1, r2, sigma) - r2


# Transfer and scattering
def M_Ax(modes, dx, w, h, B_perp=0.):
    if w is None or h is None:
        return 0

    else:
        P = 2 * (w + h)  # Perimeter of the nanostructure at x (in nm)
        r = w / (w + h)  # Aspect ratio of the nanostructure at x (in nm)
        C = - 1j * (nm ** 2 / hbar) * e * B_perp * P / pi ** 2  # Auxiliary constant for the matrix elements
        M = np.zeros(shape=(len(modes), len(modes)), dtype=np.complex128)  # Mode mixing matrix for the vector potential
        if B_perp != 0:
            i = 0
            for n1 in modes:
                j = 0
                for n2 in modes:
                    if (n1 - n2) % 2 != 0:
                        m = n1 - n2
                        M[i, j] = C * ((-1) ** ((m + 1) / 2)) * np.sin(m * pi * r / 2) / m ** 2
                    j += 1
                i += 1

        return np.kron(sigma_0, M) * dx

def M_theta(modes, dx, R, dR, w=None, h=None, B_par=0.):
    C = 0.5 * (nm ** 2) * e * B_par / hbar
    A_theta = C * R ** 2 if (w is None or h is None) else C * (w * h / pi)  # A_theta = eBa²/2hbar
    M = (np.sqrt(1 + dR ** 2) / R) * np.diag(modes - 0.5 + A_theta)  # 1/R (n-1/2 + eBa²/2hbar) term

    return np.kron(sigma_x, M) * dx

def M_EV(modes, dx, dR, E, vf, Vnm=None):
    if Vnm is None:
        Vnm = np.zeros(len(modes))
    else:
        if Vnm.shape != (len(modes), len(modes)):
            raise AssertionError('Vnm must be the Fourier transform matrix of V')

    M = (1j / vf) * np.sqrt(1 + dR ** 2) * (E * np.eye(len(modes)) - Vnm)  # i ( E delta_nm + V_nm) / vf term
    return np.kron(sigma_z, M) * dx

def transfer_to_scattering(T):
    # Transform from the transfer matrix to the scattering matrix
    # transfer_matrix: Transfer matrix to translate to scattering
    # n_modes: Number of modes contributing to transport (N_states/2 because spin momentum locking)

    # Check flux conservation
    n_modes = int(T.shape[0] / 2)
    if not np.allclose(T.T.conj() @ np.kron(sigma_z, np.eye(n_modes)) @ T, np.kron(sigma_z, np.eye(n_modes)), atol=1e-13):
        raise ValueError('Current flow not preserved by the transfer matrix!')

    # Divide transfer matrix
    inv_t = T[0: n_modes, 0: n_modes]
    inv_tp = T[n_modes:, n_modes:]
    inv_r = T[n_modes:, 0: n_modes]
    inv_rp = T[0: n_modes, n_modes:]

    # Check invertibility
    if np.linalg.cond(inv_t) > 10 or np.linalg.cond(inv_tp) > 10:
        raise ValueError('Non-invertible matrix encountered in the transfer matrix')

    # Transform to scattering
    t = np.linalg.inv(inv_t).T.conj()
    tp = np.linalg.inv(inv_tp)
    r = - tp @ inv_r
    rp = inv_rp @ tp
    S = np.block([[r, tp], [t, rp]])

    # Check unitarity
    if not np.allclose(S.T.conj() @ S, np.eye(len(S)), atol=1e-13):
        raise ValueError('Unitarity of the scattering matrix not preserved!')

    return S

def scattering_to_transfer(S):

    # Check unitarity
    n_modes = int(S.shape[0] / 2)
    if not np.allclose(S.T.conj() @ S, np.eye(len(S)), atol=1e-13):
        # raise ValueError('Unitarity of the scattering matrix not preserved!')
        print('Unitarity of the scattering matrix not preserved!')

    # Divide scattering matrix
    r = S[0: n_modes, 0: n_modes]
    rp = S[n_modes:, n_modes:]
    t = S[n_modes:, 0: n_modes]
    tp = S[0: n_modes, n_modes:]

    # Check invertibility
    # if np.linalg.cond(t) > 10 or np.linalg.cond(tp) > 10:
    #     raise ValueError('Non-invertible matrix encountered in the scattering matrix')
    print('Condition number for t: ', np.linalg.cond(t), ', det(t):', np.linalg.det(t))
    print('Condition number for tp: ', np.linalg.cond(tp), ', det(tp):', np.linalg.det(tp))

    # Transform to transfer matrix
    inv_tp = np.linalg.inv(tp)
    T_00 = np.linalg.inv(t.T.conj())
    T_01 = rp @ inv_tp
    T_10 = - inv_tp @ r
    T_11 = inv_tp
    T = np.block([[T_00, T_01], [T_10, T_11]])

    # Check flux conservation
    if not np.allclose(T.T.conj() @ np.kron(sigma_z, np.eye(n_modes)) @ T, np.kron(sigma_z, np.eye(n_modes)), atol=1e-8):
        # raise ValueError('Current flow not preserved by the transfer matrix!')
        print('Current flow not preserved by the transfer matrix!')
    return T

def scat_product(s1, s2):
    # Product combining two scattering matrices
    # s1, s2: Scattering matrices to combine
    # n_modes: Number of modes contributing to transport (N_states/2 because spin momentum locking)

    # Check shape and unitarity
    if s1.shape != s2.shape:
        raise ValueError(" Different size for scattering matrices")
    if not np.allclose(s1.T.conj() @ s1, np.eye(len(s1)), atol=1e-8):
        print('Unitarity of the scattering matrix not preserved!')
        # raise ValueError('Unitarity of the scattering matrix not preserved!')
    if not np.allclose(s2.T.conj() @ s2, np.eye(len(s2)), atol=1e-8):
        print('Unitarity of the scattering matrix not preserved!')
        # raise ValueError('Unitarity of the scattering matrix not preserved!')

    # Divide scattering matrix
    n_modes = int(s1.shape[0] / 2)
    r1, r2 = s1[0: n_modes, 0: n_modes], s2[0: n_modes, 0: n_modes]
    r1p, r2p = s1[n_modes:, n_modes:], s2[n_modes:, n_modes:]
    t1, t2 = s1[n_modes:, 0: n_modes], s2[n_modes:, 0: n_modes]
    t1p, t2p = s1[0: n_modes, n_modes:], s2[0: n_modes, n_modes:]

    # Check invertibility
    Id = np.eye(n_modes)
    if np.linalg.cond(Id - r1p @ r2) > 10 or np.linalg.cond(Id - r2 @ r1p) > 10:
        raise ValueError('Non-invertible matrix encountered in the product of scattering matrices')

    # Product of S1 S2
    inv_r1pr2 = inv(Id - r1p @ r2)
    r = r1 + t1p @ r2 @ inv_r1pr2 @ t1
    rp = r2p + t2 @ inv_r1pr2 @ r1p @ t2p
    t = t2 @ inv_r1pr2 @ t1
    tp = t1p @ inv(Id - r2 @ r1p) @ t2p
    scat_matrix = np.block([[r, tp], [t, rp]])

    # Check unitarity
    if not np.allclose(scat_matrix.T.conj() @ scat_matrix, np.eye(len(scat_matrix)), atol=1e-8):
        print('Unitarity of the scattering matrix not preserved!')
        # raise ValueError('Unitarity of the scattering matrix not preserved!')

    return scat_matrix

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


# Gaussian correlated potentials
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
    xi:           {np.float}  Correlation length of the distribution in nm
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


@dataclass
class transport:
    """ Transport calculations on 3dTI nanostructures based on their effective surface theory."""

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
        """
        Adds a nanowire to the geometry of the nanostructure.

        Params:
        -------
        x0:    {float} Initial point of the nanowire
        xf:    {float} Final point of the nanowire
        r:     {float} Radius of the nanowire
        w:     {float} Width of the nanowire
        h:     {float} Height of the nanowire

        Return:
        -------
        None, but updates self. geometry

        """

        # Initial checks
        if self.n_regions != 0 and x0 != self.geometry[self.n_regions - 1]['xf']:
            raise ValueError('Regions dont match')
        if r is None and (w is None or h is None):
            raise ValueError('Need to specify r or (w, h)')

        # Function
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
        """
        Adds a nanowire to the geometry of the nanostructure.

        Params:
        -------
        x0:         {float} Initial point of the nanocone
        xf:         {float} Final point of the nanocone
        n_points:     {int} Number of points discretising the cone
        sigma:      {float} Smoothing factor for the step functions (the bigger, the sharper)
        r1:         {float} Initial radius of the nanocone
        w1:         {float} Initial width of the nanocone
        h1:         {float} Initial height of the nanocone
        r2:         {float} Final radius of the nanocone
        w2          {float} Final width of the nanocone
        h2:         {float} Final height of the nanocone

        Return:
        -------
        None, but updates self. geometry

        """

        # Initial checks
        if self.n_regions != 0 and x0 != self.geometry[self.n_regions - 1]['xf']:
            raise ValueError('Regions dont match')
        if r1 is None and (w1 is None or h1 is None):
            raise ValueError('Need to specify r or (w, h)')
        if r2 is None and (w2 is None or h2 is None):
            raise ValueError('Need to specify r or (w, h)')

        # Function
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
    def get_transfer_matrix(self, E, x0, xf, dx, V, r, w, h, T=None, backwards=False, **kwargs):

        M = M_EV(self.modes, dx, 0, E, self.vf, V)
        M += M_theta(self.modes, dx, r, 0, w, h, B_par=self.B_par)
        M += M_Ax(self.modes, dx, w, h, B_perp=self.B_perp)
        if backwards: M = - M
        T = expm(M * (xf - x0) / dx) if T is None else expm(M * (xf - x0) / dx) @ T

        # Final checks
        n_modes = int(T.shape[0] / 2)
        if not np.allclose(T.T.conj() @ np.kron(sigma_z, np.eye(n_modes)) @ T, np.kron(sigma_z, np.eye(n_modes)),atol=1e-13):
            raise ValueError('Current flow not preserved by the transfer matrix!')

        return T

    def get_scattering_matrix(self, E, region_type, x0, xf, dx, n_points, V, r, w, h, S=None, backwards=False, debug=False):

        # For nanowires
        if region_type == 'nw':

            # No need for discretisation
            if n_points is None:
                T = self.get_transfer_matrix(E, x0, xf, dx, V, r, w, h, backwards=backwards)
                S = transfer_to_scattering(T) if S is None else scat_product(S, transfer_to_scattering(T))

                # Final checks
                if np.isnan(S).any():
                    raise OverflowError('Need for discretisation!')

                # Debug checks
                if debug:
                    if not np.allclose(T, scattering_to_transfer(S)):
                        raise ValueError('Scattering to transfer not working!')

            else:
                raise NotImplementedError('Part of the code for discretising nanowires needs to be implemented')
        else:
            raise NotImplementedError('Part of the code for nanocodes needs to be implemented')

        return S

    def get_Landauer_conductance(self, E, save_S=False):
        """
        Calculates the Landauer conductance along the whole geometry at the Fermi energy E.

        Param:
        ------
        E: {float} Fermi energy

        Return:
        ------
        G: {float} Conductance at the Fermi energy
        """

        S = None
        T_check = np.eye(2 * self.Nmodes)
        for i in range(0, self.n_regions):
            # print('--------------------')
            # print('--------------------')
            # print('region {}'.format(i))

            S = self.get_scattering_matrix(E, **self.geometry[i], S=S)
            # T_check = T @ T_check
            # T_check1 = scattering_to_transfer(S)

            # print('--------------------')
            # print('Propagation methods:')
            # print('--------------------')
            # print('Equivalent transfer matrices: ', np.allclose(T_check, T_check1))
            # print('Flux conservation from scattering propagation: ', np.allclose(T_check1.T.conj() @ np.kron(sigma_z, np.eye(self.Nmodes)) @ T_check1, np.kron(sigma_z, np.eye(self.Nmodes))))
            # print('Flux conservation from transfer propagation: ', np.allclose(T_check.T.conj() @ np.kron(sigma_z, np.eye(self.Nmodes)) @ T_check, np.kron(sigma_z, np.eye(self.Nmodes))))
            # A = None
            # T_check2 = stabilise_transfer(T_check1)
            # S = transfer_to_scattering(T_check2)
            # print('Transfer from scattering after normalising: ')
            # print(np.allclose(T_check2.T.conj() @ np.kron(sigma_z, np.eye(self.Nmodes)) @ T_check2, np.kron(sigma_z, np.eye(self.Nmodes))))


        if save_S: self.S_tot = S

        t = S[self.Nmodes:, 0: self.Nmodes]
        G = np.trace(t.T.conj() @ t)
        return G

    def get_scattering_states_S2T(self, E, theta_vec):

        psi_scatt = np.zeros((2 * len(theta_vec), self.n_regions), 'complex128')
        psi_scatt_up = np.zeros((len(theta_vec), self.n_regions), 'complex128')
        psi_scatt_down = np.zeros((len(theta_vec), self.n_regions), 'complex128')
        theta = np.repeat(theta_vec, self.Nmodes).reshape(len(theta_vec), self.Nmodes)
        Mmodes = np.tile(self.modes, len(theta_vec)).reshape(len(theta_vec), self.Nmodes)

        # Full scattering and transfer
        if self.S_tot is None:
            for i in range(0, self.n_regions):
                print('----------------------------')
                print('Region: ', i)
                self.S_tot = self.get_scattering_matrix(E, **self.geometry[i], S=self.S_tot)
                print('----------------------------')
                print('Before SVD:')
                T_0 = scattering_to_transfer(self.S_tot)
                self.S_tot = SVD_scattering(self.S_tot)
                print('----------------------------')
                print('After SVD:')
                T = scattering_to_transfer(self.S_tot)
                print('----------------------------')

        # Forward propagation
        phi = np.zeros((2 * self.Nmodes,), dtype='complex128')
        phi[0] = 1
        phi[self.Nmodes:] = self.S_tot[0: self.Nmodes, 0: self.Nmodes] @ phi[: self.Nmodes]
        phi = phi / np.linalg.norm(phi)
        phi = T @ phi
        phi = phi / np.linalg.norm(phi)

        # Change of basis
        phi = np.kron(sigma_x, np.eye(self.Nmodes)) @ phi

        # Backward propagation
        S = None
        for i in range(self.n_regions - 1, -1, -1):
            S = self.get_scattering_matrix(E, **self.geometry[i], backwards=True, S=S)
            T = scattering_to_transfer(SVD_scattering(S))
            phi = T @ phi
            phi = phi / np.linalg.norm(phi)
            psi_scatt[:, i] = np.kron(sigma_0, np.exp(1j * theta * Mmodes * self.geometry[i]['r'])) @ phi
            psi_scatt[:, i] = psi_scatt[:, i] / np.linalg.norm(psi_scatt[:, i])
            psi_scatt_up[:, i] = psi_scatt[:len(theta_vec), i]
            psi_scatt_down[:, i] = psi_scatt[len(theta_vec):, i]

        return psi_scatt_up, psi_scatt_down

    def get_scattering_states_Tproduct(self, E, theta_vec):

        psi_scatt = np.zeros((2 * len(theta_vec), self.n_regions), 'complex128')
        psi_scatt_up = np.zeros((len(theta_vec), self.n_regions), 'complex128')
        psi_scatt_down = np.zeros((len(theta_vec), self.n_regions), 'complex128')
        theta = np.repeat(theta_vec, self.Nmodes).reshape(len(theta_vec), self.Nmodes)
        Mmodes = np.tile(self.modes, len(theta_vec)).reshape(len(theta_vec), self.Nmodes)

        # Full scattering and transfer
        for i in range(0, self.n_regions):

            # Transfer through SVD chain
            T = self.get_transfer_matrix(E, **self.geometry[i])
            if i == 0:
                U, s, V = np.linalg.svd(T)
                T_tot = V
                temp = U * s
            elif i == self.n_regions - 1:
                T_tot = T @ temp @ T_tot
            else:
                U, s, V = np.linalg.svd(T @ temp)
                T_tot = V @ T_tot
                temp = U * s

            # Scattering
            if self.S_tot is None:
                self.S_tot = self.get_scattering_matrix(E, **self.geometry[i], S=self.S_tot)

        # Initial state
        phi = np.zeros((2 * self.Nmodes,), dtype='complex128')
        phi[0] = 1
        phi[self.Nmodes:] = self.S_tot[0: self.Nmodes, 0: self.Nmodes] @ phi[: self.Nmodes]
        phi = phi / np.linalg.norm(phi)

        # Forward propagation
        phi = T_tot @ phi
        phi = phi / np.linalg.norm(phi)

        # Change of basis
        phi = np.kron(sigma_x, np.eye(self.Nmodes)) @ phi

        # Backward propagation
        for i in range(self.n_regions - 1, -1, -1):
            T = self.get_transfer_matrix(E, **self.geometry[i], backwards=True)
            if i == self.n_regions - 1:
                U, s, V = np.linalg.svd(T)
                T_tot = V
                temp = U * s
                phi = temp @ T_tot @ phi
            elif i == 0:
                T_tot = T @ temp @ T_tot
                phi = T_tot @ phi
            else:
                U, s, V = np.linalg.svd(T @ temp)
                T_tot = V @ T_tot
                temp = U * s
                phi = temp @ T_tot @ phi

            phi = phi / np.linalg.norm(phi)
            psi_scatt[:, i] = np.kron(sigma_0, np.exp(1j * theta * Mmodes * self.geometry[i]['r'])) @ phi
            psi_scatt[:, i] = psi_scatt[:, i] / np.linalg.norm(psi_scatt[:, i])
            psi_scatt_up[:, i] = psi_scatt[:len(theta_vec), i]
            psi_scatt_down[:, i] = psi_scatt[len(theta_vec):, i]

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






