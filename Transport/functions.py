# Function file for the TI constriction project
import numpy as np
from numpy import pi
from numpy.linalg import inv
from scipy.linalg import expm

# Constants
hbar = 1e-34                # Planck's constant in Js
nm = 1e-9                   # Conversion from nm to m
ams = 1e-10                 # Conversion from Å to m
e = 1.6e-19                 # Electron charge in C
phi0 = 2 * pi * hbar / e    # Quantum of flux

# Pauli matrices
sigma_0 = np.eye(2)
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
tau_0 = np.eye(2)
tau_x = np.array([[0, 1], [1, 0]])
tau_y = np.array([[0, -1j], [1j, 0]])
tau_z = np.array([[1, 0], [0, -1]])

#%% Transport
def f_FD(E, mu, T):

    # Fermi-Dirac distribution
    k_B = 8.617333262e-2  # [meV/K]

    if T != 0:
        beta = 1 / (k_B * T)
        return 1 / (np.exp(beta * (E - mu)) + 1)
    else:
        return np.heaviside(mu - E, 1)

def df_FD(E, mu, T):

    # Derivative of the Fermi-Dirac distribution
    k_B = 8.617333262e-2  # [meV/K]
    beta = 1 / (k_B * T)

    if T != 0:
        return - beta * np.exp(beta * (E - mu)) / (np.exp(beta * (E - mu)) + 1) ** 2
    else:
        raise ValueError("T=0 limit undefined unless inside an integral!")

def transfer_to_scattering(transfer_matrix, n_modes):

    # Transform from the transfer matrix to the scattering matrix
    # transfer_matrix: Transfer matrix to translate to scattering
    # n_modes: Number of modes contributing to transport (N_states/2 because spin momentum locking)

    inv_T = transfer_matrix[0: n_modes, 0: n_modes]  # t^\dagger ^(-1)
    inv_Tp = transfer_matrix[n_modes:, n_modes:]     # t' ^(-1)
    inv_R = transfer_matrix[n_modes:, 0: n_modes]    # -t'^(-1) r
    inv_Rp = transfer_matrix[0: n_modes, n_modes:]   # r't'^(-1)

    T = np.conj(inv(inv_T)).T  # t
    Tp = inv(inv_Tp)           # t'
    R = - Tp @ inv_R           # r
    Rp = inv_Rp @ Tp           # r'

    scat_matrix = np.block([[R, Tp], [T, Rp]])  # scattering matrix

    return scat_matrix

def scat_product(s1, s2, n_modes):

    # Product combining two scattering matrices
    # s1, s2: Scattering matrices to combine
    # n_modes: Number of modes contributing to transport (N_states/2 because spin momentum locking)

    if s1.shape != s2.shape:
        raise ValueError(" Different size for scattering matrices")

    r1, r2 = s1[0: n_modes, 0: n_modes], s2[0: n_modes, 0: n_modes]  # r1, r2
    r1p, r2p = s1[n_modes:, n_modes:], s2[n_modes:, n_modes:]        # r1', r2'
    t1, t2 = s1[n_modes:, 0: n_modes], s2[n_modes:, 0: n_modes]      # t1, t2
    t1p, t2p = s1[0: n_modes, n_modes:], s2[0: n_modes, n_modes:]    # t1', t2'

    R = r1 + t1p @ r2 @ inv(np.eye(n_modes) - r1p @ r2) @ t1         # r
    Rp = r2p + t2 @ r1p @ inv(np.eye(n_modes) - r2 @ r1p) @ t2p      # r'
    T = t2 @ inv(np.eye(n_modes) - r1p @ r2) @ t1                    # t
    Tp = t1p @ inv(np.eye(n_modes) - r2 @ r1p) @ t2p                 # t'

    scat_matrix = np.block([[R, Tp], [T, Rp]])  # scattering matrix

    return scat_matrix

def transport_checks(transfer_matrix=None, scat_matrix=None):
    """
    Checks current conservation, unitarity and completeness of the transport calculation.
    """

    n_modes = int(len(scat_matrix[0, :]) / 2) if transfer_matrix is None else int(len(transfer_matrix[0, :]) / 2)
    sigma_z = np.array([[1, 0], [0, -1]])  # Pauli z

    # Conservation of the current
    if transfer_matrix is not None:
        check1 = transfer_matrix @ np.kron(sigma_z, np.eye(n_modes)) @ np.conj(transfer_matrix.T)
        if not np.allclose(np.kron(sigma_z, np.eye(n_modes)), check1): raise AssertionError('Transfer matrix does not conserve current')

    # Unitarity of the scattering matrix
    if scat_matrix is not None:
        # check2 = scat_matrix @ np.conj(scat_matrix.T)
        # if not np.allclose(np.eye(2 * n_modes), check2, 1e-5): raise AssertionError('Scattering matrix is not unitary')
        if not np.allclose(np.trace(np.conj(scat_matrix).T @ scat_matrix), n_modes * 2, 1e-5): raise AssertionError('Scattering matrix is not unitary')


    # Completeness of reflection and transmission
    if scat_matrix is not None:
        t, r = scat_matrix[n_modes:, 0: n_modes], scat_matrix[0: n_modes, 0: n_modes]
        t_dagger, r_dagger = np.conj(t.T), np.conj(r.T)
        # if not np.allclose(n_modes-np.trace(r_dagger @ r), np.trace(t_dagger @ t)): raise AssertionError('Reflexion doesnt add up to transmission')


def transport_calculation(n_modes, transfer_matrix, scat_matrix0, L_grid):
    # Performs the transport calculation (asuming no change in the transfer matrix with position)
    scat_matrix = scat_matrix0  # Scattering matrix

    for pos in range(L_grid):
        if pos == 0:
            scat_matrix = scat_matrix0
        else:
            scat_matrix = scat_product(scat_matrix, scat_matrix0, n_modes)  # Propagating the scattering matrix
            # transport_checks(transfer_matrix=transfer_matrix, scat_matrix=scat_matrix)
    t = scat_matrix[n_modes:, 0: n_modes]  # Transmission matrix
    t_dagger = np.conj(t.T)  # Conjugate transmission matrix
    G = np.trace(t_dagger @ t)  # Conductance / Gq
    return G, scat_matrix

def thermal_average(T, mu, E, G):

    # Compute the thermal average of a given conductance
    # T: Temperature
    # mu: Chemical potential
    # E: Range of energies where we want to integrate.
    # G: Conductance for that range of energies

    integrand = - G * df_FD(E, mu, T)
    return np.trapz(integrand, E)

def finite_voltage_bias(T, mu1, mu2, E, G):

    # Compute conductance in a finite Voltage bias
    # T: Temperature
    # mu1, mu2: Chemical potential in the left and right leads
    # E: Range of energies where we want to integrate.
    # G: Conductance for that range of energies

    integrand = G * (f_FD(E, mu1, T) - f_FD(E, mu2, T))
    return np.trapz(integrand, E) / (mu1 - mu2)

def M_Ax(modes, w, h, dx, B_perp=0):

    # Calculates the Ax term (mode mixing) in the exponent of the transfer matrix from x to x + dx
    # modes: Vector of angular momentum modes considered for transport
    # w, h: Width and height of the nanostructure at x (in nm)
    # B_perp: Magnetic field perpendicular to the cross-section of the nanostructure

    P = 2 * (w + h)                                    # Perimeter of the nanostructure at x (in nm)
    r = w / (w + h)                                    # Aspect ratio of the nanostructure at x (in nm)
    n_modes = int(len(modes))                          # Number of angular momentum modes
    M = np.zeros([n_modes, n_modes], complex)  # Mode mixing matrix for the vector potential

    # Off-diag term: e vf < n | A_x | m >
    if B_perp != 0:
        for i, n1 in enumerate(modes):
            for j, n2 in enumerate(modes):
                if (n1 - n2) % 2 != 0:
                    m = n1 - n2
                    M[i, j] = - 1j * (nm ** 2 / hbar) * e * B_perp * P * ((-1) ** ((m + 1) / 2)) * np.sin(
                        m * pi * r / 2) / (m * m * pi * pi)

    return np.kron(sigma_0, M) * dx

def M_theta(modes, R, dR, w, h, dx, B_par=0):

    # Calculates the angular momentum term (dtheta + Atheta) in the exponent of the transfer matrix from x to x + dx
    # modes: Vector of angular momentum modes considered for transport
    # R: Radius of the nanostructure at x
    # dR: Derivative of R(x) at x
    # w, h: Width and height of the nanostructure at x (in nm)
    # B_par: Magnetic field threaded through the cross-section of the nanostructure

    geom = np.sqrt(1 + dR ** 2) / R                              # Geometric factor
    A_theta = 0.5 * (nm ** 2) * e * B_par * w * h / (pi * hbar)  # A_theta = eBa²/2hbar
    M = geom * np.diag(modes - 0.5 + A_theta)                    # 1/R (n-1/2 + eBa²/2hbar) term

    return np.kron(sigma_x, M) * dx

def M_EV(modes, dR, E, vf, dx, V=None):

    # Calculates the energy-potential term in the exponent of the transfer matrix from x to x + dx
    # modes: Vector of angular momentum modes considered for transport
    # dR: Derivative of R(x) at x
    # E: Fermi energy
    # vf: Fermi velocity
    # dx: Length along which the transfer calculation is done
    # V: External potential matrix V_nm, that can be disorder, gate, ...etc

    if V is None:
        V = np.zeros(len(modes))

    geom = np.sqrt(1 + dR ** 2)                        # Geometric factor
    M = 1j * geom * (E * np.eye(len(modes)) - V) / vf  # i ( E delta_nm + V_nm) / vf term

    return np.kron(sigma_z, M) * dx

def transfer_matrix(modes, w, h, R, dR, dx, E, vf, V=None, B_par=0, B_perp=0):

    # Calculates the transfer matrix from x to x+dx for a TI nanostructure based on its effective surface theory,
    # assuming a rectangular cross-section that can be effectively treated as cylindrical.
    # modes: Vector of angular momentum modes considered for transport
    # w, h: Width and height of the nanostructure at x
    # R: Radius of the nanostructure at x
    # dR: Derivative of R(x) at x
    # E: Fermi energy
    # vf: Fermi velocity
    # dx: Length along which the transfer calculation is done
    # V: External potential matrix V_nm, that can be disorder, gate, ...etc
    # B_par: Magnetic field threaded along the cross-section of the nanostructure
    # B_perp: Magnetic field perpendicular to the cross-section of the nanostructure

    T = expm(M_EV(modes, dR, E, vf, dx, V=V) + M_theta(modes, R, dR, w, h, dx, B_par=B_par) + M_Ax(modes, w, h, dx, B_perp=B_perp))

    return T



#%% Geometry of the nanostructure
def step(x1, x2, sigma):
    # Smoothened step function theta(x1-x2)
    return 0.5 + (1 / pi) * np.arctan(sigma * (x1 - x2))

def geom_nc(x, x1, x2, r1, r2, sigma):
    # x1, r1: Initial widest part
    # x2, r2: Final narrow part
    return r1 + (r2 - r1) * step(x, x2, sigma) + ((r2 - r1) / (x2 - x1)) * (x - x1) * (step(x, x1, sigma) - step(x, x2, sigma))

def geom_cons(x, x1, x2, x3, r1, r2, sigma):
    # x1, r1: Initial lead
    # x2, r2: Constriction
    # x3: Start of the second cone
    return geom_nc(x, x1, x2, r1, r2, sigma) + geom_nc(-x + x2 + x3, x1, x2, r1, r2, sigma) - r2

def constriction(L_lead, L_nc, L_cons, h_lead, w_lead, h_cons, w_cons, sigma, sampling='uniform', n_x=None, n_leads=None, n_nc=None, n_cons=None):
    # Geometry of a symmetric rectangular constriction with smoothing sigma and n_x points


    if sampling == 'uniform':
        if n_x is None:
            raise ValueError('Need to specify number of points!')
        x0 = 0
        x1 = x0 + L_lead
        x2 = x1 + L_nc
        x3 = x2 + L_cons
        x4 = x3 + L_nc
        x5 = x4 + L_lead
        L = x5 - x0
        L_grid = np.linspace(x0, x5, n_x)                          # x grid
        dx = L / n_x                                               # Step
        h = geom_cons(L_grid, x1, x2, x3, h_lead, h_cons, sigma)   # Height
        w = geom_cons(L_grid, x1, x2, x3, w_lead, w_cons, sigma)   # Width
        R = (w + h) / pi                                           # Radius
        dR = np.diff(R) / dx                                       # Derivative radius

    elif sampling == 'cones':
        if (n_leads is None) or (n_cons is None) or (n_nc is None):
            raise ValueError('Need to specify number of points at each part of the constriction!')
        x0 = 0
        x1 = x0 + L_lead
        x2 = x1 + L_nc
        x3 = x2 + L_cons
        x4 = x3 + L_nc
        x5 = x4 + L_lead
        lead1 = np.linspace(x0, x1, n_leads)
        cone1 = np.linspace(x1, x2, n_nc)
        cons = np.linspace(x2, x3, n_cons)
        cone2 = np.linspace(x3, x4, n_nc)
        lead2 = np.linspace(x4, x5, n_leads)
        L_grid = np.concatenate([lead1, cone1, cons, cone2, lead2])
        dx = [L_lead / n_leads, L_nc / n_nc, L_cons / n_cons]
        h = geom_cons(L_grid, x1, x2, x3, h_lead, h_cons, sigma)  # Height
        w = geom_cons(L_grid, x1, x2, x3, w_lead, w_cons, sigma)  # Width
        R = (w + h) / pi  # Radius
        dR = np.diff(R) / dx[1]  # Derivative radius

    return L_grid, dx, h, w, R, dR, x0, x1, x2, x3, x4, x5


