# Function file for the TI constriction project
import numpy as np
from numpy import pi
from numpy.linalg import inv
from scipy.linalg import expm

# Constants
hbar = 1e-34                # Planck's constant in Js
nm = 1e-9                   # Conversion from nm to m
e = 1.6e-19                 # Electron charge in C
phi0 = 2 * pi * hbar / e    # Quantum of flux

# Pauli matrices
sigma_0 = np.eye(2)
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

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

def transport_checks(n_modes, transfer_matrix=None, scat_matrix=None):

    # Check the conservation of current and the unitarity condition for transfer/scattering matrices
    # n_modes: Number of modes contributing to transport (N_states/2 because spin momentum locking)

    sigma_z = np.array([[1, 0], [0, -1]])  # Pauli z

    # Conservation of the current
    if transfer_matrix is not None:
        check1 = transfer_matrix @ np.kron(sigma_z, np.eye(n_modes)) @ np.conj(transfer_matrix.T)
        print(np.allclose(np.kron(sigma_z, np.eye(n_modes)), check1))

    # Unitarity of the scattering matrix
    if scat_matrix is not None:
        check2 = scat_matrix @ np.conj(scat_matrix.T)
        print(np.allclose(np.eye(2 * n_modes), check2))

    # Completeness of reflection and transmission
    if scat_matrix is not None:
        t, r = scat_matrix[n_modes:, 0: n_modes], scat_matrix[0: n_modes, 0: n_modes]
        t_dagger, r_dagger = np.conj(t.T), np.conj(r.T)
        print(np.allclose(n_modes-np.trace(r_dagger @ r), np.trace(t_dagger @ t)))

def transport_calculation(n_modes, transfer_matrix, scat_matrix0, L_grid):
    # Performs the transport calculation
    scat_matrix = scat_matrix0  # Scattering matrix
    for pos in range(L_grid):
        scat_matrix = scat_product(scat_matrix, scat_matrix0, n_modes)  # Propagating the scattering matrix
    t = scat_matrix[n_modes:, 0: n_modes]  # Transmission matrix
    t_dagger = np.conj(t.T)  # Conjugate transmission matrix
    G = np.trace(t_dagger @ t)  # Conductance / Gq
    return G

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

def M_Ax(modes, w, h, B_perp, dx):

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

def M_theta(modes, R, dR, w, h, B_par, dx):

    # Calculates the angular momentum term (dtheta + Atheta) in the exponent of the transfer matrix from x to x + dx
    # modes: Vector of angular momentum modes considered for transport
    # R: Radius of the nanostructure at x
    # dR: Derivative of R(x) at x
    # w, h: Width and height of the nanostructure at x (in nm)
    # B_par: Magnetic field threaded through the cross-section of the nanostructure

    geom = np.sqrt(1 + dR ** 2) / R                       # Geometric factor
    A_theta = 0.5 * (nm ** 2) * e * B_par * w * h / hbar  # A_theta = eBa²/2hbar
    M = geom * np.diag(modes - 0.5 + A_theta)             # 1/R (n-1/2 + eBa²/2hbar) term

    return np.kron(sigma_x, M) * dx

def M_EV(modes, dR, E, vf, dx, V=None):

    # Calculates the energy-potential term in the exponent of the transfer matrix from x to x + dx
    # modes: Vector of angular momentum modes considered for transport
    # dR: Derivative of R(x) at x
    # E: Fermi energy
    # vf: Fermi velocity
    # dx: Length along which the transfer calculation is done
    # V: External potential matrix V_nm, that can be disorder, gate, ...etc

    geom = np.sqrt(1 + dR ** 2)                        # Geometric factor
    M = 1j * geom * (E * np.eye(len(modes)) + V) / vf  # i ( E delta_nm + V_nm) / vf term

    return np.kron(sigma_z, M) * dx

def transfer_matrix(modes, w, h, R, dR, dx, E, vf, V=None, B_par=None, B_perp=None):

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

    T = expm(M_EV(modes, dR, E, vf, dx, V) + M_theta(modes, R, dR, w, h, B_par, dx) + M_Ax(modes, w, h, B_perp, dx))

    return T



#%% Bi2Se3 Band structure
def xtranslation(x, y, n_x, n_y):
    # Translates the vector x one site in direction x
    # x, y: Vectors with the position of the lattice sites
    # n_x, n_y: Dimension s of the lattice grid
    transx = ((x + 1) % n_x) + n_x * y
    return transx

def ytranslation(x, y, n_x, n_y):
    # Translates the vector y one site in direction y
    # x, y: Vectors with the position of the lattice sites
    # n_x, n_y: Dimension s of the lattice grid
    transy = x + n_x * ((y + 1) % n_y)
    return transy

def spectrum(H):
    # Calculates the spectrum a of the given Hamiltonian
    # H: Hamiltonian for the model
    # n_particles: Number of particles we want (needed for the projector)

    energy, eigenstates = np.linalg.eigh(H)  # Diagonalise H
    idx = energy.argsort()  # Indexes from lower to higher energy
    energy = energy[idx]  # Ordered energy eigenvalues
    eigenstates = eigenstates[:, idx]  # Ordered eigenstates

    return energy, eigenstates

def Ham_nw_Bi2Se3_1(n_sites, n_orb, L_x, L_y, x, y, kz, t, lamb, lamb_z, eps, flux, periodicity_x=False, periodicity_y=False):
    # Builds the model Hamiltonian of a translationally-invariant Bi2Se3 nanowire in OBC at momentum kz.
    # n_sites: Number of sites of the lattice
    # n_orb: Number of orbitals of the model
    # L_x, L_y: System size on each direction
    # x, y: Position of the sites on each direction
    # kz: Momentum alongside the nanowire
    # t, lamb, lamb_z, eps: Parameters of the model
    # flux: Magnetic flux through the cross-section in units of the flux quantum
    # periodicity_xy: Trye if we want any of these directions to be periodic

    # Declarations
    n_states = n_sites * n_orb                                      # Number of basis states
    cross_section = (L_x - 1) * (L_y - 1)                           # Area of the xy cross-section
    transx = xtranslation(x, y, L_x, L_y)                           # List of neighbours in x direction
    transy = ytranslation(x, y, L_x, L_y)                           # List of neighbours in y direction
    H_offdiag = np.zeros((n_states, n_states), dtype='complex_')    # Hamiltonian for the xy cross-section

    # Block hoppings along x, y
    block_x = (1j * 0.5 * lamb * np.kron(sigma_z, sigma_y)) - t * np.kron(sigma_x, sigma_0)
    block_y = (-1j * 0.5 * lamb * np.kron(sigma_z, sigma_x)) - t * np.kron(sigma_x, sigma_0)
    block_z = (eps - 2 * t * np.cos(kz)) * np.kron(sigma_x, sigma_0) + (lamb_z * np.sin(kz)) * np.kron(sigma_y, sigma_0)
    peierls = np.exp((2 * pi * 1j / cross_section) * flux * y)

    # Hopping along x and y
    for site in range(0, n_sites):

        # Sites connected by the hamiltonian
        row = site * n_orb
        colx = transx[site] * n_orb
        coly = transy[site] * n_orb

        # Hopping along x
        if periodicity_x:
            H_offdiag[row: row + n_orb, colx: colx + n_orb] = block_x * peierls[site]
        else:
            if (site + 1) % L_x != 0:
                H_offdiag[row: row + n_orb, colx: colx + n_orb] = block_x * peierls[site]

        # Hopping along y
        if periodicity_y:
            H_offdiag[row: row + n_orb, coly: coly + n_orb] = block_y
        else:
            if (site + L_x) < n_sites:
                H_offdiag[row: row + n_orb, coly: coly + n_orb] = block_y

    # Hamiltonian
    H_diag = np.kron(np.eye(n_sites), block_z)
    H_offdiag = H_offdiag + np.conj(H_offdiag.T)
    H = H_diag + H_offdiag

    return H


def Ham_nw_Bi2Se3_2(n_sites, n_orb, L_x, L_y, x, y, kz, C, M, D1, D2, B1, B2, A1, A2, flux, periodicity_x=False, periodicity_y=False):
    # Builds the model Hamiltonian of a translationally-invariant Bi2Se3 nanowire in OBC at momentum kz.
    # n_sites: Number of sites of the lattice
    # n_orb: Number of orbitals of the model
    # L_x, L_y: System size on each direction
    # x, y: Position of the sites on each direction
    # kz: Momentum alongside the nanowire
    # A1, A2, B1, B2, C, D1, D2, M: Parameters of the model
    # flux: Magnetic flux through the cross-section in units of the flux quantum
    # periodicity_xy: True if we want any of these directions to be periodic

    # Declarations
    n_states = n_sites * n_orb                                      # Number of basis states
    cross_section = (L_x - 1) * (L_y - 1)                           # Area of the xy cross-section
    transx = xtranslation(x, y, L_x, L_y)                           # List of neighbours in x direction
    transy = ytranslation(x, y, L_x, L_y)                           # List of neighbours in y direction
    H_offdiag = np.zeros((n_states, n_states), dtype='complex_')    # Hamiltonian for the xy cross-section

    # Block hoppings along x, y
    block_z = ((C + 4 * D2) + 2 * D1 * (1 - np.cos(kz))) * np.kron(sigma_0, sigma_0) \
            + ((M - 4 * B2) + 2 * B1 * (1 - np.cos(kz))) * np.kron(sigma_z, sigma_0) \
            + A1 * np.sin(kz) * np.kron(sigma_z, sigma_z)
    block_y = D2 * np.kron(sigma_0, sigma_0) - B2 * np.kron(sigma_z, sigma_0) - 0.5 * 1j * A2 * np.kron(sigma_x, sigma_x)
    block_x = D2 * np.kron(sigma_0, sigma_0) - B2 * np.kron(sigma_z, sigma_0) - 0.5 * 1j * A2 * np.kron(sigma_x, sigma_y)
    peierls = np.exp((2 * pi * 1j / cross_section) * flux * y)

    # Hopping along x and y
    for site in range(0, n_sites):

        # Sites connected by the hamiltonian
        row = site * n_orb
        colx = transx[site] * n_orb
        coly = transy[site] * n_orb

        # Hopping along x
        if periodicity_x:
            H_offdiag[row: row + n_orb, colx: colx + n_orb] = block_x * peierls[site]
        else:
            if (site + 1) % L_x != 0:
                H_offdiag[row: row + n_orb, colx: colx + n_orb] = block_x * peierls[site]

        # Hopping along y
        if periodicity_y:
            H_offdiag[row: row + n_orb, coly: coly + n_orb] = block_y
        else:
            if (site + L_x) < n_sites:
                H_offdiag[row: row + n_orb, coly: coly + n_orb] = block_y

    # Hamiltonian
    H_diag = np.kron(np.eye(n_sites), block_z)
    H_offdiag = H_offdiag + np.conj(H_offdiag.T)
    H = H_diag + H_offdiag

    return H





