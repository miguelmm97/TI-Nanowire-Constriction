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
    # Performs the transport calculation (asuming no change in the transfer matrix with position)
    scat_matrix = scat_matrix0  # Scattering matrix

    for pos in range(L_grid):
        if pos == 0:
            scat_matrix = scat_matrix0
        else:
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


#%% Bi2Se3 Band structure and hybridisation gap
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

def Ham_bulk_FB3dTI(kx, ky, kz, t, lamb, lamb_z, eps, a):
    """
          Calculates the bulk band structure for Bi2Se3 taking the Fu and Berg model

          Parameters
          ----------
          kx, ky, kz: {float} Momentum along different directions
          t, lamb, lamb_z, eps, a: {float} Parameters of the model

          Returns:
          -------
          H: {4x4 np.array} Matrix hamiltonian

          """

    H = (eps - 2 * t * np.cos(kz * a)) * np.kron(tau_x, sigma_0) + lamb_z * np.sin(kz * a) * np.kron(tau_y, sigma_0) + \
        - 2 * t * np.cos(ky * a) * np.kron(tau_x, sigma_0) + lamb * np.sin(ky * a) * np.kron(tau_z, sigma_x) + \
        - 2 * t * np.cos(kx * a) * np.kron(tau_x, sigma_0) - lamb * np.sin(kx * a) * np.kron(tau_z, sigma_y)

    return H

def Ham_bulk_Bi2Se3(kx, ky, kz, C, M, D1, D2, B1, B2, A1, A2, a):
    """
    Calculates the bulk band structure for Bi2Se3 by taking the low energy ab initio theory to the lattice

    Parameters
    ----------
    kx, ky, kz: {float} Momentum along different directions
    M, D1, D2, B1, B2, A1, A2, C: {float} Parameters of the model
    a: {float} Lattice constant

    Returns:
    -------
    H: {4x4 np.array} Matrix hamiltonian

    """

    # Bulk hamiltonian
    H = (C + 2 * (D1 / a ** 2) * (1 - np.cos(kz * a)) + 2 * (D2 / a ** 2) * (
                2 - np.cos(kx * a) - np.cos(ky * a))) * np.kron(tau_0, sigma_0) \
        + (M - 2 * (B1 / a ** 2) * (1 - np.cos(kz * a)) - 2 * (B2 / a ** 2) * (
                2 - np.cos(kx * a) - np.cos(ky * a))) * np.kron(tau_z, sigma_0) \
        + (A1 / a) * np.sin(kz * a) * np.kron(tau_x, sigma_z) \
        + (A2 / a) * np.sin(kx * a) * np.kron(tau_x, sigma_x) \
        + (A2 / a) * np.sin(ky * a) * np.kron(tau_x, sigma_y)

    return H

def Ham_bulk_LowEnergy_Bi2Se3(kx, ky, kz, C, M, D1, D2, B1, B2, A1, A2):
    """
      Calculates the bulk band structure for Bi2Se3 taking the low energy ab initio theory

      Parameters
      ----------
      kx, ky, kz: {float} Momentum along different directions
      M, D1, D2, B1, B2, A1, A2, C: {float} Parameters of the model

      Returns:
      -------
      H: {4x4 np.array} Matrix hamiltonian

      """

    # Bulk hamiltonian
    H = (C + D1 * (kz ** 2) + D2 * (kx ** 2 + ky ** 2)) * np.kron(tau_0, sigma_0) \
        + (M - B1 * (kz ** 2) - B2 * (kx ** 2 + ky ** 2)) * np.kron(tau_z, sigma_0) \
        + A1 * kz * np.kron(tau_x, sigma_z) \
        + A2 * kx * np.kron(tau_x, sigma_x) \
        + A2 * ky * np.kron(tau_x, sigma_y)

    return H

def Ham_ThinFilm_FB3dTI(L_x, x, ky, kz, t, lamb, lamb_z, eps, a, B):
    """
    Calculates the hamiltonian for a thin film along x y and z, with finite dimension x, for the Fu and Berg model,
    in a parallel magnetic field B along the thin film.
    Parameters
    ----------
    L_x: {int} Length on x direction
    x: {np.array} x position of the sites
    ky, kz: {float} Momentum along ky and kz
    t, lamb, lamb_z, eps: {float} Parameters of the model
    a: {float} Lattice constant
    B: {float} Magnetic field

    Returns
    ------
    H: {np.array(n_states, n_states)} Matrix hamiltonian

    """
    # Declarations
    n_states = len(x) * 4                                         # Number of basis states
    transx = ((x + 1) % L_x)                                      # List of neighbours in x direction
    H_offdiag = np.zeros((n_states, n_states), dtype='complex_')  # Hamiltonian for the xy cross-section

    # Block hoppings
    Xhopp = (1j * 0.5 * lamb * np.kron(sigma_z, sigma_y)) - t * np.kron(sigma_x, sigma_0)
    bulk = (eps - 2 * t * np.cos(kz * a)) * np.kron(sigma_x, sigma_0) + lamb_z * np.sin(kz * a) * np.kron(sigma_y, sigma_0)
    bulkB = lamb * np.sin(ky * a) * np.kron(sigma_z, sigma_x)
    bulkB += - 2 * t * np.cos(ky * a) * np.kron(sigma_x, sigma_0)
    peierls = np.exp(- 1j * 2 * pi * B * x * a * a / phi0)

    # Hopping along x
    for site in range(0, L_x):
        row = site * 4
        col = transx[site] * 4
        if (site + 1) % L_x != 0:
            H_offdiag[row: row + 4, col: col + 4] = Xhopp

    # Hamiltonians
    H_diag = np.kron(np.eye(L_x), bulk) + np.kron(np.diag(peierls), bulkB)
    H_offdiag += np.conj(H_offdiag).T
    H= H_diag + H_offdiag

    return H

def Ham_ThinFilm_Bi2Se3(L_z, z, kx, ky, C, M, D1, D2, B1, B2, A1, A2, a, B):
    """
      Calculates the hamiltonian for a thin film along x y and z, with finite dimension z, for the ab initio model,
      in a parallel magnetic field B along the thin film.
      Parameters
      ----------
      L_z: {int} Length on x direction
      z: {np.array} x position of the sites
      kx, ky: {float} Momentum along ky and kz
      C, M, D1, D2, B1, B2, A1, A2: {float} Parameters of the model
      a: {float} Lattice constant
      B: {float} Magnetic field

      Returns
      ------
      H: {np.array(n_states, n_states)} Matrix hamiltonian

      """



    # Definitions
    n_states = L_z * 4                                            # Number of basis states
    transz = (z + 1) % L_z                                        # Translated vector of z
    H_offdiag = np.zeros((n_states, n_states), dtype='complex_')  # Hamiltonian for the xy cross-section
    # peierls = 2 * pi * (z - (L_z - 1)/2) * a * a * B * ams * ams / phi0           # Peierls factor
    peierls = 2 * pi * z * a * a * B * ams * ams / phi0

    # Block hoppings along x, y, z
    block_x = (C + 2 * (D1 / a ** 2) + 2 * (D2 / a ** 2) * (1 - np.cos(kx * a))) * np.kron(tau_0, sigma_0)\
             + (M - 2 * (B1 / a ** 2) - 2 * (B2 / a ** 2) * (1 - np.cos(kx * a))) * np.kron(tau_z, sigma_0)\
             + (A2 / a) * np.sin(kx * a) * np.kron(tau_x, sigma_x)

    block_y = 2 * (D2 / a ** 2) * np.kron(np.diag(1 - np.cos(ky * a - peierls)), np.kron(tau_0, sigma_0)) +\
              - 2 * (B2 / a ** 2) * np.kron(np.diag(1 - np.cos(ky * a - peierls)), np.kron(tau_z, sigma_0)) + \
              + (A2 / a) * np.kron(np.diag(np.sin(ky * a - peierls)), np.kron(tau_x, sigma_y))

    block_z = - (D1 / a ** 2) * np.kron(tau_0, sigma_0) \
              + (B1 / a ** 2) * np.kron(tau_z, sigma_0) \
              - (A1 / a) * (1j / 2) * np.kron(tau_x, sigma_z)


    # Hopping along z
    for site in range(0, L_z):

        # Sites connected by the hamiltonian
        row = site * 4
        col = transz[site] * 4
        # Hopping along z (open boundaries)
        if (site + 1) % L_z != 0:
            H_offdiag[row: row + 4, col: col + 4] = block_z

    # Hamiltonian
    H_diag = np.kron(np.eye(L_z), block_x) + block_y
    H_offdiag = H_offdiag + np.conj(H_offdiag.T)
    H = H_diag + H_offdiag

    return H

def Ham_nw_FB3dTI(n_sites, n_orb, L_x, L_y, x, y, k, t, lamb, lamb_z, eps, a, flux, periodicity_x=False, periodicity_y=False):
    """
        Calculates the hamiltonian for a 3dTI (Fu and Berg model) nanowire bulk z direction, and finite but possibly
        periodic boundaries along x and y.

        Parameters
        ---------
        n_sites: Number of lattice sites
        n_orb:  Number of orbitals
        L_x, L_y: Number of sites on each direction
        x, y: Position of each site
        k: Momentum along y direction
        t, lamb, lamb_z, eps: Parameters of the model
        a: Lattice constant
        flux: Magnetic flux threaded through the cross-section in units of the quantum of flux
        periodicity_x: True/False
        periodicity_y: True/False

        Returns:
        -------
        H: Matrix Hamiltonian

    """

    # Declarations
    n_states = n_sites * n_orb                                      # Number of basis states
    cross_section = (L_x - 1) * (L_y - 1)                           # Area of the xy cross-section
    transx = xtranslation(x, y, L_x, L_y)                           # List of neighbours in x direction
    transy = ytranslation(x, y, L_x, L_y)                           # List of neighbours in y direction
    H_offdiag = np.zeros((n_states, n_states), dtype='complex_')    # Hamiltonian for the xy cross-section

    # Block hoppings along x, y
    block_x = (1j * 0.5 * lamb * np.kron(tau_z, sigma_y)) - t * np.kron(tau_x, sigma_0)
    block_y = (-1j * 0.5 * lamb * np.kron(tau_z, sigma_x)) - t * np.kron(tau_x, sigma_0)
    block_z = (eps - 2 * t * np.cos(k * a)) * np.kron(tau_x, sigma_0) + (lamb_z * np.sin(k * a)) * np.kron(tau_y, sigma_0)
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

def Ham_nw_Bi2Se3(n_sites, n_orb, L_x, L_z, x, z, k, C, M, D1, D2, B1, B2, A1, A2, a, flux, periodicity_x=False, periodicity_z=False):
    """
    Calculates the hamiltonian for a Bi2Se3 nanowire with a bulk y direction, and finite but possibly periodic boundaries
    along x and z.

    Parameters
    ---------
    n_sites: Number of lattice sites
    n_orb:  Number of orbitals
    L_x, L_z: Number of sites on each direction
    x, z: Position of each site
    k: Momentum along y direction
    M, D1, D2, B1, B2, A1, A2, C: Parameters of the model
    a: Lattice constant
    flux: Magnetic flux threaded through the cross-section in units of the quantum of flux
    periodicity_x: True/False
    periodicity_z: True/False

    Returns:
    -------
    H: Matrix Hamiltonian

    """

    # Definitions
    n_states = n_sites * n_orb                                    # Number of basis states
    transx = xtranslation(x, z, L_x, L_z)                         # List of neighbours in x direction
    transy = ytranslation(x, z, L_x, L_z)                         # List of neighbours in y direction
    cross_section = (L_x - 1) * (L_z - 1)                         # Area of the xy cross-section
    H_offdiag = np.zeros((n_states, n_states), dtype='complex_')  # Hamiltonian for the xy cross-section

    # Block hoppings along x, y, z
    block_x = - (D2 / a ** 2) * np.kron(tau_0, sigma_0) \
              + (B2 / a ** 2) * np.kron(tau_z, sigma_0) \
              - (A2 / a) * (1j / 2) * np.kron(tau_x, sigma_x)
    block_y = (C + 2 * ((D1 + D2) / a ** 2) + 2 * (D2 / a ** 2) * (1 - np.cos(k * a))) * np.kron(tau_0, sigma_0) \
              + (M - 2 * ((B1 + B2) / a ** 2) - 2 * (B2 / a ** 2) * (1 - np.cos(k * a))) * np.kron(tau_z, sigma_0) \
              + (A2 / a) * np.sin(k * a) * np.kron(tau_x, sigma_y)
    block_z = - (D1 / a ** 2) * np.kron(tau_0, sigma_0) \
              + (B1 / a ** 2) * np.kron(tau_z, sigma_0) \
              - (A1 / a) * (1j / 2) * np.kron(tau_x, sigma_z)
    peierls = np.exp((2 * pi * 1j / cross_section) * flux * z)

    # Hopping along x and y
    for site in range(0, n_sites):

        # Sites connected by the hamiltonian
        row = site * n_orb
        colx = transx[site] * n_orb
        colz = transy[site] * n_orb

        # Hopping along x
        if periodicity_x:
            H_offdiag[row: row + n_orb, colx: colx + n_orb] = block_x * peierls[site]
        else:
            if (site + 1) % L_x != 0:
                H_offdiag[row: row + n_orb, colx: colx + n_orb] = block_x * peierls[site]

        # Hopping along y
        if periodicity_z:
            H_offdiag[row: row + n_orb, colz: colz + n_orb] = block_z
        else:
            if (site + L_x) < n_sites:
                H_offdiag[row: row + n_orb, colz: colz + n_orb] = block_z

    # Hamiltonian
    H_diag = np.kron(np.eye(n_sites), block_y)
    H_offdiag = H_offdiag + np.conj(H_offdiag.T)
    H = H_diag + H_offdiag

    return H

