# Function file for the TI constriction project
import numpy as np
from numpy.linalg import inv

def f_FD(E, mu, T):

    # Fermi-Dirac distribution
    k_B = 8.617333262e-8  # [meV/K]
    beta = 1 / (k_B * T)

    if T != 0:
        return 1 / (np.exp(beta * (E - mu)) + 1)
    else:
        return np.heaviside(E - mu, 1)

def df_FD(E, mu, T):

    # Derivative of the Fermi-Dirac distribution
    k_B = 8.617333262e-8  # [meV/K]
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

    R = r1 + t1p @ r2 @ inv(np.eye(n_modes) - r1p @ r2) @ t1      # r
    Rp = r2p + t2 @ r1p @ inv(np.eye(n_modes) - r2 @ r1p) @ t2p   # r'
    T = t2 @ inv(np.eye(n_modes) - r1p @ r2) @ t1                 # t
    Tp = t1p @ inv(np.eye(n_modes) - r2 @ r1p) @ t2p              # t'

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

def thermal_average(T, mu, E, G):

    # Compute the thermal average of a given conductance
    # T: Temperature
    # mu: Chemical potential
    # E: Range of energies where we want to integrate.
    # G: Conductance for that range of energies

    integrand = - G * df_FD(E, mu, T)
    return np.trapz(integrand, E)

def finite_Voltage_bias(T, mu1, mu2, E, G):

    # Compute conductance in a finite Voltage bias
    # T: Temperature
    # mu1, mu2: Chemical potential in the left and right leads
    # E: Range of energies where we want to integrate.
    # G: Conductance for that range of energies

    integrand = G * (f_FD(E, mu1, T) - f_FD(E, mu2, T))
    return np.trapz(integrand, E)

def Conductance(E, Vg, V_bias):
    n_bands = len(E[:, 0])
    conductance = 0.0000000001
    E = E + Vg
    for index in range(n_bands):
        if max(E[index, :]) > V_bias > min(E[index, :]):
            conductance = conductance + 1

    return conductance