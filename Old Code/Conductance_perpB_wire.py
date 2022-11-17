# Transport calculation for a clean TI nanowire in a perpendicular magnetic field

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.linalg import expm
from functions import transfer_to_scattering, scat_product, transport_checks, thermal_average, finite_voltage_bias


#%% Parameters

# Constants and set up of the model
hbar = 1                            # Planck's constant
e = 1.6e-19                         # Electron charge in C
G_q = ((e ** 2) / hbar)             # Conductance quantum
vf = 330                            # Fermi velocity in meV nm
w, h = 150, 15                       # Width and height of the wire in nm
L = 100                             # Length of the nanowire
E_F = np.linspace(-120, 120, 1500)  # Fermi energy
dE = E_F[1] - E_F[0]                # Separation in energies
B = 3                               # Perpendicular magnetic field in T

# Declarations
k_vec = np.linspace(-10, 10, 5000)                 # Momentum in 1/nm
n_k = int(len(k_vec))                              # Momentum grid size
l_cutoff = 30                                      # Cutoff for the number of angular momentum modes that we consider
modes = np.arange(-l_cutoff, l_cutoff+1)           # Angular momentum modes
n_modes = int(len(modes))                          # Number of l modes
n_s = 2                                            # Spin components
L_grid = 300                                       # Grid for x direction
dx = L / L_grid                                    # Transfer step
P = (2 * w) + (2 * h)                              # Perimeter of the wire
r = w / (w + h)                                    # Useful parameter to define the vector potential
aux = 1e16 * hbar * vf * e * B * P                 # Auxiliary amplitude for the vector potential
G = np.empty([len(E_F), ])                         # Conductance vector
M_offdiag = np.zeros([n_modes, n_modes], )         # Mode mixing matrix for the vector potential
energy_bands = np.empty([n_modes * n_s, n_k], )          # Energy bands

# For aux we use 1e16 cause T nm = nm/m² so we need 1/1e18, and we need to divide by hbar because we are pulling a factor
# hbar over all but the minimal coupling says p --> p+eA, so to pull the hbar factor out we need to write
# eA/hbar so an added 1e34

# Pauli matrices
sigma_0 = np.eye(2)                      # Pauli 0
sigma_x = np.array([[0, 1], [1, 0]])     # Pauli x
sigma_y = np.array([[0, -1j], [1j, 0]])  # Pauli y
sigma_z = np.array([[1, 0], [0, -1]])    # Pauli z


# Thermal average parameters
T = 30                                                          # Temperature in K
thermal_interval = 20                                           # Energy range above and below mu that we include on the thermal average in meV
sample_points_therm = int(thermal_interval / dE)                # Points included in the integration above and below
E_F_thermal = E_F[sample_points_therm: -sample_points_therm]    # Range over which we calculate the thermal conductance
G_therm = np.zeros((len(E_F_thermal), ))                        # Thermal conductance vector


# Finite voltage bias parameters
eVb,T_bias = 25, 0                                        # Finite voltage bias and temperature in meV and K
thermal_interval_bias = 20                                # Energy range above and below mu that we include on the thermal average in meV
bias_interval = 0.5 * eVb + thermal_interval_bias         # Combined thermal and bias energy range above and below mu included in the bias calculation
sample_points_bias = int(bias_interval / dE)              # Points included in the integration above and below
E_F_bias = E_F[sample_points_bias: -sample_points_bias]   # Range over which we calculate the thermal conductance
G_bias = np.zeros((len(E_F_bias), ))                      # Thermal conductance vector
print(bias_interval, sample_points_bias)
#%% Transfer matrix calculation

# Mode mixing matrix for eA
for i, n1 in enumerate(modes):
    for j, n2 in enumerate(modes):

        if (n1 - n2) % 2 != 0:
            m = n1 - n2
            M_offdiag[i, j] = aux * ((-1) ** ((m + 1) / 2)) * np.sin(m * pi * r / 2) / (m * m * pi * pi)


# Band structure calculation
aux_l = (hbar * vf * 2 * pi / P) * (modes - (1/2))   # sigma_y component, with different l for each mode
ham_y = np.kron(np.diag(aux_l), sigma_y)             # Proper sigma_y diag matrix in the tensor product basis
ham_offdiag = np.kron(M_offdiag, sigma_x)            # Proper sigma_x off-diag matrix in the tensor product basis
for i, k in enumerate(k_vec):

    aux_k = ((hbar * vf) * k).repeat(n_modes)       # sigma_x component with same k for each block
    ham_x = np.kron(np.diag(aux_k), sigma_x)        # Proper sigma_x  diag matrix in the tensor product basis
    Hamiltonian = ham_x + ham_y + ham_offdiag       # H(k) in the tensor product basis
    energy_bands[:, i] = np.linalg.eigvalsh(Hamiltonian)  # Eigenenergies E(k)
    idx = energy_bands[:, i].argsort()                    # Ordering the energy bands at k
    energy_bands[:, i] = energy_bands[idx, i]                   # Ordered energy bands


#%% Transport calculation
M_spin2 = sigma_x                                # Exponent of the spin transfer matrix
M_modes1 = np.eye(n_modes)                       # Exponent of the mode transfer matrix
M_modes2 = (2 * pi / P) * np.diag(modes - 0.5)   # Exponent of the mode transfer matrix
for i, fermi_level in enumerate(E_F):

    print(str(i) + "/" + str(len(E_F)-1))
    M_spin1 = 1j * (fermi_level / vf) * sigma_z                          # Exponent of the spin transfer matrix
    M_diag = np.kron(M_spin1, M_modes1) + np.kron(M_spin2, M_modes2)     # Exponent of the diagonal transfer matrix
    M = M_diag - 1j * np.kron(sigma_0, M_offdiag / vf)                   # Exponent of the transfer matrix

    transfer_matrix = expm(M * dx)                                       # Transfer matrix
    scat_matrix0 = transfer_to_scattering(transfer_matrix, n_modes)      # Scattering matrix
    scat_matrix = scat_matrix0                                           # Scattering matrix

    for pos in range(L_grid):
        scat_matrix = scat_product(scat_matrix, scat_matrix0, n_modes)   # Propagating the scattering matrix

    t = scat_matrix[n_modes:, 0: n_modes]                                # Transmission matrix
    t_dagger = np.conj(t.T)                                              # Conjugate transmission matrix
    G[i] = np.trace(t_dagger @ t)                                        # Conductance / Gq


#%% Low temperature thermal average of the conductance
for i, energy in enumerate(E_F_thermal):
    print(str(i) + "/" + str(len(E_F_thermal)))
    j = i + sample_points_therm                                                       # Index in the complete E_f vector
    integration_interval = E_F[j - sample_points_therm: j + sample_points_therm]      # Energy range for integration
    G_interval = G[j - sample_points_therm: j + sample_points_therm]                  # Conductance range for integration
    G_therm[i] = thermal_average(T, E_F[j], integration_interval, G_interval)         # Thermal averaged conductance


#%% Finite voltage bias
for i, energy in enumerate(E_F_bias):
    print(str(i) + "/" + str(len(E_F_bias)))
    j = i + sample_points_bias                                                       # Index in the complete E_f vector
    mu1, mu2 = E_F[j] + (2 * eVb), E_F[j] - (2 * eVb)                            # Chemical potentials
    integration_interval = E_F[j - sample_points_bias: j + sample_points_bias]       # Energy range for integration
    G_interval = G[j - sample_points_bias: j + sample_points_bias]                   # Conductance range for integration
    G_bias[i] = finite_voltage_bias(T_bias, mu1, mu2, integration_interval, G_interval)   # Thermal averaged conductance


#%% Figures

fig, (ax1, ax2) = plt.subplots(1, 2)
# Spectrum E(k)
for index in range(int(n_modes * n_s)):
    ax1.plot(k_vec, energy_bands[index, :], ".b", markersize=2)

# Conductance
ax2.plot(E_F, G, '-b', linewidth=1)
ax2.plot(E_F_thermal, G_therm, '.r', markersize=2)
ax2.plot(E_F_bias, G_bias, '.m', markersize=2)
for i in range(n_s * n_modes):
    ax2.plot(min(energy_bands[i, :]).repeat(100), np.linspace(0, max(G), 100), '-.k', linewidth=1, alpha=0.3)
    ax2.plot(-min(energy_bands[i, :]).repeat(100), np.linspace(0, max(G), 100), '-.k', linewidth=1, alpha=0.3)
plt.plot(E_F, np.repeat(1, len(E_F)), '-.k', linewidth=1, alpha=0.3)
plt.plot(E_F, np.repeat(3, len(E_F)), '-.k', linewidth=1, alpha=0.3)
plt.plot(E_F, np.repeat(5, len(E_F)), '-.k', linewidth=1, alpha=0.3)
plt.plot(E_F, np.repeat(7, len(E_F)), '-.k', linewidth=1, alpha=0.3)
plt.legend(("T=0K", "Thermal Average " + str(T) + "K")) # , "Vb=" + str(eVb) + " meV"))

# Axis limits and formatting
ax1.set(xlabel="$k(nm^{-1})$", ylabel="$E(meV)$")
ax2.set(xlabel="$E_F$ (meV)", ylabel="$G/G_Q$")
ax1.set_xlim([-1, 1])
ax1.set_ylim([-100, 100])
ax2.set_xlim([-100, 100])
ax2.set_ylim([0, max(G)])
fig.suptitle("$B_\perp =$" + str(B) + ", $L=$" + str(L) + ", $w=$" + str(w) + ", $h=$" + str(h))

plt.show()