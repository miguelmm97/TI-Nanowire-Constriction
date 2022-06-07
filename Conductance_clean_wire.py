# Transport calculation for a clean TI nanowire

import numpy as np
from numpy import pi
import time
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.linalg import block_diag, expm
from functions import transfer_to_scattering, scat_product, transport_checks, thermal_average, finite_voltage_bias


#%% Parameters

# Constants and set up of the model
hbar = 1                            # Planck's constant
e = 1.6e-19                         # Electron charge in C
G_q = ((e ** 2) / hbar)             # Conductance quantum
vf = 330                            # Fermi velocity in meV nm
w, h = 120, 20                      # Width and height of the wire in nm
L = 600                             # Length of the nanowire
P = (2 * w) + (2 * h)               # Perimeter of the wire
E_F = np.linspace(-35, 35, 600)     # Fermi energy
dE = E_F[1] - E_F[0]                # Separation in energies


# Declarations
l_cutoff = 10                                  # Cutoff for the number of angular momentum modes that we consider
modes = np.arange(-l_cutoff, l_cutoff+1)       # Angular momentum modes
n_modes = int(len(modes))                      # Number of l modes
n_s = 2                                        # Spin components
G = np.zeros((len(E_F), ))                     # Conductance vector
G_an = np.zeros((len(E_F), ))                  # Analytical conductance vector
L_grid = 1000
dx = L / L_grid


# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]])     # Pauli x
sigma_y = np.array([[0, -1j], [1j, 0]])  # Pauli y
sigma_z = np.array([[1, 0], [0, -1]])    # Pauli z


# Thermal average parameters
T = 10                                                          # Temperature in K
thermal_interval = 10                                           # Energy range above and below mu that we include on the thermal average in meV
sample_points_therm = int(thermal_interval / dE)                # Points included in the integration above and below
E_F_thermal = E_F[sample_points_therm: -sample_points_therm]    # Range over which we calculate the thermal conductance
G_therm = np.zeros((len(E_F_thermal), ))                        # Thermal conductance vector


# Finite voltage bias parameters
eVb = 10                                                  # Finite voltage bias
sample_points_bias = int(eVb / dE)                        # Points included in the integration above and below
E_F_bias = E_F[sample_points_bias: -sample_points_bias]   # Range over which we calculate the thermal conductance
G_bias = np.zeros((len(E_F_bias), ))                      # Thermal conductance vector
#%% Transport calculation

# Zero-temperature and bias calculation
M_spin2 = sigma_x                                # Exponent of the spin transfer matrix
M_modes1 = np.eye(n_modes)                       # Exponent of the mode transfer matrix
M_modes2 = (2 * pi / P) * np.diag(modes - 0.5)   # Exponent of the mode transfer matrix
for i, energy in enumerate(E_F):

    print(str(i) + "/" + str(len(E_F)))
    M_spin1 = 1j * (energy / vf) * sigma_z                            # Exponent of the spin transfer matrix                                # Exponent of the mode transfer matrix
    M_tot = np.kron(M_spin1, M_modes1) + np.kron(M_spin2, M_modes2)   # Exponent of the transfer matrix

    # Quick transport calculation particular for the clean wire
    transfer_matrix = expm(M_tot * L)                                 # Transfer matrix
    scat_matrix = transfer_to_scattering(transfer_matrix, n_modes)    # Scattering matrix

    t = scat_matrix[n_modes:, 0: n_modes]                             # Transmission matrix
    t_dagger = np.conj(t.T)                                           # Conjugate transmission matrix
    G[i] = np.trace(t_dagger @ t)                                     # Conductance


# Analytical expression (see Quantum-limited shot noise in graphene paper)
for n in modes:
    aux = ((2 * pi / P) * (n - 0.5)) ** 2
    aux2 = ((E_F / vf) ** 2) - np.repeat(aux, len(E_F))
    k = np.sqrt(aux2 + 0j)
    Tn = np.abs(k / (k * np.cos(k * L) + 1j * (E_F / vf) * np.sin(k * L))) ** 2
    G_an = G_an + Tn


# Low temperature thermal average of the conductance
for i, energy in enumerate(E_F_thermal):
    print(str(i) + "/" + str(len(E_F_thermal)))
    j = i + sample_points_therm                                                       # Index in the complete E_f vector
    integration_interval = E_F[j - sample_points_therm: j + sample_points_therm]      # Energy range for integration
    G_interval = G[j - sample_points_therm: j + sample_points_therm]                  # Conductance range for integration
    G_therm[i] = thermal_average(T, E_F[j], integration_interval, G_interval)         # Thermal averaged conductance


# Finite voltage bias
for i, energy in enumerate(E_F_bias):
    print(str(i) + "/" + str(len(E_F_bias)))
    j = i + sample_points_bias                                                       # Index in the complete E_f vector
    mu1, mu2 = E_F[j] + (0.5 * eVb), E_F[j] - (0.5 * eVb)                            # Chemical potentials
    integration_interval = E_F[j - sample_points_bias: j + sample_points_bias]       # Energy range for integration
    G_interval = G[j - sample_points_bias: j + sample_points_bias]                   # Conductance range for integration
    G_bias[i] = finite_voltage_bias(0, mu1, mu2, integration_interval, G_interval)   # Thermal averaged conductance
    print(G_bias[i])





#%% Figures
plt.plot(E_F, G, '.k', markersize=5)
plt.plot(E_F, G_an, 'b', linewidth=1)
plt.plot(E_F_thermal, G_therm, '.r', markersize=2)
plt.plot(E_F_bias, G_bias, '.m', markersize=2)
plt.plot(E_F, np.repeat(2, len(E_F)), '-.k')
plt.plot(E_F, np.repeat(4, len(E_F)), '-.k')
plt.plot(E_F, np.repeat(6, len(E_F)), '-.k')
plt.plot(E_F, np.repeat(8, len(E_F)), '-.k')
plt.xlim(E_F_thermal[0], E_F_thermal[-1])
plt.ylim(0, 6)
plt.legend(("Numerical", "Analytical", "Thermal Average " + str(T) + "K", "Vb=" + str(eVb) + " meV"))
plt.xlabel("$E_F$ (meV)")
plt.ylabel("$G/G_Q$")
plt.title("$B_\perp =0$" + ", $L=$" + str(L) + ", $w=$" + str(w) + ", $h=$" + str(h))
plt.show()

