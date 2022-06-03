# Transport calculation for a clean TI nanowire

import numpy as np
from numpy import pi
import time
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.linalg import block_diag, expm
from functions import transfer_to_scattering, scat_product, transport_checks, thermal_average, df_FD


#%% Parameters

# Constants and set up of the model
hbar = 1                            # Planck's constant
e = 1.6e-19                         # Electron charge in C
G_q = ((e ** 2) / hbar)             # Conductance quantum
vf = 330                            # Fermi velocity in meV nm
w, h = 120, 20                      # Width and height of the wire in nm
L = 1100                            # Length of the nanowire
P = (2 * w) + (2 * h)               # Perimeter of the wire
E_F = np.linspace(-30, 30, 600)     # Fermi energy
dE = E_F[1] - E_F[0]                # Separation in energies


# Declarations
l_cutoff = 10                                  # Cutoff for the number of angular momentum modes that we consider
modes = np.arange(-l_cutoff, l_cutoff+1)       # Angular momentum modes
n_modes = int(len(modes))                      # Number of l modes
n_s = 2                                        # Spin components
G = np.zeros((len(E_F), ))                     # Conductance vector
L_grid = 1000
dx = L / L_grid


# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]])     # Pauli x
sigma_y = np.array([[0, -1j], [1j, 0]])  # Pauli y
sigma_z = np.array([[1, 0], [0, -1]])    # Pauli z


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
G_an = np.zeros((len(E_F), ))
for n in modes:
    aux = ((2 * pi / P) * (n - 0.5)) ** 2
    aux2 = ((E_F / vf) ** 2) - np.repeat(aux, len(E_F))
    k = np.sqrt(aux2 + 0j)
    Tn = np.abs(k / (k * np.cos(k * L) + 1j * (E_F / vf) * np.sin(k * L))) ** 2
    G_an = G_an + Tn


# Low temperature thermal average of the conductance
T = 10                                                  # Temperature in K
thermal_interval = 5                                   # Energy range above and below mu that we include on the thermal average in meV
sample_points = int(thermal_interval / dE)              # Points included in the integration above and below
E_F_thermal = E_F[sample_points: -sample_points]        # Range over which we calculate the thermal conductance
G_therm = np.zeros((len(E_F_thermal), ))                # Thermal conductance vector


print(len(E_F_thermal))

for i, energy in enumerate(E_F_thermal):
    print(str(i) + "/" + str(len(E_F_thermal)))
    j = i + sample_points                                                       # Index in the complete E_f vector
    integration_interval = E_F[j - sample_points: j + sample_points]            # Energy range for integration
    G_interval = G[j - sample_points: j + sample_points]                        # Conductance range for integration
    G_therm[i] = thermal_average(T, E_F[j], integration_interval, G_interval)   # Thermal averaged conductance

#%% Figures
plt.plot(E_F, G, '.k', markersize=5)
plt.plot(E_F, G_an, 'b', linewidth=1)
plt.plot(E_F_thermal, G_therm, '.r', markersize=2)
plt.plot(E_F, np.repeat(2, len(E_F)), '-.k')
plt.plot(E_F, np.repeat(4, len(E_F)), '-.k')
plt.plot(E_F, np.repeat(6, len(E_F)), '-.k')
plt.plot(E_F, np.repeat(8, len(E_F)), '-.k')
plt.xlim(E_F_thermal[0], E_F_thermal[-1])
plt.ylim(0, 6)
plt.legend(("Numerical", "Analytical", "Thermal Average " + str(T) + "K"))
plt.xlabel("$E_F$ (meV)")
plt.ylabel("$G/G_Q$")
plt.title("$B_\perp =0$" + ", $L=$" + str(L) + ", $w=$" + str(w) + ", $h=$" + str(h))
plt.show()

