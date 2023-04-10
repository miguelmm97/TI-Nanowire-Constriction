# Transport calculation for a clean TI nanowire

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from scipy.linalg import expm
from functions import transfer_to_scattering, transport_checks, thermal_average, finite_voltage_bias
import time

start_time = time.time()
#%% Parameters

# Constants and set up of the model
hbar = 1e-34                                                    # Planck's constant in Js
nm = 1e-9                                                       # Conversion from nm to m
e = 1.6e-19                                                     # Electron charge in C
G_q = ((e ** 2) / hbar)                                         # Conductance quantum
vf = 330                                                        # Fermi velocity in meV nm
w, h = 120, 20                                                  # Width and height of the wire in nm
L = 300                                                         # Length of the nanowire
P = 2 * pi * 150                                           # Perimeter of the wire
E_F = np.linspace(0, 20, 200)                                 # Fermi energy


# Declarations
l_cutoff = 30                                                   # Cutoff for the number of angular momentum modes that we consider
modes = np.arange(-l_cutoff, l_cutoff+1)                        # Angular momentum modes
n_modes = int(len(modes))                                       # Number of l modes
n_s = 2                                                         # Spin components
G = np.zeros((len(E_F), ))                                      # Conductance vector
G_an = np.zeros((len(E_F), ))                                   # Analytical conductance vector
L_grid = 1000                                                   # Number of points in the position grid
dx = L / L_grid                                                 # Discretisation step of the position grid
dE = E_F[1] - E_F[0]                                            # Separation in energies


# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]])                            # Pauli x
sigma_y = np.array([[0, -1j], [1j, 0]])                         # Pauli y
sigma_z = np.array([[1, 0], [0, -1]])                           # Pauli z
sigma_0 = np.eye(2)                                             # Pauli 0


# Thermal average parameters
T = 0.5                                                         # Temperature in K
thermal_interval = 20                                           # Energy range above and below mu included on the thermal average (in meV)
sample_points_therm = int(thermal_interval / dE)                # Points included in the integration above and below
E_F_thermal = E_F[sample_points_therm: -sample_points_therm]    # EF taking into account lost points for doing the average
G_therm = np.zeros((len(E_F_thermal), ))                        # Thermal conductance vector


# Finite voltage bias parameters
eVb, T_bias = 25, 0                                             # Finite voltage bias and temperature in meV and K
thermal_interval_bias = 20                                      # Energy range above and below mu that we include on the thermal average in meV
bias_interval = 0.5 * eVb + thermal_interval_bias               # Combined thermal and bias energy range above and below mu included in the bias calculation
sample_points_bias = int(bias_interval / dE)                    # Points included in the integration above and below
E_F_bias = E_F[sample_points_bias: -sample_points_bias]         # Range over which we calculate the thermal conductance
G_bias = np.zeros((len(E_F_bias), ))                            # Thermal conductance vector



#%% Zero-temperature and zero-bias calculation

## Transfer matrix

# Diag terms
M_modesX = (2 * pi / P) * np.diag(modes - 0.5)                        # 1/R (n-1/2) term
for i, energy in enumerate(E_F):

    print(str(i) + "/" + str(len(E_F)))
    M_modesZ = 1j * (energy / vf) * np.eye(n_modes)                   # i E / vf delta_nm term
    M_tot = np.kron(sigma_z, M_modesZ) + np.kron(sigma_x, M_modesX)   # M = i (E/vf) delta_nm * sigmaZ +
                                                                      #  (1/R)(n-1/2) delta_nm * sigmaX
    ## Transport calculation
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


#%% Low temperature thermal average of the conductance
for i, energy in enumerate(E_F_thermal):
    j = i + sample_points_therm                                                       # Index in the complete E_f vector
    integration_interval = E_F[j - sample_points_therm: j + sample_points_therm]      # Energy range for integration
    G_interval = G[j - sample_points_therm: j + sample_points_therm]                  # Conductance range for integration
    G_therm[i] = thermal_average(T, E_F[j], integration_interval, G_interval)         # Thermal averaged conductance


#%% Finite voltage bias
for i, energy in enumerate(E_F_bias):
    j = i + sample_points_bias                                                            # Index in the complete E_f vector
    mu1, mu2 = E_F[j] + (0.5 * eVb), E_F[j] - (0.5 * eVb)                                 # Chemical potentials
    integration_interval = E_F[j - sample_points_bias: j + sample_points_bias]            # Energy range for integration
    G_interval = G[j - sample_points_bias: j + sample_points_bias]                        # Conductance range for integration
    G_bias[i] = finite_voltage_bias(T_bias, mu1, mu2, integration_interval, G_interval)   # Thermal averaged conductance
    print(G_bias[i])

#%% Figures
plt.plot(E_F, G, '.k', markersize=5)
plt.plot(E_F, G_an, 'b', linewidth=1)
# plt.plot(E_F_thermal, G_therm, '.r', markersize=2)
# plt.plot(E_F_bias, G_bias, '.m', markersize=2)
plt.plot(E_F, np.repeat(2, len(E_F)), '-.k')
plt.plot(E_F, np.repeat(4, len(E_F)), '-.k')
plt.plot(E_F, np.repeat(6, len(E_F)), '-.k')
plt.plot(E_F, np.repeat(8, len(E_F)), '-.k')
plt.xlim(0, 20)
plt.ylim(0, 10)
plt.legend(("Numerical", "Analytical", "Thermal Average " + str(T) + "K")) # , "Vb=" + str(eVb) + " meV"))
plt.xlabel("$E_F$ (meV)")
plt.ylabel("$G/G_Q$")
plt.title("$B_\perp =0$" + ", $L=$" + str(L) + ", $w=$" + str(w) + ", $h=$" + str(h))
plt.show()

end_time = time.time()

print('Time elapsed= ' + str(end_time - start_time) + ' s')