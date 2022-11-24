# Transport calculation for a clean TI nanowire

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from scipy.linalg import expm
from functions import transfer_to_scattering, transport_calculation, thermal_average, finite_voltage_bias, transfer_matrix, scat_product


#%% Parameters

# Constants and set up of the model
phi0 = 2 * pi * 1e-34 / 1.6e-19                                 # Quantum of flux
vf = 330                                                        # Fermi velocity in meV nm
w, h = 120, 20                                                  # Width and height of the wire in nm
L = 1100                                                         # Length of the nanowire
E_F = np.linspace(0, 60, 200)                                   # Fermi energy
B_perp = 4                                                      # Perpendicular magnetic field in T
n_flux = 0                                                      # Number of flux quanta threaded through the wire
B_par = n_flux * phi0 / ((w * h) * 1e-9 ** 2)                   # Parallel magnetic field in T


# Declarations
l_cutoff = 30                                                   # Cutoff for the number of angular momentum modes that we consider
modes = np.arange(-l_cutoff, l_cutoff+1)                        # Angular momentum modes
n_modes = len(modes)                                            # NUmber of modes
G = np.zeros((len(E_F), ))                                      # Conductance vector
L_grid = 200                                                    # Number of points in the position grid
dx = L / L_grid                                                 # Discretisation step of the position grid
dE = E_F[1] - E_F[0]                                            # Separation in energies


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

for i, energy in enumerate(E_F):
    print(str(i) + "/" + str(len(E_F)))

    # Transport calculation
    T = transfer_matrix(modes, w, h, (w + h) / pi, 0, dx, energy, vf, B_par=B_par, B_perp=B_perp)
    scat_matrix0 = transfer_to_scattering(T, n_modes)                   # Scattering matrix
    G[i] = transport_calculation(n_modes, T, scat_matrix0, L_grid)      # Conductance / Gq




#%% Low temperature thermal average of the conductance
# for i, energy in enumerate(E_F_thermal):
#     j = i + sample_points_therm                                                       # Index in the complete E_f vector
#     integration_interval = E_F[j - sample_points_therm: j + sample_points_therm]      # Energy range for integration
#     G_interval = G[j - sample_points_therm: j + sample_points_therm]                  # Conductance range for integration
#     G_therm[i] = thermal_average(T, E_F[j], integration_interval, G_interval)         # Thermal averaged conductance
#
#
# #%% Finite voltage bias
# for i, energy in enumerate(E_F_bias):
#     j = i + sample_points_bias                                                       # Index in the complete E_f vector
#     mu1, mu2 = E_F[j] + (0.5 * eVb), E_F[j] - (0.5 * eVb)                            # Chemical potentials
#     integration_interval = E_F[j - sample_points_bias: j + sample_points_bias]       # Energy range for integration
#     G_interval = G[j - sample_points_bias: j + sample_points_bias]                   # Conductance range for integration
#     G_bias[i] = finite_voltage_bias(T_bias, mu1, mu2, integration_interval, G_interval)   # Thermal averaged conductance
#     print(G_bias[i])

#%% Figures
plt.plot(E_F, G, '-b', markersize=5)
# plt.plot(E_F_thermal, G_therm, '.r', markersize=2)
# plt.plot(E_F_bias, G_bias, '.m', markersize=2)
plt.plot(E_F, np.repeat(2, len(E_F)), '-.k')
plt.plot(E_F, np.repeat(4, len(E_F)), '-.k')
plt.plot(E_F, np.repeat(6, len(E_F)), '-.k')
plt.plot(E_F, np.repeat(8, len(E_F)), '-.k')
plt.xlim(0, 60)
plt.ylim(0, 7)
# plt.legend(("Numerical", "Analytical", "Thermal Average " + str(T) + "K")) # , "Vb=" + str(eVb) + " meV"))
plt.xlabel("$E_F$ (meV)")
plt.ylabel("$G/G_Q$")
plt.title("$B_\perp =$" + str(B_perp) + ", $L=$" + str(L) + ", $w=$" + str(w) + ", $h=$" + str(h))
plt.show()

