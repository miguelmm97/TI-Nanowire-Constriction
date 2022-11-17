# Transport calculation for a clean TI nanocone

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from functions import transfer_to_scattering, scat_product,transfer_matrix


#%% Parameters

# Constants
hbar = 1e-34                # Planck's constant in Js
nm = 1e-9                   # Conversion from nm to m
e = 1.6e-19                 # Electron charge in C
phi0 = 2 * pi * hbar / e    # Quantum of flux

# Parameters
vf = 330                                        # Fermi velocity in meV nm
E_F = np.linspace(-0.01, 0.01, 100)             # Fermi energy
B_perp = 0                                      # Perpendicular magnetic field in T
B_par = 0                                       # Parallel magnetic field in T


# Geometry of the nanostructure
def h(x):
    return
def w(x):
    return
def R(x):
    return
def dR(x):
    return


# Definitions
l_cutoff = 40                                      # Cutoff for the number of angular momentum modes that we consider
modes = np.arange(-l_cutoff, l_cutoff+1)           # Angular momentum modes
n_modes = int(len(modes))                          # Number of l modes
n_x = 100                                          # Number of x intervals
L_grid = np.linspace(x1, x2, n_x)                  # Grid for x direction
dx = L / n_x                                       # Transfer step
G = np.zeros((len(E_F), ))                         # Conductance vector


# Plot of the geometry
plt.plot(L_grid, radius(L_grid), 'r')
plt.plot(L_grid, width(L_grid), '.-b')
plt.plot(L_grid, width(L_grid), '--m')
plt.xlabel("$x(nm)$")
plt.ylabel("$(nm)$")
plt.xlim([x1, x2])
plt.legend(("$a(x)$", "$w(x)$", "$h(x)$"))
plt.title("Geometry of the nanostructure")
plt.show()

#%% Transport calculation

for i, E in enumerate(E_F):
    print(str(i) + "/" + str(len(E_F)-1))

    # Propagation of the scattering matrix
    for j, x in enumerate(L_grid):

        transfer_matrix = transfer_matrix(modes, w(x), h(x), R(x), dR(x), dx, E, vf, B_par=B_par, B_perp=B_perp)  # Transfer matrix

        if j == 0:
            scat_matrix = transfer_to_scattering(transfer_matrix, n_modes)     # Initial scattering matrix
        else:
            scat_matrix_dx = transfer_to_scattering(transfer_matrix, n_modes)  # Scattering matrix for dx
            scat_matrix = scat_product(scat_matrix, scat_matrix_dx, n_modes)   # Propagating the scattering matrix


    t = scat_matrix[n_modes:, 0: n_modes]      # Transmission matrix
    t_dagger = np.conj(t.T)                    # Conjugate transmission matrix
    G[i] = np.trace(t_dagger @ t)              # Conductance / Gq

#%% Figures

# Conductance
plt.plot(E_F, G, '-b', linewidth=1)
plt.plot(E_F, np.repeat(1, len(E_F)), '-.k', linewidth=1, alpha=0.3)
plt.plot(E_F, np.repeat(3, len(E_F)), '-.k', linewidth=1, alpha=0.3)
plt.plot(E_F, np.repeat(5, len(E_F)), '-.k', linewidth=1, alpha=0.3)
plt.plot(E_F, np.repeat(7, len(E_F)), '-.k', linewidth=1, alpha=0.3)

plt.xlabel("$E_F$ (meV)")
plt.ylabel("$G/G_Q$")
plt.xlim([0, E_F[-1]])
plt.ylim([0, 6])
plt.title("$B_\perp =$" + str(B_perp) + "$B =$" + str(B_par) + ", $L=$" + str(L))
plt.show()