# Transport calculation for a clean TI nanocone

import numpy as np
from numpy import pi
import time
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.linalg import block_diag, expm
from functions import transfer_to_scattering, scat_product, transport_checks


#%% Parameters

# Constants and set up of the model
hbar = 1                            # Planck's constant
e = 1.6e-19                         # Electron charge in C
G_q = ((e ** 2) / hbar)             # Conductance quantum
vf = 330                            # Fermi velocity in meV nm
E_F = np.linspace(-0.01, 0.01, 100)    # Fermi energy
B_par, B_perp = 2, 0                # Parallel and perpendicular magnetic fields in T

# Geometry of the nanostructure
C1, C2 = 492, 984
w1, h1 = C1/4, C1/4                # Width and height of the wire in nm at x1
w2, h2 = C2/4, C2/4                # Width and height of the wire in nm at x2
a1 = (w1 + h1) / pi                # Radius at x1
a2 = (w2 + h2) / pi                # Radius at x2
L = np.sqrt(600 ** 2 - (a2-a1) ** 2)   # Length of the nanowire in nm
x1, x2 = 0, L                      # Initial and final points of the geometry in nm
slope_w = (w2 - w1) / (x2 - x1)    # Slope of the width
slope_h = (h2 - h1) / (x2 - x1)    # Slope of the height
slope_a = (a2 - a1) / (x2 - x1)    # Slope of the radius
def height(x):
    return h1 + slope_h * (x - x1)
def width(x):
    return w1 + slope_w * (x - x1)
def radius(x):
    return a1 + slope_a * (x - x1)

# Declarations
l_cutoff = 40                                       # Cutoff for the number of angular momentum modes that we consider
modes = np.arange(-l_cutoff, l_cutoff+1)           # Angular momentum modes
n_modes = int(len(modes))                          # Number of l modes
n_s = 2                                            # Spin components
n_x = 100                                          # Number of x intervals
L_grid = np.linspace(x1, x2, n_x)                  # Grid for x direction
dx = L / n_x                                       # Transfer step
aux1 = 0.5 * 1e16 * e * B_par                      # Auxiliary amplitude for the parallel vector potential
aux2 = 1e16 * e * B_perp                           # Auxiliary amplitude for the perpendicular vector potential
G = np.zeros((len(E_F), ))                         # Conductance vector

# We use 1e16 cause T nm = nm/m² so we need 1/1e18, and we need to divide by hbar because we are pulling a factor
# hbar over all but the minimal coupling says p --> p+eA, so to pull the hbar factor out we need to write
# eA/hbar so an added 1e34.

# Pauli matrices
sigma_0 = np.eye(2)                      # Pauli 0
sigma_x = np.array([[0, 1], [1, 0]])     # Pauli x
sigma_y = np.array([[0, -1j], [1j, 0]])  # Pauli y
sigma_z = np.array([[1, 0], [0, -1]])    # Pauli z

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

factor1 = 1j * np.sqrt(1 + slope_a ** 2) / vf    # Coefficient of the exponent of the transfer matrix for the Ef term
factor2 = np.sqrt(1 + slope_a ** 2)              # Coefficient of the exponent of the transfer matrix for the p_y term
factor3 = np.sqrt(1 + slope_a ** 2) * aux1       # Coefficient of the exponent of the transfer matrix for the eA_y term
factor4 = -1j / vf                               # Coefficient of the exponent of the transfer matrix for the eA_x term
M_modes1 = np.eye(n_modes)                       # Exponent of the mode transfer matrix for the Ef term
M_modes2 = np.diag(modes - 0.5)                  # Exponent of the mode transfer matrix for the p_y term
M1_aux = factor1 * np.kron(sigma_z, M_modes1)    # Exponent of the transfer matrix for the Ef term (wo Ef)
M2_aux = factor2 * np.kron(sigma_x, M_modes2)    # Exponent of the transfer matrix for the p_y term (wo 1/a)
M3_aux = factor3 * np.kron(sigma_x, M_modes1)    # Exponent of the transfer matrix for the eA_y term (wo a)
# Mode mixing matrix for eA_x
def M_offdiag(x):

    M_eA = np.zeros([n_modes, n_modes], )  # Mode mixing matrix for the vector potential
    r = width(x) / (width(x) + height(x))  # Aspect ratio at point x in the geometry
    P = 2 * (width(x) + height(x))         # Perimeter at point x in the geometry

    for i, n1 in enumerate(modes):
        for j, n2 in enumerate(modes):

            if (n1 - n2) % 2 != 0:
                m = n1 - n2
                M_eA[i, j] = aux2 * P * vf * ((-1) ** ((m + 1) / 2)) * np.sin(m * pi * r / 2) / (m * m * pi * pi)

    return M_eA


# Sweep the Fermi level
for i, fermi_level in enumerate(E_F):

    print(str(i) + "/" + str(len(E_F)-1))
    M1 = fermi_level * M1_aux                           # Exponent of the transfer matrix for the Ef term

    # Propagation of the scattering matrix
    for j, x in enumerate(L_grid):

        M2 = M2_aux / radius(x)                         # Exponent of the transfer matrix for the p_y term
        M3 = M3_aux * radius(x)                         # Exponent of the transfer matrix for the eA_x term
        M4 = factor4 * np.kron(sigma_0, M_offdiag(x))   # Exponent of the transfer matrix for the eA_y term
        M = M1 + M2 + M3 + M4                           # Exponent of the transfer matrix
        transfer_matrix = expm(M * dx)                  # Transfer matrix

        if j == 0:
            scat_matrix = transfer_to_scattering(transfer_matrix, n_modes)     # Initial scattering matrix
        else:
            scat_matrix_dx = transfer_to_scattering(transfer_matrix, n_modes)  # Scattering matrix for dx
            scat_matrix = scat_product(scat_matrix, scat_matrix_dx, n_modes)   # Propagating the scattering matrix


    t = scat_matrix[n_modes:, 0: n_modes]      # Transmission matrix
    t_dagger = np.conj(t.T)                    # Conjugate transmission matrix
    G[i] = np.trace(t_dagger @ t)              # Conductance / Gq
    print(G[i])


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