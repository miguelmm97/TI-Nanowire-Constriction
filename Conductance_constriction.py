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
E_F = np.linspace(-0.1, 100, 1000)    # Fermi energy
B_par, B_perp = 0, 0                # Parallel and perpendicular magnetic fields in T

# Geometry of the nanostructure (MUST BE SYMMETRIC)
l_lead, l_cone, l_constriction = 100, 100, 100  # Length of the leads, cones, and the constriction (nm)
w1, h1 = 150, 30                   # Width and height of the wire in the leads (nm)
w2, h2 = 20, 30                    # Width and height of the wire in the constriction (nm)
a1 = (w1 + h1) / pi                # Radius at x1
a2 = (w2 + h2) / pi                # Radius at x2
x0 = 0                             # Initial length
x1 = x0 + l_lead                   # Final position first lead
x2 = x1 + l_cone                   # Initial position of the constriction
x3 = x2 + l_constriction           # Final position of the constriction
x4 = x3 + l_cone                   # Start of the final lead
x5 = x4 + l_lead                   # End of the nanostructure
print(x0, x1, x2, x3, x4, x5)
L = x5 - x0                        # Length of the nanowire in nm
slope_w = (w2 - w1) / (x2 - x1)    # Slope of the width
slope_h = (h2 - h1) / (x2 - x1)    # Slope of the height
slope_a = (a2 - a1) / (x2 - x1)    # Slope of the radius
def height(x):
    if x0 <= x < x1 or x4 < x <= x5:
        return h1
    if x1 <= x <= x2:
        return h1 + slope_h * (x - x1)
    if x3 <= x <= x4:
        return h2 - slope_h * (x - x3)
    if x2 < x < x3:
        return h2
def width(x):
    if x0 <= x < x1 or x4 < x <= x5:
        return w1
    if x1 <= x <= x2:
        return w1 + slope_w * (x - x1)
    if x3 <= x <= x4:
        return w2 - slope_w * (x - x3)
    if x2 < x < x3:
        return w2
def radius(x):
    if x0 <= x < x1 or x4 < x <= x5:
        return a1
    if x1 <= x <= x2:
        return a1 + slope_a * (x - x1)
    if x3 <= x <= x4:
        return a2 - slope_a * (x - x3)
    if x2 < x < x3:
        return a2
def slope(x):
    if x0 <= x < x1 or x4 < x <= x5:
        return 0
    if (x1 <= x <= x2) or (x3 <= x <= x4):
        return slope_a
    elif x2 < x < x3:
        return 0

# Declarations
l_cutoff = 50                                      # Cutoff for the number of angular momentum modes that we consider
modes = np.arange(-l_cutoff, l_cutoff+1)           # Angular momentum modes
n_modes = int(len(modes))                          # Number of l modes
n_s = 2                                            # Spin components
n_x = 300                                          # Number of x intervals
L_grid = np.linspace(x0, x5, n_x)                  # Grid for x direction
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


#%% Transport calculation

M_modes1 = np.eye(n_modes)                       # Exponent of the mode transfer matrix for the Ef term
M_modes2 = np.diag(modes - 0.5)                  # Exponent of the mode transfer matrix for the p_y term
M1_aux = (1j / vf) * np.kron(sigma_z, M_modes1)  # Exponent of the transfer matrix for the Ef term (wo Ef)
M2_aux = np.kron(sigma_x, M_modes2)              # Exponent of the transfer matrix for the p_y term (wo 1/a)
M3_aux = aux1 * np.kron(sigma_x, M_modes1)       # Exponent of the transfer matrix for the eA_y term (wo a)
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

    # Propagation of the scattering matrix
    for j, x in enumerate(L_grid):

        M1 = fermi_level * np.sqrt(1 + slope(x) ** 2) * M1_aux  # Exponent of the transfer matrix for the Ef term
        M2 = M2_aux * np.sqrt(1 + slope(x) ** 2) / radius(x)    # Exponent of the transfer matrix for the p_y term
        M3 = M3_aux * radius(x) * np.sqrt(1 + slope(x) ** 2)    # Exponent of the transfer matrix for the eA_x term
        M4 = (-1j / vf) * np.kron(sigma_0, M_offdiag(x))        # Exponent of the transfer matrix for the eA_y term
        M = M1 + M2 + M3 + M4                                   # Exponent of the transfer matrix
        transfer_matrix = expm(M * dx)                          # Transfer matrix

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
plt.ylim([0, max(G)])
plt.title("$B_\perp =$" + str(B_perp) + "$, B =$" + str(B_par) + ", $L_{leads}=$" + str(l_lead) + ", $L_{cones}=$" + str(l_cone) +
          ", $L_{constriction}=$" + str(l_constriction)  +", $w_1=$" + str(w1) +", $h_1=$" + str(h1) + ", $w_2=$" + str(w2) +
          ", $h_2=$" + str(h2) )
plt.show()