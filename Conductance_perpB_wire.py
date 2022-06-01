# Transport calculation for a clean TI nanowire in a perpendicular magnetic field

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
w, h = 150, 30                      # Width and height of the wire in nm
L = 100                             # Length of the nanowire
E_F = np.linspace(-0.1, 100, 200)   # Fermi energy
B = 6                               # Perpendicular magnetic field in T

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
energy = np.empty([n_modes * n_s, n_k], )          # Energy bands

# For aux we use 1e16 cause T nm = nm/m² so we need 1/1e18, and we need to divide by hbar because we are pulling a factor
# hbar over all but the minimal coupling says p --> p+eA, so to pull the hbar factor out we need to write
# eA/hbar so an added 1e34

# Pauli matrices
sigma_0 = np.eye(2)                      # Pauli 0
sigma_x = np.array([[0, 1], [1, 0]])     # Pauli x
sigma_y = np.array([[0, -1j], [1j, 0]])  # Pauli y
sigma_z = np.array([[1, 0], [0, -1]])    # Pauli z


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
    energy[:, i] = np.linalg.eigvalsh(Hamiltonian)  # Eigenenergies E(k)
    idx = energy[:, i].argsort()                    # Ordering the energy bands at k
    energy[:, i] = energy[idx, i]                   # Ordered energy bands


# Transport calculation
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


#%% Figures

fig, (ax1, ax2) = plt.subplots(1, 2)

# Spectrum E(k)
for index in range(int(n_modes * n_s)):
    ax1.plot(k_vec, energy[index, :], ".b", markersize=2)

# Conductance
ax2.plot(E_F, G, '-b', linewidth=1)
for i in range(n_s * n_modes):
    ax2.plot(min(energy[i, :]).repeat(100), np.linspace(0, max(G), 100), '-.k', linewidth=1, alpha=0.3)
plt.plot(E_F, np.repeat(1, len(E_F)), '-.k', linewidth=1, alpha=0.3)
plt.plot(E_F, np.repeat(3, len(E_F)), '-.k', linewidth=1, alpha=0.3)
plt.plot(E_F, np.repeat(5, len(E_F)), '-.k', linewidth=1, alpha=0.3)
plt.plot(E_F, np.repeat(7, len(E_F)), '-.k', linewidth=1, alpha=0.3)

# Axis limits and formatting
ax1.set(xlabel="$k(nm^{-1})$", ylabel="$E(meV)$")
ax2.set(xlabel="$E_F$ (meV)", ylabel="$G/G_Q$")
ax1.set_xlim([-1, 1])
ax1.set_ylim([-100, 100])
ax2.set_xlim([0, E_F[-1]])
ax2.set_ylim([0, 6])
fig.suptitle("$B_\perp =$" + str(B) + ", $L=$" + str(L) + ", $w=$" + str(w) + ", $h=$" + str(h))

plt.show()