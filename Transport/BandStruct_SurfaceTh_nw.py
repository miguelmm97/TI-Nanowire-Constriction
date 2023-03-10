# TI Nanowire Constriction
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

# %% Rectangular nanowire with magnetic field (translation invariant)

# Constants and set up of the model
hbar = 1e-34                                    # Planck's constant Js
nm = 1e-9                                       # Conversion from nm to m
e = 1.6e-19                                     # Electron charge in C
phi0 = 2 * pi * hbar / e                        # Quantum of flux
vf = 330                                        # Fermi velocity in meV nm
w, h = 120, 20                                  # Width and height of the wire in nm
P = (2 * w) + (2 * h)                           # Perimeter of the wire
r = w / (w + h)                                 # Useful ratio
B_perp = 0                                    # Perpendicular magnetic field in T
n_flux = 0.5                                      # Number of flux quanta threaded through the wire
B_par = n_flux * phi0 / ((w * h) * nm ** 2)     # Parallel magnetic field in T

# Declarations
k = np.linspace(-10, 10, 5000)                  # Momentum in 1/nm
n_k = int(len(k))                               # Number of k modes
l_cutoff = 30                                   # Cutoff for the number of angular momentum modes that we consider
modes = np.arange(-l_cutoff, l_cutoff+1)        # Angular momentum modes
n_modes = int(len(modes))                       # Number of l modes
n_s = 2                                         # Spin components
energy = np.zeros((int(n_modes * n_s), n_k))    # Energy matrix declaration
off_diag = np.zeros((n_modes, n_modes))         # Off diag auxiliary matrix declaration

# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]])            # Pauli x
sigma_y = np.array([[0, -1j], [1j, 0]])         # Pauli y


# %% Main code

## Hamiltonian (we work on the |k,l,s> basis) and in units of hbar=1

# Off-diag term: e vf < n | A_x | m >
for index1 in range(n_modes):
    for index2 in range(n_modes):
        if (modes[index1] - modes[index2]) % 2 != 0:
            m = modes[index1] - modes[index2]
            off_diag[index1, index2] = (nm ** 2 / hbar) * vf * e * B_perp * P * ((-1) ** ((m + 1) / 2)) * \
                                                                     np.sin(m * pi * r / 2) / (m * m * pi * pi)
ham_offdiag = np.kron(off_diag, sigma_x)    # e vf < n | A_x | m > * sigma_x

# Diag terms
A_theta = 0.5 * (nm ** 2) * e * B_par * w * h / hbar          # A_theta = eBa²/2hbar
aux_l = (vf * 2 * pi / P) * (modes - (1/2) + A_theta)         # hbar vf 2pi/P (n-1/2+eBa²/2hbar)
ham_y = np.kron(np.diag(aux_l), sigma_y)                      # hbar vf 2pi/P (n-1/2) * sigma_y
for index in range(n_k):
    aux_k = (vf * k[index]).repeat(n_modes)                   # hbar vf k
    ham_x = np.kron(np.diag(aux_k), sigma_x)                  # hbar vf k * sigma_x
    Hamiltonian = ham_x  + ham_y + ham_offdiag                 # H(k)

    ## Energy bands
    energy[:, index] = np.linalg.eigvalsh(Hamiltonian)        # E(k)
    idx = energy[:, index].argsort()                          # Ordering the energy bands at k
    energy[:, index] = energy[idx, index]                     # Ordered E(k)


#%% Plots and figures
# E(k)
for index in range(int(n_modes * n_s)):
    plt.plot(k, energy[index, :], "b", markersize=2)
plt.xlabel("$k(nm^{-1})$")
plt.ylabel("$E(meV)$")
plt.xlim(-0.5, 0.5)
plt.ylim(-60, 60)
plt.title("$B_\perp =$" + str(B_perp))
# plt.legend(string_legend)
plt.show()










