# TI Nanowire Constriction
import numpy as np
from numpy import pi
import time
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

from functions import Conductance
# %% Rectangular nanowire with perpendicular magnetic field


# WARNING: In order to plot the spectrum, reduced values of k and n_modes are valid. However for the conductance
# we should see whether cutting the k range or having less modes actually affects the calculation at high
# energies, since we might  be skipping some additional bands at high energy

# Constants and set up of the model
hbar = 1                            # Planck's constant
e = 1.6e-19                         # Electron charge in C
vf = 330                            # Fermi velocity in meV nm
w, h = 150, 15                      # Width and height of the wire in nm
P = (2 * w) + (2 * h)               # Perimeter of the wire
r = w / (w + h)                     # Useful ratio
B = 0                               # Perpendicular magnetic field in T
Vg = np.linspace(-100, 100, 300)    # Gate voltage in mV
V_bias = np.linspace(-25, 25, 100)  # Bias voltage (fermi energy) in mV

# Declarations
k = np.linspace(-10, 10, 5000)                  # Momentum in 1/nm
l_cutoff = 30                                   # Cutoff for the number of angular momentum modes that we consider
modes = np.arange(-l_cutoff, l_cutoff+1)        # Angular momentum modes
n_k = int(len(k))                               # Number of k modes
n_modes = int(len(modes))                       # Number of l modes
n_s = 2                                         # Spin components
energy = np.zeros((int(n_modes * n_s), n_k))    # Energy matrix declaration
off_diag = np.zeros((n_modes, n_modes))         # Off diag auxiliary matrix declaration
conductance = np.zeros((len(V_bias), len(Vg)))  # Conductance matrix declaration

# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]])     # Pauli x
sigma_y = np.array([[0, -1j], [1j, 0]])  # Pauli y


# %% Main code

# Hamiltonian and energy eigenvalues
# Off diagonal elements (coupling between angular momentum modes)
for index1 in range(n_modes):
    for index2 in range(n_modes):
        if (modes[index1] - modes[index2]) % 2 != 0:
            m = modes[index1] - modes[index2]
            off_diag[index1, index2] = 1e16 * hbar * vf * e * B * P * ((-1) ** ((m + 1) / 2)) * \
                                          np.sin(m * pi * r / 2) / (m * m * pi * pi)
            # 1e16 cause T nm = nm/m² so we need 1/1e18 and we need to divide by hbar cause we are pulling a factor
            # hbar over all but the minimal coupling says p --> p+eA, so to pull the hbar factor out we need to write
            # eA/hbar so an added 1e34

aux_l = (hbar * vf * 2 * pi / P) * (modes - (1/2))  # sigma_y component, with different l for each mode
ham_y = np.kron(np.diag(aux_l), sigma_y)            # Proper sigma_y diag matrix in the tensor product basis
ham_offdiag = np.kron(off_diag, sigma_x)            # Proper sigma_x off-diag matrix in the tensor product basis

for index in range(n_k):
    aux_k = ((hbar * vf) * k[index]).repeat(n_modes)    # sigma_x component with same k for each block
    ham_x = np.kron(np.diag(aux_k), sigma_x)            # Proper sigma_x  diag matrix in the tensor product basis
    Hamiltonian = ham_x + ham_y + ham_offdiag           # H(k) in the tensor product basis
    energy[:, index] = np.linalg.eigvalsh(Hamiltonian)  # Eigenenergies E(k)
    idx = energy[:, index].argsort()                    # Ordering the energy bands at k
    energy[:, index] = energy[idx, index]               # Ordered energy bands

# # Conductance (mode-counting approximation)
# for index1 in range(len(V_bias)):
#     print(index1)
#     for index2 in range(len(Vg)):
#         conductance[index1, index2] = Conductance(energy, Vg[index2], V_bias[index1])


# Plots and figures
# # Spectrum E(k)
for index in range(int(n_modes * n_s)):
    plt.plot(k, energy[index, :], ".b", markersize=2)
plt.xlabel("$k(nm^{-1})$")
plt.ylabel("$E(meV)$")
plt.xlim(-0.5, 0.5)
plt.ylim(-60, 60)
plt.title("$B_\perp =$" + str(B))
# plt.legend(string_legend)
plt.show()


# # Conductance as a function of the bias voltage
# plt.plot(V_bias, conductance[:, 0])
# plt.xlabel("$V_{bias}(mV)$")
# plt.ylabel("$G$")
# plt.title("$B_\perp =$" + str(B) + "  $V_{g}=0$")
# plt.show()


# Conductance as a function of the gate voltage
# plt.plot(Vg, conductance[0, :])
# plt.xlabel("$V_{g}(mV)$")
# plt.ylabel("$G$")
# plt.title("$B_\perp =$" + str(B) + "  $V_{bias}=0$")
# plt.show()
#

# Conductance as a function of the bias voltage and the gate voltage
# logG = np.log(conductance)
# im = plt.pcolormesh(Vg, V_bias, conductance)
# cb = plt.colorbar(im)
# cb.set_label("$G$", rotation=90)
# plt.xlabel("$V_g(V)$")
# plt.ylabel("$V_{bias}(V)$")
# plt.title("$B_\perp =$" + str(B))
# plt.show()
#
#









