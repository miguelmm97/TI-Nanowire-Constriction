"""
Calculation of the band structure of a thin film for the different models of Bi2Se3 defined in the functions file.
Depending on the model the geometry of the fim changes, being finite along x or z.
"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from functions import spectrum, Ham_ThinFilm_Bi2Se3, Ham_ThinFilm_FB3dTI


#%% Global definitions

# Parameters of the model
a = 10                                   # Lattice constant in Å
B = 3                                    # Flux through the cross-section in units of the flux quantum
kx = np.linspace(0, 0.0005, 100000)      # kx in 1 / Å
ky = np.linspace(0, 0.0005, 100000)      # kx in 1 / Å

# Parameters of the hamiltonian
A1 = 2.2                                # eV Å    (ab-initio fit to Bi2Se3)
A2 = 4.1                                # eV Å    (ab-initio fit to Bi2Se3)
B1 = 10                                 # eV Å^2  (ab-initio fit to Bi2Se3)
B2 = 56.6                               # eV Å^2  (ab-initio fit to Bi2Se3)
D1 = 1.3                                # eV Å^2  (ab-initio fit to Bi2Se3)
D2 = 19.6                               # eV Å^2  (ab-initio fit to Bi2Se3)
C = - 6.8e-3                            # eV      (ab-initio fit to Bi2Se3)
M = 0.28                                # eV      (Fu and Berg model)
lamb = 0.15                             # eV      (Fu and Berg model)
eps = 4 * lamb                          # eV      (Fu and Berg model)
lamb_z = 2 * lamb                       # eV      (Fu and Berg model)
t = lamb                                # eV      (Fu and Berg model)

# Lattice definition
L_z = 15                                # In units of a (average bond length)
n_states = L_z * 4                      # Number of basis states
z = np.arange(0, L_z)                   # Array with the number of each site

# Definitions
E = np.zeros((n_states, len(kx)))       # Band structure
band = int(np.floor(n_states / 2))      # Conduction band index

#%% Calculations

# Band structure
for j in range(len(kx)):
    print(str(j) + "/" + str(len(kx)))
    H = Ham_ThinFilm_Bi2Se3(L_z, z, 0, ky[j], C, M, D1, D2, B1, B2, A1, A2, a, B)
    E[:, j] = spectrum(H)[0]

# Gap calculation
Eup = min(E[band, :])
Edown = max(E[band - 1, :])
kup = np.where(E[band, :] == Eup)[0][0]
kdown = np.where(E[band - 1, :] == Edown)[0][0]
gap = Eup - Edown
# %% Figures

# Format
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 13, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=15)
fig = plt.figure()
gs = GridSpec(4, 8, figure=fig, wspace=4, hspace=2)
ax1 = fig.add_subplot(gs[:, 0:4])
ax2 = fig.add_subplot(gs[:, 4:8])

# Full band structure
for j in range(n_states):
    ax1.plot(kx, E[j, :], 'b')
ax1.set_ylabel("$E$[eV]", fontsize=15)
ax1.set_xlabel("$k_{\perp}[1/Å]$", fontsize=15)
ax1.set_xlim(-0.4, 0.4)

# Band structure close to the Dirac points
for j in range(n_states):
    ax2.plot(kx, E[j, :], 'b', markersize=1)
ax2.plot(ky[kup], Eup, '.r', markersize=10)
ax2.plot(ky[kdown], Edown, '.c', markersize=10)
ax2.plot(np.zeros((10, )), np.arange(-1, 9), '-k')
ax2.set_xlabel("$k_{\perp}[1/Å]$", fontsize=15)
ax2.set_xlim(-0.05, 0.05)
ax2.set_ylim(-0.1, 0.1)
fig.suptitle("Bi$_2$Se$_3$ thin film (010), $L_z=$" + str(L_z) + " nm, $B=$" + str(B) + " T, $E_g=$ " + '{:.10f}\n'.format(gap) + " eV")
plt.show()

