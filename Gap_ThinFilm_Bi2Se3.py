"""
Calculation of the gap dependence with Ly and B in a Bi2Se3 thin film.
"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from numpy import pi
import random
from functions import spectrum, Ham_ThinFilm_Bi2Se3, Ham_ThinFilm_FB3dTI

# %%  Global definitions

# Model
Blist = [3]
L_z = np.arange(2, 20, 1)
Npoints = [10, 100, 1000, 10000, 100000, 1000000]

# Values from the papers
paper_valuesx = np.array([2, 3, 4, 6, 7, 8, 9, 10, 11, 12])
paper_valuesy = np.array([0.1, 0.03, 0.02, 0.0025, 0.0005, 0.00025, 1e-6, 1e-4, 1e-5, 2e-6])
exp_valuesx = np.array([2, 3, 4, 5, 6, 7, 8])
exp_valuesy = np.array([0.252, 0.138, 0.07, 0.041, 0, 0, 0])

# Parameters
A1 = 2.2                                     # eV Å
A2 = 4.1                                     # eV Å
B1 = 10                                      # eV Å^2
B2 = 56.6                                    # eV Å^2
D1 = 1.3                                     # eV Å^2
D2 = 19.6                                    # eV Å^2
C = - 6.8e-3                                 # eV
M = 0.28                                     # eV
a = 10                                       # Å
lamb = 0.15                                  # eV
eps = 4 * lamb                               # eV
lamb_z = 2 * lamb                            # eV
t = lamb                                     # eV

# Definitions
gap = np.zeros((len(L_z), len(Blist), len(Npoints)))
a = 10                                       # Lattice constant in Å
hbar = 1e-34                                 # Planck's constant in Js
nm = 1e-9                                    # Conversion from nm to m
ams = 1e-10                                  # Conversion from Å to m
e = 1.6e-19                                  # Electron charge in C
phi0 = 2 * pi * hbar / e                     # Quantum of flux
# %% Diagonalisation

# Band
for b, B in enumerate(Blist):
    for i, lz in enumerate(L_z):

        z = np.arange(0, lz)  # Array with the number of each site
        n_states = lz * 4     # Number of basis states
        band = int(np.floor(n_states / 2))

        for n, Np in enumerate(Npoints):

            print("B=" + str(B) + ", Lz=" + str(i) + "/" + str(len(L_z)) + "Nk=" + str(Np))
            ky = np.linspace(0, 0.0005, Np)
            E = np.zeros((n_states, len(ky)))

            # Energy dispersion
            for j, k in enumerate(ky):
                H = Ham_ThinFilm_Bi2Se3(lz, z, 0, k, C, M, D1, D2, B1, B2, A1, A2, a, B)
                E[:, j] = spectrum(H)[0]

            # Gap calculation
            Eup = min(E[band, :])
            Edown = max(E[band - 1, :])
            kup = np.where(E[band, :] == Eup)[0][0]
            kdown = np.where(E[band - 1, :] == Edown)[0][0]
            gap[i, b, n] = Eup - Edown  # Gap at the Dirac cone

#%% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 13, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=15)

fig1 = plt.figure()
gs = GridSpec(1, 1, figure=fig1, wspace=1, hspace=1)
ax1 = fig1.add_subplot(gs[:, :])
ax1.set_ylabel("$\Delta$[eV]", fontsize=15)
ax1.set_xlabel("$L$ unit cells", fontsize=15)
# ax1.plot(paper_valuesx, paper_valuesy, 'or')
ax1.set_yscale('log')
# ax1.legend(("$\phi/\phi_0$=0", "$\phi/\phi_0$=" + str(flux[1])))
ax1.set_ylim([1e-10, 1e-1])
plt.title("Bi2Se3 Thin Film (010)")
for i in range(len(Npoints)):
    hex = ["#" + ''.join([random.choice('ABCDEF0123456789') for j in range(6)])]
    hex2 = ["#" + ''.join([random.choice('ABCDEF0123456789') for j in range(6)])]
    ax1.plot(L_z, gap[:, 0, i], 's', color=hex[0], label='$B= $' + str(Blist[0]) + " T, " + '$N_k= $' + str(Npoints[i]))
    ax1.plot(L_z, gap[:, 0, i], color=hex[0])
    # ax1.plot(L_z, gap[:, 1, i], '.', color=hex2[0], label='$B= $' + str(Blist[1]) + " T"+ '$Nk= $' + str(Npoints[i]))
    # ax1.plot(L_z, gap[:, 1, i], color=hex2[0])
ax1.legend(loc='upper right', ncol=1)
plt.show()

# fig2 = plt.figure()
# gs = GridSpec(1, 1, figure=fig2, wspace=1, hspace=1)
# ax2 = fig2.add_subplot(gs[:, :])
# ax2.set_ylabel("$\Delta$[eV]", fontsize=15)
# ax2.set_xlabel("$L$ unit cells", fontsize=15)
# ax2.plot(L_y, gap, '.b')
# # ax2.plot(exp_valuesx, exp_valuesy, 'or')
# ax2.set_yscale('log')
# ax2.legend(("Calculation", "Exp"))
# ax2.set_ylim([1e-6, 1e-1])
# plt.show()

