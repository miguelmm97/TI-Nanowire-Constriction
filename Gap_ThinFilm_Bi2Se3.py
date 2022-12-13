import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from functions import spectrum, Ham_ThinFilm_Bi2Se3, Ham_ThinFilm_FB3dTI

# %%  Global definitions

# Model
B = 0

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

# System size
L_z = [5]  # np.arange(2, 10, 1)

# Definitions
gap0 = np.zeros((len(L_z),))
# %% Diagonalisation

# Band structure
for j, lz in enumerate(L_z):
    print(", Lz=" + str(j) + "/" + str(len(L_z)))
    n_states = int(4 * lz)             # Number of basis states
    sites = np.arange(0, lz)           # Array with the number of each site
    H = Ham_ThinFilm_FB3dTI(lz, sites, 0, 0, t, lamb, lamb_z, eps, a, B)
    bands = spectrum(H)[0]
    gap0[j] = bands[int(np.floor(n_states / 2))] - bands[int(np.floor(n_states / 2)) - 1]



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
ax1.set_ylim([1e-5, 1e-1])
plt.title("Bi2Se3 Thin Film")
ax1.plot(L_z, gap0, '.b')
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

