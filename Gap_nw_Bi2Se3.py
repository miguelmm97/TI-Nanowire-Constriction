import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from numpy import pi
from functions import spectrum, Ham_nw_Bi2Se3, Ham_nw_FB3dTI

# %%  Global definitions

# Model
n_orb = 4                                   # Number of orbitals per site
flux = [0.0, 0.7]                           # Flux through the cross-section in units of the flux quantum
ky = np.linspace(-0.4, 0.4, 1000)           # ky [1/Å]

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
L_x = np.arange(15, 50, 5)
L_y = np.arange(2, 10, 1)

# Definitions
gap0 = np.zeros((len(L_x), len(L_y)))
gap_flux = np.zeros((len(L_x), len(L_y)))
# %% Diagonalisation

# Band structure
for i, lx in enumerate(L_x):
    for j, ly in enumerate(L_y):
        print("Lx=" + str(i) + "/" + str(len(L_x)) + ", Ly=" + str(j) + "/" + str(len(L_y)))

        for ind in range(len(flux)):
            n_sites = int(lx * ly)             # Number of sites in the lattice
            n_states = n_sites * n_orb         # Number of basis states
            sites = np.arange(0, lx * ly)      # Array with the number of each site
            x = sites % lx                     # x position of the sites
            y = sites // lx                    # y position of the sites
            H = Ham_nw_Bi2Se3(n_sites, n_orb, L_x, L_y, x, y, 0, C, M, D1, D2, B1, B2, A1, A2, a, flux, periodicity_x=False, periodicity_z=False)
            bands = spectrum(H)[0]

            if ind == 0:
                gap0[i, j] = bands[int(np.floor(n_states / 2))] - bands[int(np.floor(n_states / 2)) - 1]
            else:
                gap_flux[i, j] = bands[int(np.floor(n_states / 2))] - bands[int(np.floor(n_states / 2)) - 1]



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
plt.title("Bi2Se3 NW, $L_x=$" + str(L_x))
colour = ['r', 'b', 'g', 'm', 'k', 'tab:orange', 'tab:purple', 'c','y']
for i in range(len(L_x)):
    ax1.plot(L_y, gap0[i, :], marker='.', c=colour[i], label="$L_x=$" + str(L_x[i]), )
    ax1.plot(L_y, gap_flux[i, :], marker='^', c=colour[i])
ax1.legend(loc='lower left', ncol=3)
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

