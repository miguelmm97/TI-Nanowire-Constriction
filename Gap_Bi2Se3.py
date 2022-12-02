import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from numpy import pi
from functions import spectrum, Ham_nw_Bi2Se3, Ham_ThinFilm_Bi2Se3_bulkY,  xtranslation, ytranslation

# %%  Global definitions

# Parameters of the model
n_orb = 4                                   # Number of orbitals per site
lamb = 0.15                                 # eV
eps = 4 * lamb                              # eV
lamb_z = 2 * lamb                           # eV
t = lamb                                    # eV
flux = 0.0                                  # Flux through the cross-section in units of the flux quantum
ky = np.linspace(-0.4, 0.4, 1000)           # ky [1/Å]
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

# Lattice definition
L_x= 10
L_y = np.arange(2, 14, 1)
# State that we want to show in the figure
band = 0                                    # Band that we are plotting
momentum = int(len(ky) / 2) + 5             # Momentum index to plot the wavefunctions

# Declarations
gap = np.zeros((len(L_y), ))
# %% Diagonalisation

# Band structure
for i, l in enumerate(L_y):

    print(i)

    n_sites = int(L_x * l)  # Number of sites in the lattice
    n_states = n_sites * n_orb  # Number of basis states
    sites = np.arange(0, L_x * l)  # Array with the number of each site
    x = sites % L_x  # x position of the sites
    y = sites // L_x  # y position of the sites
    H = Ham_ThinFilm_Bi2Se3_bulkY(n_sites, n_orb, L_x, l, x, y, 0, C, M, D1, D2, B1, B2, A1, A2, a, periodicity_x=True)
    bands= spectrum(H)[0]
    gap[i] = bands[int(np.floor(n_states / 2))] - bands[int(np.floor(n_states / 2)) - 1]


font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 13, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=15)


fig1 = plt.figure()
gs = GridSpec(1, 1, figure=fig1, wspace=1, hspace=1)
ax1 = fig1.add_subplot(gs[:, :])
ax1.set_ylabel("$\Delta$[eV]", fontsize=15)
ax1.set_xlabel("$L$ unit cells", fontsize=15)
ax1.plot(L_y, gap, '.b')
ax1.plot(paper_valuesx, paper_valuesy, 'or')
ax1.set_yscale('log')
ax1.legend(("Calculation", "Paper"))
plt.show()

fig2 = plt.figure()
gs = GridSpec(1, 1, figure=fig2, wspace=1, hspace=1)
ax2 = fig2.add_subplot(gs[:, :])
ax2.set_ylabel("$\Delta$[eV]", fontsize=15)
ax2.set_xlabel("$L$ unit cells", fontsize=15)
ax2.plot(L_y, gap, '.b')
ax2.plot(exp_valuesx, exp_valuesy, 'or')
ax2.set_yscale('log')
ax2.legend(("Calculation", "Exp"))
plt.show()

