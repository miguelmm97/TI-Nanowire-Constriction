import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import random
from functions import spectrum, Ham_ThinFilm_Bi2Se3, Ham_ThinFilm_FB3dTI

# %%  Global definitions

# Model
Blist = [0, 1, 3, 5] #, 1, 2, 3] #, 4, 5, 6, 7]
L_z = np.arange(2, 20, 1)
Npoints = 2000
kym = np.linspace(-0.05, 0.0, Npoints)
kyp = np.linspace(0.0, 0.05, Npoints)
ky = np.concatenate((kym, kyp))
ky = [0]

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
gap = np.zeros((len(L_z), len(Blist)))

# %% Diagonalisation

# Band
for b, B in enumerate(Blist):
    for i, lz in enumerate(L_z):
        print("B=" + str(b) + ", Lz=" + str(i) + "/" + str(len(L_z)))

        gap1 = 100                # Auxiliary value to minimise the gap
        n_states = int(4 * lz)    # Number of basis states
        sites = np.arange(0, lz)  # Array with the number of each site
        for j, k in enumerate(ky):
            H = Ham_ThinFilm_Bi2Se3(lz, sites, 0, k, C, M, D1, D2, B1, B2, A1, A2, a, B)
            bands = spectrum(H)[0]
            gap2 = bands[int(np.floor(n_states / 2))] - bands[int(np.floor(n_states / 2)) - 1]
            if gap2 < gap1:
                gap[i, b] = bands[int(np.floor(n_states / 2))] - bands[int(np.floor(n_states / 2)) - 1]
                gap1 = gap2


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
plt.title("Bi2Se3 Thin Film (010)")
for i in range(len(Blist)):
    hex = ["#" + ''.join([random.choice('ABCDEF0123456789') for j in range(6)])]
    ax1.plot(L_z, gap[:, i], '.', color=hex[0], label='$B= $' + str(Blist[i]) + " T")
    ax1.plot(L_z, gap[:, i], color=hex[0])
ax1.legend(loc='upper right', ncol=2)
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

