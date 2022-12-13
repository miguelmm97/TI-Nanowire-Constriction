import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from functions import spectrum, Ham_ThinFilm_Bi2Se3, Ham_ThinFilm_FB3dTI

# %%  Global definitions

# Parameters of the model
a = 10                                  # Lattice constant in Å
B = 2                                   # Flux through the cross-section in units of the flux quantum
kx = np.linspace(-0.4, 0.4, 10000)      # kx in 1 / Å
ky = np.linspace(-0.4, 0.4, 10000)      # ky in 1 / Å

# Parameters
A1 = 2.2                                # eV Å
A2 = 4.1                                # eV Å
B1 = 10                                 # eV Å^2
B2 = 56.6                               # eV Å^2
D1 = 1.3                                # eV Å^2
D2 = 19.6                               # eV Å^2
C = - 6.8e-3                            # eV
M = 0.28                                # eV
lamb = 0.15                             # eV
eps = 4 * lamb                          # eV
lamb_z = 2 * lamb                       # eV
t = lamb                                # eV

# Lattice definition
L_z = 8                               # In units of a (average bond length)
n_states = L_z * 4                      # Number of basis states
z = np.arange(0, L_z)                   # Array with the number of each site

# Definitions
E = np.zeros((n_states, len(kx)))
band = 0
momentum = int(np.floor(len(ky) / 2))


# %% Diagonalisation

# Band structure
for j in range(len(kx)):
    print(str(j) + "/" + str(len(kx)))
    H = Ham_ThinFilm_FB3dTI(L_z, z, kx[j], ky[j], t, lamb, lamb_z, eps, a, B)
    # H = Ham_ThinFilm_Bi2Se3(L_z, z, kx[j], ky[j], C, M, D1, D2, B1, B2, A1, A2, a)
    E[:, j] = spectrum(H)[0]

gap = E[int(np.floor(n_states / 2)) + band, momentum] - E[int(np.floor(n_states / 2)) - band - 1, momentum]

# %% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 13, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=15)


fig = plt.figure()
gs = GridSpec(4, 8, figure=fig, wspace=4, hspace=2)
ax1 = fig.add_subplot(gs[:, 0:4])
ax2 = fig.add_subplot(gs[:, 4:8])


for j in range(n_states):
    ax1.plot(np.sqrt(2) * kx, E[j, :], 'b')
ax1.set_ylabel("$E$[eV]", fontsize=15)
ax1.set_xlabel("$k_{\perp}[1/Å]$", fontsize=15)
ax1.set_xlim(-0.4, 0.4)



for j in range(n_states):
    ax2.plot(np.sqrt(2) * kx, E[j, :], '.b', markersize=0.25)
ax2.plot(np.sqrt(2) * kx[momentum], E[int(np.floor(n_states / 2)) + band, momentum], '.r', markersize=10)
ax2.plot(np.sqrt(2) * kx[momentum], E[int(np.floor(n_states / 2)) - band - 1, momentum], '.c', markersize=10)
ax2.set_xlabel("$k_{\perp}[1/Å]$", fontsize=15)
ax2.set_xlim(-0.1, 0.1)
ax2.set_ylim(-0.5, 0.5)
fig.suptitle("Bi$_2$Se$_3$ thin film, $L_z=$" + str(L_z) + " nm"+ ",  $E_g=$ " + '{:.5f}\n'.format(gap) + " eV")
plt.show()


for j in range(n_states):
    plt.plot(np.sqrt(2) * kx, E[j, :], 'b')
plt.ylabel("$E$[eV]", fontsize=15)
plt.xlabel("$k_{\perp}[1/Å]$", fontsize=15)
plt.xlim(-0.4, 0.4)
plt.ylim(-1.5, 1.5)
plt.title("Bi$_2$Se$_3$ (110) thin film")
plt.show()