import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from functions import spectrum, Ham_nw_Bi2Se3, xtranslation, ytranslation

# %%  Global definitions

# Parameters of the model
n_orb = 4  # Number of orbitals per site
lamb = 150  # meV
eps = 4 * lamb  # meV
lamb_z = 2 * lamb  # meV
t = lamb  # meV
flux = 0.5  # Flux through the cross section in units of the flux quantum
kz = np.linspace(-0.1, 0.1, 1001)  # Momentum space
k_shown = 499  # Momentum index to plot the wavefunctions

# Lattice definition
L_x, L_y = 20, 7  # In units of a (average bond length)
n_sites = int(L_x * L_y)  # Number of sites in the lattice
n_states = n_sites * n_orb  # Number of basis states
sites = np.arange(0, L_x * L_y)  # Array with the number of each site
x = sites % L_x  # x position of the sites
y = sites // L_x  # y position of the sites

# Declarations
bands = np.zeros((n_states, len(kz)))
eigenstates = np.zeros((n_states, n_states, len(kz)), complex)
prob_density = np.zeros((n_sites, n_states))
transx = xtranslation(x, y, L_x, L_y)
transy = ytranslation(x, y, L_x, L_y)

# %% Diagonalisation


for j, k in enumerate(kz):
    print(j)

    # Band structure
    H = Ham_nw_Bi2Se3(n_sites, n_orb, L_x, L_y, x, y, k, t, lamb, lamb_z, eps, flux)
    bands[:, j], eigenstates[:, :, j] = spectrum(H)

# Probability density
for ind, site in enumerate(sites):
    aux1 = eigenstates[site * n_orb: site * n_orb + n_orb, :, k_shown]
    aux2 = np.conj(aux1.T) @ aux1
    prob_density[ind, :] = np.diag(aux2)

# %% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 13, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=15)

fig = plt.figure()
gs = GridSpec(4, 6, figure=fig, wspace=4, hspace=1)
ax1 = fig.add_subplot(gs[:, 0:2])
ax2 = fig.add_subplot(gs[0:2, 2:6])
ax3 = fig.add_subplot(gs[2:, 2:6])

# Band Structure
for j in range(n_states):
    ax1.plot(kz, bands[j, :], '.b', markersize=0.5)
ax1.plot(kz[k_shown], bands[int(np.floor(n_states / 2)), k_shown], '.r', markersize=2)
ax1.plot(kz[k_shown], bands[int(np.floor(n_states / 2)) - 1, k_shown], '.c', markersize=2)

# Axis labels and limits
ax1.set_ylabel("$E$[meV]", fontsize=15)
ax1.set_xlabel("$ka$", fontsize=15)
ax1.set_xlim(-0.1, 0.1)
ax1.set_ylim(-100, 100)

# Probability density
for site in range(n_sites):
    sitex = transx[site]
    sitey = transy[site]

    # Hopping along x
    if (site + 1) % L_x != 0:
        ax2.plot([x[site], x[sitex]], [y[site], y[sitex]], 'tab:gray', linewidth=1, alpha=0.3)

    # Hopping along y
    if (site + L_x) < n_sites:
        ax2.plot([x[site], x[sitey]], [y[site], y[sitey]], 'tab:gray', linewidth=1, alpha=0.3)

    # Probability density
    ax2.scatter(x[site], y[site], 1000 * prob_density[site, int(np.floor(n_states / 2))], 'r')

# ax2.set_xlabel("$x$[a]", fontsize=25)
ax2.set_ylabel("$y$[a]", fontsize=15)
ax2.set_xlim(-1, L_x)
ax2.set_ylim(-1, L_y)

# Probability density
for site in range(n_sites):
    sitex = transx[site]
    sitey = transy[site]

    # Hopping along x
    if (site + 1) % L_x != 0:
        ax3.plot([x[site], x[sitex]], [y[site], y[sitex]], 'tab:gray', linewidth=1, alpha=0.3)

    # Hopping along y
    if (site + L_x) < n_sites:
        ax3.plot([x[site], x[sitey]], [y[site], y[sitey]], 'tab:gray', linewidth=1, alpha=0.3)

    # Probability density
    ax3.scatter(x[site], y[site], 1000 * prob_density[site, int(np.floor(n_states / 2)) - 1], 'c')

ax3.set_xlabel("$x$[a]", fontsize=15)
ax3.set_ylabel("$y$[a]", fontsize=15)
ax3.set_xlim(-1, L_x)
ax3.set_ylim(-1, L_y)

fig.suptitle("Bi$_2$Se$_3$ nanowire for " + str(L_y) + " unit cells, $\phi/\phi_0=$ " + str(flux))
title = "Bi2Se3_" + str(L_y) + "_layers" + ".pdf"
# plt.savefig(title, bbox_inches="tight")
plt.show()











