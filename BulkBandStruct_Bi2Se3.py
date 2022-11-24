import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from functions import spectrum, Ham_nw_Bi2Se3_1, Ham_nw_Bi2Se3_2, xtranslation, ytranslation

# %%  Global definitions

# Parameters of the model
n_orb = 4                                   # Number of orbitals per site
Arms = 0.1                                     # From armstrongs to nm
lamb = 150                                  # meV
eps = 4 * lamb                              # meV
lamb_z = 2 * lamb                           # meV
t = lamb                                    # meV
flux = 0.0                                  # Flux through the cross-section in units of the flux quantum
kz = np.linspace(-0.1, 0.1, 500)            # Momentum space
A1, A2 = 2200 * Arms, 4100 * Arms           # meV
B1, B2, D1, D2 = 10000 * (Arms ** 2), 56600 * (Arms ** 2), 1300 * (Arms ** 2), 19600 * (Arms ** 2)  # meV
C, M = -6.8, 280                            # meV

# Lattice definition
L_x, L_y = 10, 10                           # In units of a (average bond length)
n_sites = int(L_x * L_y)                    # Number of sites in the lattice
n_states = n_sites * n_orb                  # Number of basis states
sites = np.arange(0, L_x * L_y)             # Array with the number of each site
x = sites % L_x                             # x position of the sites
y = sites // L_x                            # y position of the sites

# State that we want to show in the figure
band = 0                                    # Band that we are plotting
momentum = 300                             # Momentum index to plot the wavefunctions

# Declarations
bands = np.zeros((n_states, len(kz)))
eigenstates = np.zeros((n_states, n_states, len(kz)), complex)
prob_density = np.zeros((n_sites, n_states))
transx = xtranslation(x, y, L_x, L_y)
transy = ytranslation(x, y, L_x, L_y)

# %% Diagonalisation

# Band structure
for j, k in enumerate(kz):
    print(str(j) + "/" + str(len(kz)))
    H = Ham_nw_Bi2Se3_2(n_sites, n_orb, L_x, L_y, x, y, k, A1, A2, B1, B2, C, D1, D2, M, flux)
    # H = Ham_nw_Bi2Se3_1(n_sites, n_orb, L_x, L_y, x, y, k, t, lamb, lamb_z, eps, flux, periodicity_x=True)
    bands[:, j], eigenstates[:, :, j] = spectrum(H)

# Probability density
for ind, site in enumerate(sites):
    aux1 = eigenstates[site * n_orb: site * n_orb + n_orb, :, momentum]
    aux2 = np.conj(aux1.T) @ aux1
    prob_density[ind, :] = np.diag(aux2)

# Band gap
gap = bands[int(np.floor(n_states / 2)), int(np.floor(len(kz) / 2))] - bands[int(np.floor(n_states / 2)) - 1, int(np.floor(len(kz) / 2))]


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
ax1.plot(kz[momentum], bands[int(np.floor(n_states / 2)) + band, momentum], '.r', markersize=5)
ax1.plot(kz[momentum], bands[int(np.floor(n_states / 2)) - band - 1, momentum], '.c', markersize=5)

# Axis labels and limits
ax1.set_ylabel("$E$[meV]", fontsize=15)
ax1.set_xlabel("$ka$", fontsize=15)
ax1.set_xlim(-0.1, 0.1)
ax1.set_ylim(-100, 100)

# Probability density upper band
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
    ax2.scatter(x[site], y[site], 1000 * prob_density[site, int(np.floor(n_states / 2)) + band], 'r')

# ax2.set_xlabel("$x$[a]", fontsize=25)
ax2.set_ylabel("$y$[a]", fontsize=15)
ax2.set_xlim(-1, L_x)
ax2.set_ylim(-1, L_y)

# Probability density lower band
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
    ax3.scatter(x[site], y[site], 1000 * prob_density[site, int(np.floor(n_states / 2)) - band - 1], 'c')

ax3.set_xlabel("$x$[a]", fontsize=15)
ax3.set_ylabel("$y$[a]", fontsize=15)
ax3.set_xlim(-1, L_x)
ax3.set_ylim(-1, L_y)

fig.suptitle("Bi$_2$Se$_3$ nw, $L_y=$" + str(L_y) + " nm , $\phi/\phi_0=$ " + str(flux) + ", $E_g=$ " + '{:.1f}\n'.format(gap) + " meV")
title = "Bi2Se3_" + str(L_y) + "_layers" + ".pdf"
# plt.savefig(title, bbox_inches="tight")
plt.show()











