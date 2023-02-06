"""
Calculation of the band structure of a nanowire for the different models of Bi2Se3 defined in the functions file.
Depending on the model the geometry of the wire changes, being periodic along y or z.
"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from functions import spectrum, Ham_nw_Bi2Se3, Ham_nw_FB3dTI, xtranslation, ytranslation

#%% Global definitions

# Parameters of the model
n_orb = 4                                   # Number of orbitals per site
flux = 0.7                                  # Flux through the cross-section in units of the flux quantum
ky = np.linspace(-0.4, 0.4, 400)           # ky [1/Å]

# Parameters
A1 = 2.2                                    # eV Å      (ab-initio fit to Bi2Se3)
A2 = 4.1                                    # eV Å      (ab-initio fit to Bi2Se3)
B1 = 10                                     # eV Å^2    (ab-initio fit to Bi2Se3)
B2 = 56.6                                   # eV Å^2    (ab-initio fit to Bi2Se3)
D1 = 1.3                                    # eV Å^2    (ab-initio fit to Bi2Se3)
D2 = 19.6                                   # eV Å^2    (ab-initio fit to Bi2Se3)
C = - 6.8e-3                                # eV        (ab-initio fit to Bi2Se3)
M = 0.28                                    # eV        (Fu and Berg model)
a = 10                                      # Å         (Fu and Berg model)
lamb = 0.15                                 # eV        (Fu and Berg model)
eps = 4 * lamb                              # eV        (Fu and Berg model)
lamb_z = 2 * lamb                           # eV        (Fu and Berg model)
t = lamb                                    # eV        (Fu and Berg model)

# Lattice definition
L_x, L_y = 10, 10                           # In units of a (average bond length)
n_sites = int(L_x * L_y)                    # Number of sites in the lattice
n_states = n_sites * n_orb                  # Number of basis states
sites = np.arange(0, L_x * L_y)             # Array with the number of each site
x = sites % L_x                             # x position of the sites
y = sites // L_x                            # y position of the sites

# State that we want to show in the figure
band = 0                                    # Band that we are plotting
momentum = int(len(ky) / 2) + 5             # Momentum index to plot the wave functions

# Definitions
bands = np.zeros((n_states, len(ky)))
eigenstates = np.zeros((n_states, n_states, len(ky)), complex)
prob_density = np.zeros((n_sites, n_states))
transx = xtranslation(x, y, L_x, L_y)
transy = ytranslation(x, y, L_x, L_y)

# %% Diagonalisation

# Band structure
for j, k in enumerate(ky):
    print(str(j) + "/" + str(len(ky)))
    H = Ham_nw_Bi2Se3(n_sites, n_orb, L_x, L_y, x, y, k, C, M, D1, D2, B1, B2, A1, A2, a, flux, periodicity_x=True)
    # H = Ham_nw_FB3dTI(n_sites, n_orb, L_x, L_y, x, y, k, t, lamb, lamb_z, eps, a, flux, periodicity_y=True)
    bands[:, j], eigenstates[:, :, j] = spectrum(H)

# Probability density
for ind, site in enumerate(sites):
    aux1 = eigenstates[site * n_orb: site * n_orb + n_orb, :, momentum]
    aux2 = np.conj(aux1.T) @ aux1
    prob_density[ind, :] = np.diag(aux2)

# Band gap
gap = bands[int(np.floor(n_states / 2)), int(np.floor(len(ky) / 2))] - bands[int(np.floor(n_states / 2)) - 1, int(np.floor(len(ky) / 2))]


# %% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 13, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=15)

fig = plt.figure()
gs = GridSpec(4, 10, figure=fig, wspace=5, hspace=1)
ax1 = fig.add_subplot(gs[:, 0:4])
ax2 = fig.add_subplot(gs[0:2, 4:10])
ax3 = fig.add_subplot(gs[2:, 4:10])

# Band Structure
for j in range(n_states):
    ax1.plot(ky, bands[j, :], 'b', markersize=0.5)
ax1.plot(ky[momentum], bands[int(np.floor(n_states / 2)) + band, momentum], '.r', markersize=10)
ax1.plot(ky[momentum], bands[int(np.floor(n_states / 2)) - band - 1, momentum], '.c', markersize=10)

# Axis labels and limits
ax1.set_ylabel("$E$[eV]", fontsize=15)
ax1.set_xlabel("$k[1/Å]$", fontsize=15)
ax1.set_xlim(-0.02, 0.02)
ax1.set_ylim(-0.2, 0.2)

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

fig.suptitle("Bi$_2$Se$_3$ NW, $L_z=$" + str(L_y) + " nm , $\phi/\phi_0$ =" + str(flux) + ",  $E_g=$ " + '{:.5f}\n'.format(gap) + " eV")
title = "Bi2Se3_" + str(L_y) + "_layers" + ".pdf"
# plt.savefig(title, bbox_inches="tight")
plt.show()




# Full band structure
fig2 = plt.figure()
for j in range(n_states):
    plt.plot(ky, bands[j, :], 'b', markersize=0.5)
# Axis labels and limits
plt.ylabel("$E$[eV]", fontsize=15)
plt.xlabel("$k[1/Å]$", fontsize=15)
plt.ylim(-1.5, 1.5)
plt.xlim(-0.4, 0.4)
plt.title("Bi$_2$Se$_3$ (001) nanowire")
plt.show()






