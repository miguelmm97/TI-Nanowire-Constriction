# Transport calculation for a clean TI nanocone

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from functions import transfer_to_scattering, scat_product, transfer_matrix, constriction, transport_calculation


#%% Parameters

# Parameters
vf = 330                                        # Fermi velocity in meV nm
E_F = np.linspace(0, 12, 200)                   # Fermi energy
B_perp = 0.2                                      # Perpendicular magnetic field in T
B_par = 0                                       # Parallel magnetic field in T


# Geometry of the nanostructure
n_x = 1400
sigma = 0.01
L_lead, L_nc, L_cons = 100, 549.7, (800 - 549.7)
r_lead = 156.6
r_cons = r_lead / 2
h_lead = 2 * pi * r_lead / 4
w_lead = h_lead
h_cons = 2 * pi * r_cons / 4
w_cons = h_cons
L_grid, dx, h, w, R, dR = constriction(L_lead, L_nc, L_cons, h_lead, w_lead, h_cons, w_cons, sigma, n_x)

# Definitions
l_cutoff = 30                                      # Cutoff for the number of angular momentum modes that we consider
modes = np.arange(-l_cutoff, l_cutoff+1)           # Angular momentum modes
n_modes = int(len(modes))                          # Number of l modes
G = np.zeros((len(E_F), ))                         # Conductance vector

# Plot of the geometry
plt.plot(L_grid, R, 'r')
plt.plot(L_grid, w, '-b')
plt.plot(L_grid, h, '-m')
plt.xlabel("$x(nm)$")
plt.ylabel("$R(nm)$")
plt.xlim([0, L_grid[-1]])
plt.ylim([0, h_lead])
plt.legend(("$R(x)$", "$w(x)$", "$h(x)$"))
plt.title("Geometry of the nanostructure")
plt.show()
#%% Transport calculation

for i, E in enumerate(E_F):
    print(str(i) + "/" + str(len(E_F)-1))

    # Propagation of the scattering matrix
    for j, x in enumerate(L_grid[:-1]):

        T = transfer_matrix(modes, w[j], h[j], R[j], dR[j], dx, E, vf, B_par=B_par, B_perp=B_perp)  # Transfer matrix

        if j == 0:
            scat_matrix = transfer_to_scattering(T, n_modes)                    # Initial scattering matrix
        else:
            scat_matrix_dx = transfer_to_scattering(T, n_modes)                # Scattering matrix for dx
            scat_matrix = scat_product(scat_matrix, scat_matrix_dx, n_modes)   # Propagating the scattering matrix


    t = scat_matrix[n_modes:, 0: n_modes]      # Transmission matrix
    t_dagger = np.conj(t.T)                    # Conjugate transmission matrix
    G[i] = np.trace(t_dagger @ t)              # Conductance / Gq

#%% Figures

# Conductance
plt.plot(E_F, G, '-b', linewidth=1)
for j in range(10):
    plt.plot(E_F, np.repeat(j, len(E_F)), '-.k', linewidth=1, alpha=1)

plt.xlabel("$E_F$ (meV)")
plt.ylabel("$G/G_Q$")
plt.xlim([0, E_F[-1]])
plt.ylim([0, 7])
plt.title("$B_\perp =$" + str(B_perp) + "$, B =$" + str(B_par) + ", $L=$" + str(L_grid[-1]))
plt.show()