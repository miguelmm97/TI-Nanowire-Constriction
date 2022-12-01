# Transport calculation for a clean TI nanocone

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from functions import transfer_to_scattering, scat_product, transfer_matrix, constriction, transport_calculation
import time
start_time = time.time()
#%% Parameters

# Parameters
vf = 330                                            # Fermi velocity in meV nm
E_F = np.linspace(0, 12, 1200)                       # Fermi energy
B_perp = 0                                          # Perpendicular magnetic field in T
B_par = 0.2                                         # Parallel magnetic field in T


# Geometry of the nanostructure
n_x = 250                                           # Number of points in the grid
n_leads, n_nc, n_cons = 10, 250, 10                 # Number of points in the grid for each part of the constriction
sigma = 0.01                                        # Smoothing factor
L_lead, L_nc, L_cons = 100, 549.7, (800 - 549.7)    # Lengths of the different parts of the constriction
# h_lead, w_lead = 20, 120                            # Height and width of the leads
# h_cons, w_cons = 20, 120                            # Height and width of the constriction
# r_lead = (w_lead + h_lead) / pi
# r_cons = (w_cons + h_cons) / pi
r_lead = 156.6
r_cons = r_lead / 2
h_lead = 2 * pi * r_lead / 4
w_lead = h_lead
h_cons = 2 * pi * r_cons / 4
w_cons = h_cons
L_grid, dx, h, w, R, dR, x0, x1, x2, x3, x4, x5 = constriction(L_lead, L_nc, L_cons, h_lead, w_lead, h_cons, w_cons, sigma, n_x=n_x,
                                                    sampling='cones', n_leads=n_leads, n_nc=n_nc, n_cons=n_cons)
                                                    # Grid, step, height, width, radius, derivative, changing points
# Definitions
l_cutoff = 30                                       # Cutoff for the number of angular momentum modes that we consider
modes = np.arange(-l_cutoff, l_cutoff+1)            # Angular momentum modes
n_modes = int(len(modes))                           # Number of l modes
G = np.zeros((len(E_F), ))                          # Conductance vector

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

    # Transfer matrices for the leads and the constriction
    T_lead = transfer_matrix(modes, w_lead, h_lead, r_lead, 0, dx[0], E, vf, B_par=B_par, B_perp=B_perp)  # Transfer matrix
    T_cons = transfer_matrix(modes, w_cons, h_cons, r_cons, 0, dx[2], E, vf, B_par=B_par, B_perp=B_perp)  # Transfer matrix

    # Go through the nanostructure
    for j, x in enumerate(L_grid[:-1]):

        # Selection of transfer matrix
        if x0 <= x < x1 or x4 <= x < x5:
            T = T_lead
        elif x2 <= x < x3:
            T = T_cons
        else:
            T = transfer_matrix(modes, w[j], h[j], R[j], dR[j], dx[1], E, vf, B_par=B_par, B_perp=B_perp)  # Transfer matrix

        # Propagation of scattering matrix
        if j == 0:
            scat_matrix = transfer_to_scattering(T, n_modes)                   # Initial scattering matrix
        else:
            scat_matrix_dx = transfer_to_scattering(T, n_modes)                # Scattering matrix for dx
            scat_matrix = scat_product(scat_matrix, scat_matrix_dx, n_modes)   # Propagating the scattering matrix


    t = scat_matrix[n_modes:, 0: n_modes]      # Transmission matrix
    t_dagger = np.conj(t.T)                    # Conjugate transmission matrix
    G[i] = np.trace(t_dagger @ t)              # Conductance / Gq


end_time = time.time()
print("Elapsed time " + str(end_time - start_time))
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