import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from TransportClass import transport
import time

start_time = time.time()
#%% Parameters

# Constants and set up of the model
phi0 = 2 * pi * 1e-34 / 1.6e-19                                           # Quantum of flux
vf = 330                                                                  # Fermi velocity in meV nm
B_perp = 0                                                                # Perpendicular magnetic field in T
n_flux = 0.5                                                              # Number of flux quanta threaded
B_par = 0.2  # n_flux * phi0 / (120 * 20 * 1e-9 * 1e-9)                          # Parallel magnetic field in T
l_cutoff = 2                                                            # Cutoff number modes
fermi = np.linspace(0, 12, 12)                                            # Fermi level
G = np.zeros(fermi.shape)                                                 # Conductance preallocation
nanostructure = transport(vf, B_perp, B_par)                              # Instance of the transport class

# Geometry
x0 = 0; x1 = 100; x2 = x1 + 594.7; x3 = x2 + 800 - 594.7; x4 = x3 + 594.7; x5 = x4 + 100
r_lead = 156.6; r_cons = r_lead / 2
sigma = 0.01                                                              # Smoothing factor
# nanostructure.add_nw(x0, x1, n_points=100, r=r_lead)                       # Lead 1
nanostructure.add_nc(x1, x2, 250, sigma=sigma, r1=r_lead, r2=r_cons)      # Nanocone 1
# nanostructure.add_nw(x2, x3, n_points=10, r=r_cons)                       # Constriction
# nanostructure.add_nc(x3, x4, 250, sigma=sigma, r1=r_cons, r2=r_lead)      # Nanocone 2
# nanostructure.add_nw(x4, x5, n_points=10, r=r_lead)                       # Lead 2


# Conductance calculation
for i, E in enumerate(fermi):
    start_iter = time.time()
    G[i] = nanostructure.get_Landauer_conductance(l_cutoff, E)
    print('iter: {}/{} | time: {:.3e} s | G: {:.2e}'.format(i, len(fermi), time.time() - start_iter, G[i]))

# Figures
plt.plot(fermi, G, 'k', markersize=5)
plt.plot(fermi, np.repeat(2, len(fermi)), '-.k')
plt.plot(fermi, np.repeat(4, len(fermi)), '-.k')
plt.plot(fermi, np.repeat(6, len(fermi)), '-.k')
plt.plot(fermi, np.repeat(8, len(fermi)), '-.k')
plt.xlim(0, max(fermi))
plt.ylim(0, 7)
plt.xlabel("$E_F$ (meV)")
plt.ylabel("$G/G_Q$")
#plt.title("$B_\perp =$" + str(B_perp) + ", $L=$" + str(xf) + ", $w=$" + str(w) + ", $h=$" + str(h))
plt.show()

print('Time elapsed: {:.2e} s'.format(time.time() - start_time))