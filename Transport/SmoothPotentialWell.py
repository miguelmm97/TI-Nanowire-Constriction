import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from TransportClass import transport, geom_cons
import time

start_time = time.time()
#%% Parameters

# Constants and set up of the model
phi0      = 2 * pi * 1e-34 / 1.6e-19                                 # Quantum of flux
vf        = 330                                                      # Fermi velocity in meV nm
B_perp    = 0                                                        # Perpendicular magnetic field in T
n_flux    = 0.0                                                      # Number of flux quanta threaded
B_par     = 0  # n_flux * phi0 / (120 * 20 * 1e-9 * 1e-9)            # Parallel magnetic field in T
l_cutoff  = 30                                                       # Cutoff number modes
fermi     = np.linspace(-100, 100, 1000)
G = np.zeros(fermi.shape)
model = transport(vf, B_perp, B_par, l_cutoff)

# Geometry parameters
r = 25
Vmax = 50; Vmin = 0
Lleads = 20; Lwell = 5; Ltransition = 5
L = 2 * Lleads + 2 * Ltransition + Lwell
Npoints = 20; dx = L / Npoints
smoothing_factor = 0.1


#%% Transport calculation

# Discretisation of the smooth potential
Xwell0 = Lleads; Xwell1 = Xwell0 + Ltransition; Xwell2 = Xwell1 + Lwell; Xwell3 = Xwell2 + Ltransition
x = np.linspace(0, Xwell3 + Lleads, Npoints)
V = geom_cons(x, Xwell0, Xwell1, Xwell2, Vmax, Vmin, smoothing_factor)
r = np.repeat(r, x.shape[0])
model.build_geometry(r, x, V)
for i, E in enumerate(fermi):
    start_iter = time.time()
    G[i] = model.get_Landauer_conductance(E)
    print('iter: {}/{} | time: {:.3e} s | G: {:.2e}'.format(i, G.shape[0], time.time() - start_iter, G[i]))

fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(x, V, color='#FF7256')
ax1.set_xlim(x[0], x[-1])
ax1.set_ylim(Vmin - 10, Vmax + 10)
ax1.set_xlabel("$L$ [nm]")
ax1.set_ylabel("$V$ [meV]")
plt.plot()

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.plot(fermi, G, color='#FF7256')
ax2.set_xlim(min(fermi), max(fermi))
ax2.set_ylim(0, 10)
ax2.set_xlabel("$E_F$ [meV]")
ax2.set_ylabel("$G[2e^2/h]$")
# ax1.set_title('$N=$ {}, $L=$ {} nm, $V\in(0,${}$)$ meV'.format(Nregions, L, Vmax))
# ax1.legend(loc='upper right', ncol=1)
plt.show()
