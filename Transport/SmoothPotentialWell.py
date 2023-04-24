import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from TransportClass import transport, geom_cons
import time

start_time = time.time()
#%% Parameters

# Constants and set up of the model
phi0         = 2 * pi * 1e-34 / 1.6e-19                                 # Quantum of flux
vf           = 330                                                      # Fermi velocity in meV nm
B_perp       = 0                                                        # Perpendicular magnetic field in T
n_flux       = 0.0                                                      # Number of flux quanta threaded
B_par        = 0  # n_flux * phi0 / (120 * 20 * 1e-9 * 1e-9)            # Parallel magnetic field in T
l_cutoff     = 30                                                       # Cutoff number modes
fermi        = np.linspace(-100, 100, 10)
G_smooth     = np.zeros(fermi.shape)
G_sharp      = np.zeros(fermi.shape)
model_smooth = transport(vf, B_perp, B_par, l_cutoff)
model_sharp  = transport(vf, B_perp, B_par, l_cutoff)

# Geometry parameters
r = 8
Vmax = 0; Vmin = -50
Lleads = 20; Lwell = 5; Ltransition = 5
L = 2 * Lleads + 2 * Ltransition + Lwell
Npoints = 100; dx = L / Npoints
smoothing_factor = 0.1





#%% Transport calculation

# Discretisation of the smooth potential
Xwell0 = Lleads; Xwell1 = Xwell0 + Ltransition; Xwell2 = Xwell1 + Lwell; Xwell3 = Xwell2 + Ltransition
x = np.linspace(0, Xwell3 + Lleads, Npoints)
V = geom_cons(x, Xwell0, Xwell1, Xwell2, Vmax, Vmin, smoothing_factor)
r = np.repeat(r, x.shape[0])
model_smooth.build_geometry(r, x, V)

# Sharp potential well
r1 = 8; conf_gap1 = vf / r1
x01 = 0; x11 = 20; x21 = x11 + 15; x31 = x21 + x11
V1 = -26
Vnm1 = V1 * np.eye(2 * l_cutoff + 1)
model_sharp.add_nw(x01, x11, r=r1)
model_sharp.add_nw(x11, x21, r=r1, Vnm=Vnm1)
model_sharp.add_nw(x21, x31, r=r1)


for i, E in enumerate(fermi):
    start_iter = time.time()
    G_smooth[i] = model_smooth.get_Landauer_conductance(E)
    G_sharp[i] = model_sharp.get_Landauer_conductance(E)
    print('iter: {}/{} | time: {:.3e} s | G: {:.2e}'.format(i, G_smooth.shape[0], time.time() - start_iter, G_smooth[i]))


fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(x, V, color='#FF7256')
ax1.set_xlim(x[0], x[-1])
ax1.set_ylim(Vmin - 10, Vmax + 10)
ax1.set_xlabel("$L$ [nm]")
ax1.set_ylabel("$V$ [meV]")
plt.plot()

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.plot(fermi, G_smooth, color='#FF7256', label='Smooth')
ax2.plot(fermi, G_sharp, '.-', color='#9A32CD', label='Sharp')
ax2.set_xlim(min(fermi), max(fermi))
ax2.set_ylim(0, 10)
ax2.set_xlabel("$E_F$ [meV]")
ax2.set_ylabel("$G[2e^2/h]$")
ax2.set_title('$N=$ {}, $L_l=$ {} nm, $L_t=$ {} nm, $L_w=$ {} nm, $V=-$ {} meV, $r=$ {} nm'.format(Npoints, Lleads, Ltransition, Lwell, Vmin, r[0]))
ax2.legend(loc='upper right', ncol=1)
plt.show()
