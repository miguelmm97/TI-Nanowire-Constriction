import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from TransportClass import transport
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
Adis      = 5                                                        # Disorder amplitude


# Random potential landscape parameters
L         = 500
Nregions  = 10
Vmax      = 60
conf_gap  = 20
fermi     = np.linspace(-100, 150, 1000)

#%% Transport calculation

# Geometry
# Ldis = np.random.uniform(0.75, 1.25, size=Nregions)
# Ldis = L * Ldis / np.sum(Ldis)
Ldis = (L/Nregions) * np.ones((Nregions, ))
Vdis = np.random.uniform(0, Vmax, size=Nregions)
# Vdis = Vmax * np.tile(np.array([1, 0]), int(Nregions/2))
r = vf / (2 * conf_gap)

# Instance of the transport class
l0 = 0
random_model = transport(vf, B_perp, B_par, l_cutoff)
for l, V in zip(Ldis, Vdis):
    random_model.add_nw(l0, l0 + l, r=r, Vnm=V * np.eye(2 * l_cutoff + 1))
    l0 += l

# Conductance calculation
G = np.zeros(fermi.shape)
for i, E in enumerate(fermi):
    start_iter = time.time()
    G[i] = random_model.get_Landauer_conductance(E)
    print('iter: {}/{} | time: {:.3e} s | G: {:.2e}'.format(i, len(fermi), time.time() - start_iter, G[i]))

print('Time elapsed: {:.2e} s'.format(time.time() - start_time))
#%% Figures
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 13, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=15)

# Conductance as a function of the fermi level
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(fermi, G, color='#6495ED')
ax1.set_xlim(min(fermi), max(fermi))
ax1.set_ylim(0, 10)
ax1.set_xlabel("$E_F$ [meV]")
ax1.set_ylabel("$G[2e^2/h]$")
# ax1.set_title('$N=$ {}, $r$ = {} nm, $V\in(0,${}$)$ meV, $E_g=$ {} meV'.format(Nregions, r, Vmax, conf_gap))
ax1.set_title('$N=$ {}, $r$ = {} nm, $L=$ {} nm, $V=$ {} meV, $E_g=$ {} meV'.format(Nregions, r, L, Vmax, conf_gap))
plt.show()

# Potential landscape
l0, V0 = 0, 0
fig2, ax2 = plt.subplots(figsize=(8, 6))
for l, V in zip(Ldis, Vdis):
    ax2.plot(np.linspace(l0, l0 + l, 10), V * np.ones((10, )), '--', color='#6495ED')
    ax2.plot(l0 * np.ones((10, )), np.linspace(V0, V, 10), '--', color='#6495ED')
    ax2.plot(np.linspace(l0, l0 + l, 10), (V + conf_gap/2) * np.ones((10,)), '-', color="#00CD66")
    ax2.plot(l0 * np.ones((10,)), np.linspace((V0 + conf_gap/2), (V + conf_gap/2), 10), '-', color="#00CD66")
    ax2.plot(np.linspace(l0, l0 + l, 10), (V - conf_gap/2) * np.ones((10,)), '-', color="#00CD66")
    ax2.plot(l0 * np.ones((10,)), np.linspace((V0 - conf_gap/2), (V - conf_gap/2), 10), '-', color="#00CD66")
    ax2.fill_between(np.linspace(l0, l0 + l, 10), (V - conf_gap/2) * np.ones((10, )), y2 = -conf_gap, color="#00CD66", alpha=0.5)
    ax2.fill_between(np.linspace(l0, l0 + l, 10), (V + conf_gap / 2) * np.ones((10,)), y2=Vmax + conf_gap, color="#00CD66", alpha=0.5)
    l0, V0 = l0 + l, V

ax2.set_xlim(0, L)
ax2.set_ylim(-conf_gap, Vmax + conf_gap)
ax2.set_xlabel("$L$ [nm]")
ax2.set_ylabel("$V$ [meV]")
ax2.set_title('Random potential landscape')
plt.show()
