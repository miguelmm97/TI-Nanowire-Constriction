import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from TransportClass import transport
import time

start_time = time.time()
#%% Parameters

# Constants and set up of the model
phi0     = 2 * pi * 1e-34 / 1.6e-19                                  # Quantum of flux
vf       = 330                                                       # Fermi velocity in meV nm
B_perp   = 0                                                         # Perpendicular magnetic field in T
n_flux   = 0.0                                                       # Number of flux quanta threaded
B_par    = 0.2  # n_flux * phi0 / (120 * 20 * 1e-9 * 1e-9)           # Parallel magnetic field in T
l_cutoff = 30                                                        # Cutoff number modes
fermi    = np.linspace(0, 12, 240)                                   # Fermi level
G        = np.zeros(fermi.shape)                                     # Conductance preallocation
model    = transport(vf, B_perp, B_par, l_cutoff)                    # Instance of the transport class

# Geometry
x0 = 0; x1 = 100; x2 = x1 + 594.7; x3 = x2 + 800 - 594.7; x4 = x3 + 594.7; x5 = x4 + 100
r_lead = 156.6; r_cons = r_lead / 2
sigma  = 0.01                                                       # Smoothing factor
model.add_nw(x0, x1, r=r_lead)                                      # Lead 1
model.add_nc(x1, x2, 250, r1=r_lead, r2=r_cons)                     # Nanocone 1
model.add_nw(x2, x3, r=r_cons)                                      # Constriction
model.add_nc(x3, x4, 250, sigma=sigma, r1=r_cons, r2=r_lead)        # Nanocone 2
model.add_nw(x4, x5, r=r_lead)                                      # Lead 2


# Bands in the leads and in the constriction
k_range = np.linspace(-10, 10, 5000)
Eleads, Vleads = model.get_bands_nw(0, k_range)
Econs, Vcons = model.get_bands_nw(2, k_range)


# Conductance calculation
for i, E in enumerate(fermi):
    start_iter = time.time()
    G[i] = model.get_Landauer_conductance(E)
    print('iter: {}/{} | time: {:.3e} s | G: {:.2e}'.format(i, len(fermi), time.time() - start_iter, G[i]))

#%% Figures
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 13, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=15)


# Bands in the leads
fig, ax1 = plt.subplots(figsize=(8, 6))
for i in range(len(Eleads[:, 0])):
    ax1.plot(k_range, Eleads[i, :], '.', color='#6495ED', markersize=2)
    # ax1.plot(k_range, np.ones(k_range.shape) * Vleads[i], '--k')
ax1.set_xlim(-0.5, 0.5)
ax1.set_ylim(-50, 50)
ax1.set_xlabel("$k$ [1/nm]")
ax1.set_ylabel("$E_{lead}$ [meV]")


# Bands in the constriction
fig, ax2 = plt.subplots(figsize=(8, 6))
for i in range(len(Econs[:, 0])):
    ax2.plot(k_range, Econs[i, :], '.', color='#6495ED', markersize=2)
    # ax1.plot(k_range, np.ones(k_range.shape) * Vleads[i], '--k')
ax2.set_xlim(-0.5, 0.5)
ax2.set_ylim(-50, 50)
ax2.set_xlabel("$k$ [1/nm]")
ax2.set_ylabel("$E_{cons}$ [meV]")



# Conductance as a function of the fermi level
fig, ax3 = plt.subplots(figsize=(8, 6))
ax3.plot(fermi, G, color='#6495ED')
for i in range(len(Vleads)):
    ax3.plot(np.ones((10,)) * Vcons[i], np.linspace(0, max(fermi), 10), color='#A9A9A9', alpha=0.5)
for i in range(1, 20):
    ax3.plot(fermi, np.repeat(i, len(fermi)), color='#A9A9A9', alpha=0.5)
ax3.set_xlim(0, max(fermi))
ax3.set_ylim(0, 6)
ax3.set_xlabel("$E_F$ (meV)")
ax3.set_ylabel("$G/G_Q$")


plt.show()
print('Time elapsed: {:.2e} s'.format(time.time() - start_time))