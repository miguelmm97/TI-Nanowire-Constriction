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
Adis      = 5                                                       # Disorder amplitude


#%% Geometry for an array of wells
# model1 = transport(vf, B_perp, B_par, l_cutoff)                     # Instance of the transport class
# n_wells = 20
# r1 = 50; conf_gap1 = vf / r1
# L1 = 150; L2 = 2* L1; x0 = 0
# V1 = 4 * conf_gap1
# Vnm1 = np.eye(2 * l_cutoff + 1)
#
# for i in np.arange(n_wells):
#     Vdis = conf_gap1  * np.random.uniform(-Adis, Adis, size=1)
#     if i == 0:
#         model1.add_nw(x0, x0 + L1, r=r1)
#         x0 = x0 + L1
#     model1.add_nw(x0, x0 + L2, r=r1, Vnm=-Vnm1 * Vdis)
#     model1.add_nw(x0 + L2, x0 + L2 + L1, r=r1)
#     x0 = x0 + L1 + L2

#%% Geometry for different nanorwires
model2 = transport(vf, B_perp, B_par, l_cutoff)                    # Instance of the transport class
r2 = 8; conf_gap2 = vf / r2
x02 = 0; x12 = 50; x22 = 3 * x12; x32 = x22 + x12
V2 = 4 * conf_gap2
Vnm2 = V2 * np.eye(2 * l_cutoff + 1)
model2.add_nw(x02, x12, r=r2)


# Geometry
# model3 = transport(vf, B_perp, B_par, l_cutoff)                    # Instance of the transport class
# r3 = 50; conf_gap3 = vf / r3
# x03 = 0; x13 = 150; x23 = 6 * x13; x33 = x23 + x13
# V3 = 6 * conf_gap3
# Vnm3 = V3 * np.eye(2 * l_cutoff + 1)
# model3.add_nw(x03, x13, r=r3)
# model3.add_nw(x13, x23, r=r3, Vnm=-Vnm3)
# model3.add_nw(x23, x33, r=r3)

# Geometry
# model4 = transport(vf, B_perp, B_par, l_cutoff)                    # Instance of the transport class
# r4 = 50; conf_gap4 = vf / r4
# x04 = 0; x14 = 150; x24 = 6 * x14; x34 = x24 + x14
# V4 = 8 * conf_gap3
# Vnm4 = V4 * np.eye(2 * l_cutoff + 1)
# model4.add_nw(x04, x14, r=r4)
# model4.add_nw(x14, x24, r=r4, Vnm=-Vnm4)
# model4.add_nw(x24, x34, r=r4)
#

# Bands
# k_range = np.linspace(-1, 1, 5000)
# Enw, Vnw = model1.get_bands_nw(0, k_range)

#%% Conductance calculation
# fermi     = np.linspace(-0.5 * conf_gap1, 0.5 * conf_gap1, 500)      # Fermi level
fermi     = np.linspace(-conf_gap2 - 10, conf_gap2 + 10, 1000)
G1        = np.zeros(fermi.shape)                                    # Conductance preallocation
G2        = np.zeros(fermi.shape)                                    # Conductance preallocation
G3        = np.zeros(fermi.shape)                                    # Conductance preallocation
G4        = np.zeros(fermi.shape)                                    # Conductance preallocation
for i, E in enumerate(fermi):
    start_iter = time.time()
    # G1[i] = model1.get_Landauer_conductance(E)
    G2[i] = model2.get_Landauer_conductance(E)
    # G3[i] = model3.get_Landauer_conductance(E)
    # G4[i] = model4.get_Landauer_conductance(E)
    print('iter: {}/{} | time: {:.3e} s | G: {:.2e}'.format(i, len(fermi), time.time() - start_iter, G1[i]))

#%% Figures
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 13, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=15)


# # Bands in the leads
# fig, ax1 = plt.subplots(figsize=(8, 6))
# for i in range(len(Enw[:, 0])):
#     ax1.plot(k_range, Enw[i, :] - V1, '.', color='#6495ED', markersize=2)
#     ax1.plot(k_range, np.ones(k_range.shape) * Vnw[i], color='#A9A9A9', alpha=0.5)
# ax1.set_xlim(-0.1, 0.1)
# ax1.set_ylim(-max(fermi), max(fermi))
# ax1.set_xlabel("$k$ [1/nm]")
# ax1.set_ylabel("$E_{l<L/2}$ [meV]")
# ax1.set_title('$r=$ {}nm, $L_1=$ {}nm'.format(r1, x11))

# # Bands in the leads
# fig2, ax2 = plt.subplots(figsize=(8, 6))
# for i in range(len(Enw2[:, 0])):
#     ax2.plot(k_range, Enw2[i, :] - V, '.', color='#6495ED', markersize=2)
#     ax2.plot(k_range, np.ones(k_range.shape) * (Vnw2[i] - V), '--', color='#A9A9A9', alpha=0.5)
# ax2.set_xlim(-0.1, 0.1)
# ax2.set_ylim(-max(fermi), max(fermi))
# ax2.set_xlabel("$k$ [1/nm]")
# ax2.set_ylabel("$E_{l>L/2}$ [meV]")
# ax2.set_title('$r=$ {}nm, $L_2=$ {}nm'.format(r, x2 - x1))



# Conductance as a function of the fermi level
fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.plot(fermi, G1, color='#6495ED')
ax3.plot(fermi, G2, color='#FF7256')
ax3.plot(fermi, G3, color='#CD2626')
ax3.plot(fermi, G4, color='#00CD66')
# for i in range(len(Vnw)):
    # ax3.plot(np.ones((10,)) * Vnw[i], np.linspace(0, max(fermi), 10), color='#A9A9A9', alpha=0.5)
    # ax3.plot(np.ones((10,)) * (Vnw2[i] - V), np.linspace(0, max(fermi), 10), '--', color='#A9A9A9', alpha=0.5)
# ax3.plot(-0.5 * V1 * np.ones((10,)), np.linspace(0, 10, 10), '--', color='b', alpha=0.5, label='$\Delta V/2$')
# ax3.plot(0 * np.ones((10,)), np.linspace(0, 10, 10), '--', color='r', alpha=0.5, label='$V_{nw}$')
# ax3.plot(- V1 * np.ones((10,)), np.linspace(0, 10, 10), '--', color='r', alpha=0.5)
# ax3.plot(0.5 * conf_gap1 * np.ones((10,)), np.linspace(0, 10, 10), '--', color='#A9A9A9', alpha=0.5, label='$E_g/2$')
# ax3.plot(-0.5 * conf_gap1 * np.ones((10,)), np.linspace(0, 10, 10), '--', color='#A9A9A9', alpha=0.5)
# ax3.plot((-V1 + 0.5 * conf_gap1) * np.ones((10,)), np.linspace(0, 10, 10), '--', color='#A9A9A9', alpha=0.5)
# ax3.plot((-V1 -0.5 * conf_gap1) * np.ones((10,)), np.linspace(0, 10, 10), '--', color='#A9A9A9', alpha=0.5)
# for i in range(1, 20):
#     ax3.plot(fermi, np.repeat(i, len(fermi)), color='#A9A9A9', alpha=0.5)
# ax3.axvspan(-V1 - 0.5 * conf_gap1, -V1 + 0.5 * conf_gap1, color='#A9A9A9', alpha=0.5, lw=0)
# ax3.axvspan(-0.5 * conf_gap1, 0.5 * conf_gap1, color='#A9A9A9', alpha=0.5, lw=0)
# ax3.axvspan(-0.5 * conf_gap1 - V1, 0.5 * conf_gap1, color='#FFA07A', alpha=0.5, lw=0)


ax3.set_xlim(min(fermi), max(fermi))
ax3.set_ylim(0, 10)
ax3.set_xlabel("$E_F$ [meV]")
ax3.set_ylabel("$G[2e^2/h]$")
# ax3.set_title('$n_w=$ {}, $L_1=L_3=$ {} nm, $L_2=$ {}, $r$ = {} nm, $\Delta V=$ {} meV'.format(n_wells, L1, L2, r1, V1))
# ax3.set_title('$r=$ {} nm, $L_1=$ {} nm, $L_2=$ {} nm, $\Delta V= {} $meV'.format(r, x1, x2-x1, V))
# ax3.set_title('Nanowire conductance for a potential well with $L_1=L_3=$ {} nm, $L_2=$ {}, $r$ = {} nm'.format(x11, x21-x11, r1))
# ax3.set_title('Nanowire conductance for a potential step, $L=$ {} nm, $r=$ {} nm'.format(x11, r1))
# ax3.legend(('$L=$ {} nm'.format(x21-x11), '$L=$ {} nm'.format(x22-x12), '$L=$ {} nm'.format(x23-x13), '$L=$ {} nm'.format(x24-x14)))
# ax3.legend(('$V=$ -{0:.2g} meV'.format(V1), '$V=$ -{0:.2g} meV'.format(V2), '$V=$ -{0:.2g} meV'.format(V3), '$V=$ -{0:.2g} meV'.format(V4)))
# ax3.legend(('$r=$ {} nm'.format(r1), '$r=$ {} nm'.format(r2), '$r=$ {} nm'.format(r3), '$r=$ {} nm'.format(r4)))
# ax3.legend()
plt.show()
print('Time elapsed: {:.2e} s'.format(time.time() - start_time))





# x0 = 0; x1 = 100; x2 = x1 + 594.7; x3 = x2 + 800 - 594.7; x4 = x3 + 594.7; x5 = x4 + 100
# r_lead = 156.6; r_cons = r_lead / 2
# sigma  = 0.01                                                         # Smoothing factor
# # model.add_nw(x0, x1, r=r_lead)                                      # Lead 1
# model.add_nc(0, 579, 250, r1=78.3, r2=156.6)                     # Nanocone 1
# # model.add_nw(x2, x3, r=r_cons)                                      # Constriction
# # model.add_nc(x3, x4, 250, sigma=sigma, r1=r_cons, r2=r_lead)        # Nanocone 2
# # model.add_nw(x4, x5, r=r_lead)