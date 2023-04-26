import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from TransportClass import transport, gaussian_correlated_potential
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
corr_length  = 10                                                       # Correlation length in nm
dis_strength = 10                                                      # Disorder strength in vf / xi scale
Nq           = 100                                                      # Number of points to take the FFT
fermi        = np.linspace(-100, 100, 1000)
G            = np.zeros(fermi.shape)
model_gauss  = transport(vf, B_perp, B_par, l_cutoff)


# Geometry parameters
x = np.linspace(0, 500, 400)
r = np.repeat(8, x.shape[0])
V = gaussian_correlated_potential(x, dis_strength, corr_length, vf, Nq)
model_gauss.build_geometry(r, x, 0.01 * V)

# Transport calculation
for i, E in enumerate(fermi):
    start_iter = time.time()
    G[i] = model_gauss.get_Landauer_conductance(E)
    print('iter: {}/{} | time: {:.3e} s | G: {:.2e}'.format(i, G.shape[0], time.time() - start_iter, G[i]))


#%% Figures
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(x, V[1, :], color='#FF7256')
ax1.set_xlim(x[0], x[-1])
ax1.set_ylim(- 2 * vf / corr_length, 2 * vf / corr_length)
ax1.set_xlabel("$L$ [nm]")
ax1.set_ylabel("$V$ [meV]")
ax1.set_title(" Gaussian correlated potential landscape with $\\xi=$ {} nm, $K_V=$ {} and $N_q=$ {}".format(corr_length, dis_strength, Nq))
plt.plot()

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.plot(fermi, G, color='#FF7256', label='Smooth')
ax2.set_xlim(min(fermi), max(fermi))
ax2.set_ylim(0, 10)
ax2.set_xlabel("$E_F$ [meV]")
ax2.set_ylabel("$G[2e^2/h]$")
ax2.set_title(" Gaussian correlated potential landscape with $\\xi=$ {} nm, $K_V=$ {} and $N_q=$ {}".format(corr_length, dis_strength, Nq))
ax2.legend(loc='upper right', ncol=1)
plt.show()

