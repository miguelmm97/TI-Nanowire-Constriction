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
dis_strength = np.array([1, 10, 20])                                    # Disorder strength in vf / xi scale
Nq           = 100                                                      # Number of points to take the FFT
N_samples    = 100                                                      # Number of samples to disorder average
fermi        = np.linspace(-100, 100, 200)
G            = np.zeros((N_samples, fermi.shape[0], dis_strength.shape[0]))

# Geometry parameters
x = np.linspace(0, 500, 400)
r = np.repeat(8, x.shape[0])

#%% Calculations

# Transport calculation
for i, K in enumerate(dis_strength):
    for n in range(N_samples):

        # Instance of the transport class
        model_gauss = transport(vf, B_perp, B_par, l_cutoff)
        V = gaussian_correlated_potential(x, K, corr_length, vf, Nq)
        model_gauss.build_geometry(r, x, V)

        # Conductance calculation
        for j, E in enumerate(fermi):
            start_iter = time.time()
            G[n, j, i] = model_gauss.get_Landauer_conductance(E)
            print('iter disorder: {}/{} | iter sample: {}/{} | iter transport: {}/{} | iter time: {:.3e} s'.format
                  (i, dis_strength.shape[0], n, N_samples, j, fermi.shape[0], time.time() - start_iter))


G_avg1 = np.mean(G[:, :, 0], axis=0)
G_avg2 = np.mean(G[:, :, 1], axis=0)
G_avg3 = np.mean(G[:, :, 2], axis=0)

#%% Figures

fig, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(fermi, G_avg1, color='#FF7256', label='$K_v=$ {}'.format(dis_strength[0]))
ax1.plot(fermi, G_avg2, color='#00C957', label='$K_v=$ {}'.format(dis_strength[1]))
ax1.plot(fermi, G_avg3, color='#00BFFF', label='$K_v=$ {}'.format(dis_strength[2]))
ax1.set_xlim(min(fermi), max(fermi))
ax1.set_ylim(0, 10)
ax1.set_xlabel("$E_F$ [meV]")
ax1.set_ylabel("$G[2e^2/h]$")
ax1.set_title(" Gaussian correlated: $\\xi=$ {} nm, $N_q=$ {}, $L=$ {} nm, $r=$ {} nm".format(corr_length, Nq, x[-1], r[0]))
ax1.legend(loc='upper right', ncol=1)
plt.show()

