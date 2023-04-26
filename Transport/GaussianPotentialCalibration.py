import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from TransportClass import transport, gaussian_correlated_potential
import time

start_time = time.time()
#%% Parameters
vf           = 330                                                      # Fermi velocity in meV nm
corr_length  = 10                                                       # Correlation length in nm
dis_strength = 1                                                        # Disorder strength in vf / xi scale
Nq           = 100                                                      # Number of points to take the FFT


# Geberate gaussian correlated potential
ncheck = 200
x = np.linspace(0, 500, 400)
r = np.repeat(8, x.shape[0])
V = np.zeros((ncheck, x.shape[0]))
for i in range(ncheck):
    V[i, :] = gaussian_correlated_potential(x, dis_strength, corr_length, vf, Nq)

# Different energy scales
Vstd_th = np.sqrt((dis_strength / (corr_length * np.sqrt(2 * pi))) * (vf / corr_length) ** 2)
Vstd_num = np.std(V, axis=0)[0]
Vscale_th = np.sqrt((1 / (corr_length * np.sqrt(2 * pi))) * (vf / corr_length) ** 2)
Vscale_num = np.sqrt(np.sum(np.var(V, axis=0) * (x[1] - x[0])) / (x[-1] * dis_strength))
V_avg = np.mean(V, axis=0)


# Distribution of correlated potentials
fig1, ax1 = plt.subplots(figsize=(8, 6))
for i in range(ncheck): ax1.plot(x, V[i, :], color='#00BFFF', alpha=0.1)
ax1.plot(x, V[np.random.randint(0, high=V.shape[0]), :], color='#1E90FF', linewidth=3)
ax1.plot(x, Vstd_th * np.ones(x.shape), '--k')
ax1.plot(x, -Vstd_th * np.ones(x.shape), '--k')
ax1.plot(x, 2 * Vstd_th * np.ones(x.shape), '--k')
ax1.plot(x, -2 * Vstd_th * np.ones(x.shape), '--k')
ax1.plot(x, 3 * Vstd_th * np.ones(x.shape), '--k')
ax1.plot(x, -3 * Vstd_th * np.ones(x.shape), '--k')
ax1.text(450, 1.1 * Vstd_th, '$1\sigma$', fontsize=20)
ax1.text(450, 2.1 * Vstd_th, '$2\sigma$', fontsize=20)
ax1.text(450, 3.1 * Vstd_th, '$3\sigma$', fontsize=20)
ax1.text(450, -1.5 * Vstd_th, '$1\sigma$', fontsize=20)
ax1.text(450, -2.5 * Vstd_th, '$2\sigma$', fontsize=20)
ax1.text(450, -3.5 * Vstd_th, '$3\sigma$', fontsize=20)
ax1.set_xlim(x[0], x[-1])
ax1.set_ylim(-4 * Vstd_th, 4 * Vstd_th)
ax1.set_xlabel("$L$ [nm]")
ax1.set_ylabel("$V$ [meV]")
ax1.set_title(" Gaussian correlated potential samples with $\\xi=$ {} nm, $K_V=$ {} and $N_q=$ {}".format(corr_length, dis_strength, Nq))
plt.plot()

# Average potential
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.plot(x, V_avg, color='#1E90FF')
ax2.plot(x, Vstd_th * np.ones(x.shape) / np.sqrt(ncheck), '--k')
ax2.plot(x, -Vstd_th * np.ones(x.shape) / np.sqrt(ncheck), '--k')
ax2.plot(x, 2 * Vstd_th * np.ones(x.shape) / np.sqrt(ncheck), '--k')
ax2.plot(x, -2 * Vstd_th * np.ones(x.shape) / np.sqrt(ncheck), '--k')
ax2.plot(x, 3 * Vstd_th * np.ones(x.shape) / np.sqrt(ncheck), '--k')
ax2.plot(x, -3 * Vstd_th * np.ones(x.shape) / np.sqrt(ncheck), '--k')
ax2.text(425, 1.1 * Vstd_th / np.sqrt(ncheck), '$1\sigma / \sqrt{n}$', fontsize=20)
ax2.text(425, 2.1 * Vstd_th / np.sqrt(ncheck), '$2\sigma / \sqrt{n}$', fontsize=20)
ax2.text(425, 3.1 * Vstd_th / np.sqrt(ncheck), '$3\sigma / \sqrt{n}$', fontsize=20)
ax2.text(425, -1.5 * Vstd_th / np.sqrt(ncheck), '$1\sigma / \sqrt{n}$', fontsize=20)
ax2.text(425, -2.5 * Vstd_th / np.sqrt(ncheck), '$2\sigma / \sqrt{n}$', fontsize=20)
ax2.text(425, -3.5 * Vstd_th / np.sqrt(ncheck), '$3\sigma / \sqrt{n}$', fontsize=20)
ax2.set_xlim(x[0], x[-1])
ax2.set_ylim(-4 * Vstd_th / np.sqrt(ncheck), 4 * Vstd_th / np.sqrt(ncheck))
ax2.set_xlabel("$L$ [nm]")
ax2.set_ylabel("$V$ [meV]")
ax2.set_title("Average gaussian correlated potential with $\\xi=$ {} nm, $K_V=$ {} and $N_q=$ {}".format(corr_length, dis_strength, Nq))
plt.plot()

