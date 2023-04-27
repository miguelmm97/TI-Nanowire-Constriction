import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from TransportClass import transport, gaussian_correlated_potential
import time

start_time = time.time()
#%% Parameters
vf           = 330                                                      # Fermi velocity in meV nm
corr_length  = 10                                                       # Correlation length in nm
dis_strength = 20                                                       # Disorder strength in vf / xi scale
Nq           = 100                                                      # Number of points to take the FFT


# Generate gaussian correlated potential
ncheck = 200
x = np.linspace(0, 500, 1000)
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


# Correlation
Variance = np.zeros(V.shape)
for i in range(ncheck):
    for j in range(x.shape[0]):
        aux = V[i, :] * np.roll(V[i, :], -j)
        Variance[i, j] = np.mean(aux[0: -j])

Variance = np.mean(Variance, axis=0)

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

# Probability distribution
Vhist = V.flatten()
x_vec = np.linspace(-4 * Vstd_th, 4 * Vstd_th, Vhist.shape[0])
norm = (1 / np.sqrt(2 * pi)) * (1 / Vstd_th)
Pdist = norm * np.exp(-0.5 * x_vec ** 2 / Vstd_th ** 2)
fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.hist(Vhist, bins="auto", density='True')
ax3.plot(x_vec, Pdist, 'r')
ax3.set_xlabel("$V(x)$")
ax3.set_ylabel("$P(V(x))$")
ax3.set_title("$V(x)$ distribution with $\\xi=$ {} nm, $K_V=$ {} and $N_q=$ {}".format(corr_length, dis_strength, Nq))
plt.plot()

# Spacial correlations
Variance_th = (Vstd_th ** 2) * np.exp(-0.5 * x ** 2 / corr_length ** 2)
fig4, ax4 = plt.subplots(figsize=(8, 6))
ax4.plot(x, Variance, '.r')
ax4.plot(x, Variance_th, '-b')
ax4.set_ylabel("$\langle V(x) V(x') \\rangle$ ")
ax4.set_xlabel("$x$")
ax4.set_yscale('log')
ax4.set_ylim(1, 1000)
ax4.set_xlim(0, 10 * corr_length)
ax4.set_title("$\langle V(x) V(x') \\rangle$  with $\\xi=$ {} nm, $K_V=$ {} and $N_q=$ {}".format(corr_length, dis_strength, Nq))
plt.plot()