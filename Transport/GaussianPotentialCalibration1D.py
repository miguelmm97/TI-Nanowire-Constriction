import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from TransportClass import gaussian_correlated_potential_1D
import time

start_time = time.time()
#%% Parameters
vf           = 330               # Fermi velocity in meV nm
corr_length  = 10                # Correlation length in nm
dis_strength = 6                 # Disorder strength in vf / xi scale
Nq           = 80                # Number of points to take the FFT
L            = 1000              # Length of the wire
Nx           = 500               # Number of points in the x grid


# Generate gaussian correlated potential
ncheck = 200
x = np.linspace(0, L, Nx)
r = np.repeat(8, x.shape[0])
V = np.zeros((ncheck, x.shape[0]))
for i in range(ncheck):
    print('Sample {}/{}'.format(i, ncheck - 1))
    V[i, :] = gaussian_correlated_potential_1D(x, dis_strength, corr_length, vf, Nq)[0]
print('dx = {}, 1/q = {}'.format(x[1] - x[0], (x[-1] - x[0]) / (2 * pi * Nq)))


# Different energy scales
Vstd_th = np.sqrt((dis_strength / (corr_length * np.sqrt(2 * pi))) * (vf / corr_length) ** 2)
Vstd_num = np.std(V, axis=0)[0]
Vscale_th = np.sqrt((1 / (corr_length * np.sqrt(2 * pi))) * (vf / corr_length) ** 2)
Vscale_num = np.sqrt(np.sum(np.var(V, axis=0) * (x[1] - x[0])) / (x[-1] * dis_strength))
V_avg = np.mean(V, axis=0)


# Correlations
Variance_aux = np.zeros(V.shape)
for i in range(ncheck):
    for j in range(x.shape[0]):
        aux = V[i, :] * np.roll(V[i, :], -j)
        if j == 0: Variance_aux[i, j] = np.mean(aux)
        else: Variance_aux[i, j] = np.mean(aux[0: -j])
Variance = np.mean(Variance_aux, axis=0)


# Quality of the fit depending on the system size
# L_vec = np.linspace(100, 2000, 20); dx = 0.2; dq = 0.19; ncheck2 = 200   # dx = 2.04; dq = 1.9;
# num_error = np.zeros(L_vec.shape)
# for i, l in enumerate(L_vec):
#     print('Length {}/{}'.format(i, L_vec.shape[0]))
#     Nx = int(np.floor(l / dx))
#     Nq = int(np.floor(l / (2 * pi * dq)))
#
#     # Generate potential
#     x2 = np.linspace(0, l, Nx)
#     V2 = np.zeros((ncheck2, x2.shape[0]))
#     for j in range(ncheck2):
#         V2[j, :] = gaussian_correlated_potential_1D(x2, dis_strength, corr_length, vf, Nq)
#
#     # Correlations
#     Variance_aux2 = np.zeros(V2.shape)
#     for n in range(ncheck2):
#         for m in range(x2.shape[0]):
#             aux = V2[n, :] * np.roll(V2[n, :], -m)
#             if m == 0: Variance_aux2[n, m] = np.mean(aux)
#             else: Variance_aux2[n, m] = np.mean(aux[0: -m])
#     Variance2 = np.mean(Variance_aux2, axis=0)
#
#     # Difference from the theoretical value
#     num_error[i] = (Vstd_th ** 2) - Variance2[0]



#%% Figures

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

# Numerical error with system size
# fig5, ax5 = plt.subplots(figsize=(8, 6))
# ax5.plot(L_vec, num_error/ (Vstd_th ** 2), '.r')
# ax5.set_ylabel("$\langle V_t^2 \\rangle - \langle V^2 \\rangle$ ")
# ax5.set_xlabel("$L$")
# ax5.set_xlim(0, L_vec[-1])
# ax5.set_title('Numerical error in the variance for $dx = $ {}, $1/q = $ {}'.format(dx, dq))


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
plt.show()


# Average potential
# fig2, ax2 = plt.subplots(figsize=(8, 6))
# ax2.plot(x, V_avg, color='#1E90FF')
# ax2.plot(x, Vstd_th * np.ones(x.shape) / np.sqrt(ncheck), '--k')
# ax2.plot(x, -Vstd_th * np.ones(x.shape) / np.sqrt(ncheck), '--k')
# ax2.plot(x, 2 * Vstd_th * np.ones(x.shape) / np.sqrt(ncheck), '--k')
# ax2.plot(x, -2 * Vstd_th * np.ones(x.shape) / np.sqrt(ncheck), '--k')
# ax2.plot(x, 3 * Vstd_th * np.ones(x.shape) / np.sqrt(ncheck), '--k')
# ax2.plot(x, -3 * Vstd_th * np.ones(x.shape) / np.sqrt(ncheck), '--k')
# ax2.text(425, 1.1 * Vstd_th / np.sqrt(ncheck), '$1\sigma / \sqrt{n}$', fontsize=20)
# ax2.text(425, 2.1 * Vstd_th / np.sqrt(ncheck), '$2\sigma / \sqrt{n}$', fontsize=20)
# ax2.text(425, 3.1 * Vstd_th / np.sqrt(ncheck), '$3\sigma / \sqrt{n}$', fontsize=20)
# ax2.text(425, -1.5 * Vstd_th / np.sqrt(ncheck), '$1\sigma / \sqrt{n}$', fontsize=20)
# ax2.text(425, -2.5 * Vstd_th / np.sqrt(ncheck), '$2\sigma / \sqrt{n}$', fontsize=20)
# ax2.text(425, -3.5 * Vstd_th / np.sqrt(ncheck), '$3\sigma / \sqrt{n}$', fontsize=20)
# ax2.set_xlim(x[0], x[-1])
# ax2.set_ylim(-4 * Vstd_th / np.sqrt(ncheck), 4 * Vstd_th / np.sqrt(ncheck))
# ax2.set_xlabel("$L$ [nm]")
# ax2.set_ylabel("$V$ [meV]")
# ax2.set_title("Average gaussian correlated potential with $\\xi=$ {} nm, $K_V=$ {} and $N_q=$ {}".format(corr_length, dis_strength, Nq))
# plt.plot()

# Probability distribution
# Vhist = V[:, 444].flatten()
# V_vec = np.linspace(-4 * Vstd_th, 4 * Vstd_th, Vhist.shape[0])
# norm = (1 / np.sqrt(2 * pi)) * (1 / Vstd_th)
# Pdist = norm * np.exp(-0.5 * V_vec ** 2 / Vstd_th ** 2)
# fig3, ax3 = plt.subplots(figsize=(8, 6))
# ax3.hist(Vhist, bins="auto", density='True')
# ax3.plot(V_vec, Pdist, 'r')
# ax3.set_xlabel("$V(x)$")
# ax3.set_ylabel("$P(V(x))$")
# ax3.set_title("$V(x)$ distribution with $\\xi=$ {} nm, $K_V=$ {} and $N_q=$ {}".format(corr_length, dis_strength, Nq))
# plt.plot()
