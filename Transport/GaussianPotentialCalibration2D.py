import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from TransportClass import gaussian_correlated_potential_2D
import time

start_time = time.time()
#%% Parameters
vf           = 330                                    # Fermi velocity in meV nm
corr_length  = 10                                     # Correlation length in nm
dis_strength = 6                                      # Disorder strength in vf / xi scale
Nq           = 100                                    # Number of points to take the xFFT
Ntheta       = 100                                    # Number of points to take the thetaFFT
ncheck       = 3                                      # Number of samples of the potential
x            = np.linspace(0, 52, 100)                # x vector sampled by the pontential
theta        = np.linspace(0, 2 * pi, 100)            # theta vector sampled by the potential
r            = np.repeat(8, theta.shape[0])           # r vector sampled by the potential


# Generate gaussian correlated potential
V = np.zeros((ncheck, x.shape[0], theta.shape[0]))
for i in range(ncheck):
    print(i)
    V[i, :, :] = gaussian_correlated_potential_2D(x, theta, r[0], dis_strength, corr_length, vf, Nq, Ntheta)


# Different energy scales
Vstd_th = np.sqrt(dis_strength / 2 * pi) * vf / corr_length ** 2
Vstd_num = np.std(V, axis=0)[0, 0]
V_avg = np.mean(V, axis=0)


# Correlation (varius correlations for different fixed thetas)
# Variance1 = np.zeros((ncheck, x.shape[0]))
# Variance2 = np.zeros((ncheck, x.shape[0]))
# Variance3 = np.zeros((ncheck, x.shape[0]))
# theta_sample = np.random.randint(theta.shape[0], size=6)
# for i in range(ncheck):
#     for j in range(x.shape[0]):
#         aux1 = V[i, :, theta_sample[0]] * np.roll(V[i, :, theta_sample[1]], -j)
#         aux2 = V[i, :, theta_sample[2]] * np.roll(V[i, :, theta_sample[3]], -j)
#         aux3 = V[i, :, theta_sample[6]] * np.roll(V[i, :, theta_sample[5]], -j)
#         Variance1[i, j] = np.mean(aux1[0: -j])
#         Variance2[i, j] = np.mean(aux2[0: -j])
#         Variance3[i, j] = np.mean(aux3[0: -j])
#
# Variance1 = np.mean(Variance1, axis=0)
# Variance2 = np.mean(Variance2, axis=0)
# Variance3 = np.mean(Variance3, axis=0)
# theta_diff1 = theta_sample[1] - theta_sample[0]
# theta_diff2 = theta_sample[3] - theta_sample[2]
# theta_diff3 = theta_sample[5] - theta_sample[4]


#%% Figures

# Distribution of correlated potentials for fixed random theta
fig1, ax1 = plt.subplots(figsize=(8, 6))
for i in range(ncheck): ax1.plot(x, V[i, :, np.random.randint(theta.shape[0])], color='#00BFFF', alpha=0.1)
ax1.plot(x, V[np.random.randint(ncheck), :, np.random.randint(theta.shape[0])], color='#1E90FF', linewidth=3)
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
ax1.set_title(" Gaussian correlated potential samples along $x$ and random fixed $\\theta$ with \n $\\xi=$ {} nm, $K_V=$ {}, "
              "$N_q=$ {} and $N_\\theta=$ {}".format(corr_length, dis_strength, Nq, Ntheta))
plt.plot()


# Distribution of correlated potentials for fixed random x
fig2, ax2 = plt.subplots(figsize=(8, 6))
for i in range(ncheck): ax2.plot(r[0] * theta, V[i, np.random.randint(x.shape[0]), :], color='#00BFFF', alpha=0.1)
ax2.plot(r[0] * theta, V[np.random.randint(ncheck), np.random.randint(x.shape[0]), :], color='#1E90FF', linewidth=3)
ax2.plot(r[0] * theta, Vstd_th * np.ones(theta.shape), '--k')
ax2.plot(r[0] * theta, -Vstd_th * np.ones(theta.shape), '--k')
ax2.plot(r[0] * theta, 2 * Vstd_th * np.ones(theta.shape), '--k')
ax2.plot(r[0] * theta, -2 * Vstd_th * np.ones(theta.shape), '--k')
ax2.plot(r[0] * theta, 3 * Vstd_th * np.ones(theta.shape), '--k')
ax2.plot(r[0] * theta, -3 * Vstd_th * np.ones(theta.shape), '--k')
ax2.text(450, 1.1 * Vstd_th, '$1\sigma$', fontsize=20)
ax2.text(450, 2.1 * Vstd_th, '$2\sigma$', fontsize=20)
ax2.text(450, 3.1 * Vstd_th, '$3\sigma$', fontsize=20)
ax2.text(450, -1.5 * Vstd_th, '$1\sigma$', fontsize=20)
ax2.text(450, -2.5 * Vstd_th, '$2\sigma$', fontsize=20)
ax2.text(450, -3.5 * Vstd_th, '$3\sigma$', fontsize=20)
ax2.set_xlim(r[0] * theta[0], r[0] * theta[-1])
ax2.set_ylim(-4 * Vstd_th, 4 * Vstd_th)
ax2.set_xlabel("$r\\theta$ [nm]")
ax2.set_ylabel("$V$ [meV]")
ax2.set_title(" Gaussian correlated potential samples along $\\theta$ and random fixed $x$ with \n $\\xi=$ {} nm, $K_V=$ {}, "
              "$N_q=$ {} and $N_\\theta=$ {}".format(corr_length, dis_strength, Nq, Ntheta))





#
# # Average potential
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
#
# # Probability distribution
# Vhist = V.flatten()
# x_vec = np.linspace(-4 * Vstd_th, 4 * Vstd_th, Vhist.shape[0])
# norm = (1 / np.sqrt(2 * pi)) * (1 / Vstd_th)
# Pdist = norm * np.exp(-0.5 * x_vec ** 2 / Vstd_th ** 2)
# fig3, ax3 = plt.subplots(figsize=(8, 6))
# ax3.hist(Vhist, bins="auto", density='True')
# ax3.plot(x_vec, Pdist, 'r')
# ax3.set_xlabel("$V(x)$")
# ax3.set_ylabel("$P(V(x))$")
# ax3.set_title("$V(x)$ distribution with $\\xi=$ {} nm, $K_V=$ {} and $N_q=$ {}".format(corr_length, dis_strength, Nq))
# plt.plot()
#
# # Spacial correlations
# Variance_th = (Vstd_th ** 2) * np.exp(-0.5 * x ** 2 / corr_length ** 2)
# fig4, ax4 = plt.subplots(figsize=(8, 6))
# ax4.plot(x, Variance, '.r')
# ax4.plot(x, Variance_th, '-b')
# ax4.set_ylabel("$\langle V(x) V(x') \\rangle$ ")
# ax4.set_xlabel("$x$")
# ax4.set_yscale('log')
# ax4.set_ylim(1, 1000)
# ax4.set_xlim(0, 10 * corr_length)
# ax4.set_title("$\langle V(x) V(x') \\rangle$  with $\\xi=$ {} nm, $K_V=$ {} and $N_q=$ {}".format(corr_length, dis_strength, Nq))
# plt.plot()