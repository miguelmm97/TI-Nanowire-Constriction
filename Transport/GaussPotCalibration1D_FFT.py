import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from TransportClass import gaussian_correlated_potential_1D_FFT
import time

start_time = time.time()
#%% Parameters
vf           = 330             # Fermi velocity in meV nm
corr_length  = 10              # Correlation length in nm
dis_strength = 6               # Disorder strength in vf / xi scale
L            = 500             # Length of the wire
rad          = 20              # Radius of the wire
Nx           = 501             # Number of points in the x grid
ncheck       = 2000            # Samples of the potential

# Generate gaussian correlated potential
x = np.linspace(0, L, Nx)
r = np.repeat(rad, x.shape[0])
V = np.zeros((ncheck, x.shape[0]))
for i in range(ncheck):
    print('Sample {}/{}'.format(i, ncheck - 1))
    V[i, :] = gaussian_correlated_potential_1D_FFT(L, Nx, dis_strength, corr_length, vf)

# Correlations
average_full = np.mean(V)
average_sample = np.mean(V, axis=1)
Variance = np.zeros((V.shape[0], int(x.shape[0]/2)))
for i in range(ncheck):
    aux = np.outer(V[i, :], V[i, :])
    for j in range(int(x.shape[0]/2)): Variance[i, j] = (np.trace(aux, j) + np.trace(aux, x.shape[0] - j))/(Nx)
Variance = np.mean(Variance, axis=0)

# Different energy scales
Vstd_th = np.sqrt(dis_strength / np.sqrt(2 * pi)) * (vf / corr_length)
Vstd_num = np.std(V, axis=0)[0]

# Fidelity of the sample
def funcfit(y, C1, C2): return C1 * np.exp(0.5 * (-y ** 2)/ (C2 ** 2))
fit, covariance1 = curve_fit(funcfit, x[: -int(Nx / 2) - 1], Variance)
xi_fit1 = np.sqrt((dis_strength / fit[0]) * (vf ** 2) / np.sqrt(2 * pi))
xi_fit2 = fit[1]

#%% Figures

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
ax1.set_title(" Gaussian correlated potential samples with $\\xi=$ {} nm, $K_V=$ {} and $N_x=$ {}".format(corr_length, dis_strength, Nx))
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


# Spacial correlations
Variance_th = (Vstd_th ** 2) * np.exp(-0.5 * x[: -int(Nx / 2)] ** 2 / corr_length ** 2)
fig4, ax4 = plt.subplots(figsize=(8, 6))
ax4.plot(x[: -int(Nx / 2)-1]**2, Variance, '.r')
ax4.plot(x[: -int(Nx / 2)]**2, Variance_th, '-b')
ax4.plot(x[: -int(Nx / 2) - 1]**2, (Vstd_th ** 2) * np.ones(int(Nx/2)), '--')
ax4.set_ylabel("$\langle V(x) V(x') \\rangle$ ")
ax4.set_xlabel("$x²$")
ax4.set_yscale('log')
ax4.set_ylim(1, 3000)
ax4.set_xlim(0, 2000)
ax4.set_title("$\langle V(x) V(x') \\rangle$  with $\\xi=$ {} nm, $K_V=$ {} and $N_q=$ {}".format(corr_length, dis_strength, Nx))
ax4.text(1500, 1000, f'$\\xi_{{fit1}} / \\xi_{{th}} = {xi_fit1 / corr_length:.2f}$')
ax4.text(1500, 500, f'$\\xi_{{fit2}} / \\xi_{{th}} = {xi_fit2 / corr_length:.2f}$')
ax4.text(1500, 100, f'$Scale_{{fit}} / Scale_{{th}} = {fit[0] / Vstd_th ** 2:.2f}$')

# Numerical error with system size
# fig5, ax5 = plt.subplots(figsize=(8, 6))
# ax5.plot(L_vec, num_error/ (Vstd_th ** 2), '.r')
# ax5.set_ylabel("$(\langle V_t^2 \\rangle - \langle V^2 \\rangle) / \langle V_t^2\\rangle$ ")
# ax5.set_xlabel("$L$")
# ax5.set_xlim(0, L_vec[-1])
# ax5.set_title('Numerical error in the variance for $dx = $ {} and $\\xi =$ {}'.format(dx, corr_length))




