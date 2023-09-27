import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from TransportClass import gaussian_correlated_potential_2D_FFT
import time

start_time = time.time()
#%% Parameters
vf           = 330.              # Fermi velocity in meV nm
corr_length  = 30                # Correlation length in nm
dis_strength = 6.                # Disorder strength in vf / xi scale
L            = 1000.             # Length of the wire
rad          = 20.               # Radius of the wire
Nx           = 501               # Number of points in the x grid
Ntheta       = 301               # Number of points in the x grid
ncheck       = 2000             # Number of samples of the potential

#%% Sample potential landscape
# Generate gaussian correlated potential
x = np.linspace(0, L, Nx)
theta = np.linspace(0, 2 * pi, Ntheta)
r = np.repeat(rad, theta.shape[0])
V = np.zeros((ncheck, theta.shape[0], x.shape[0]))
for i in range(ncheck):
    print(i)
    V[i, :, :] = gaussian_correlated_potential_2D_FFT(L, r[0], Nx, Ntheta, dis_strength, corr_length, vf)


# Different energy scales
Vstd_th = np.sqrt(dis_strength / (2 * pi)) * vf / corr_length
Vstd_num = np.std(V, axis=0)[0, 0]
V_avg = np.mean(V, axis=0)

# Correlations for fixed angle
average_full = np.mean(V)
average_sample = np.mean(V, axis=1)
fixed_theta = np.random.randint(Ntheta)
Variance_x = np.zeros((V.shape[0], int(x.shape[0]/2)))
for i in range(ncheck):
    aux = np.outer(V[i, fixed_theta, :], V[i, fixed_theta, :])
    for j in range(int(x.shape[0]/2)): Variance_x[i, j] = (np.trace(aux, j) + np.trace(aux, x.shape[0] - j))/Nx
Variance_x = np.mean(Variance_x, axis=0)

def funcfit(y, C1, C2): return C1 * np.exp(0.5 * (-y ** 2)/ (C2 ** 2))
fit_x, covariance_x = curve_fit(funcfit, x[: -int(Nx / 2) - 1], Variance_x)
xi_fit1_x = np.sqrt((dis_strength / fit_x[0]) * (vf ** 2) / (2 * pi))
xi_fit2_x = fit_x[1]

# Correlations for a fixed x
fixed_x = np.random.randint(Nx)
Variance_theta = np.zeros((V.shape[0], int(theta.shape[0]/2)))
for i in range(ncheck):
    aux = np.outer(V[i, :, fixed_x], V[i, :, fixed_x])
    for j in range(int(theta.shape[0]/2)): Variance_theta[i, j] = (np.trace(aux, j) + np.trace(aux, x.shape[0] - j))/Ntheta
Variance_theta = np.mean(Variance_theta, axis=0)

fit_theta, covariance_theta = curve_fit(funcfit, r[0] * theta[: -int(Ntheta / 2) - 1], Variance_theta)
xi_fit1_theta = np.sqrt((dis_strength * (vf ** 2)) / (fit_theta[0] * 2 * pi))
xi_fit2_theta = fit_theta[1]

#%% Analysis and debugging
#
# # Dependence of the variance with the correlation length
# xi_vec = np.array([1, 2, 3, 4, 5, 10, 15, 20, 30])
# Variance_xi = np.zeros(xi_vec.shape)
# for i, xi in enumerate(xi_vec):
#     print(xi)
#     V_xi = gaussian_correlated_potential_2D(x, theta, r[0], dis_strength, xi, vf, Nq, Nl)
#     aux = np.outer(V_xi[:, fixed_theta], V_xi[:, fixed_theta])
#     Variance_xi[i] = np.trace(aux) / Nx
#
# def funcfit(y, C): return C / y ** 2
# fit1, covariance1 = curve_fit(funcfit, xi_vec, Variance_xi)
# C_fit1 = fit1
# C_th1 = (dis_strength / 2 * pi) * vf ** 2
# C1_error = C_th1 / C_fit1
#
# fig5, ax5 = plt.subplots(figsize=(8, 6))
# th_curve = (dis_strength / 2 * pi) * (vf ** 2) / xi_vec ** 2
# ax5.plot(xi_vec, Variance_xi, '.r')
# # ax5.plot(xi_vec, th_curve, 'b')
# ax5.plot(xi_vec, C_fit1 / xi_vec ** 2, '--r')
#
#
#
# # Dependence of the variance with the disorder strength
# dis_vec = np.array([1, 3, 5, 7, 9, 11])
# Variance_dis = np.zeros(dis_vec.shape)
# for i, dis in enumerate(dis_vec):
#     print(dis)
#     V_dis = gaussian_correlated_potential_2D(x, theta, r[0], dis, corr_length, vf, Nq, Nl)
#     aux = np.outer(V_dis[:, fixed_theta], V_dis[:, fixed_theta])
#     Variance_dis[i] = np.trace(aux) / Nx
#
# def funcfit(y, C): return C * y
# fit2, covariance = curve_fit(funcfit, dis_vec, Variance_dis)
# C_fit2 = fit2
# C_th2 = (1 / 2 * pi) * (vf / corr_length) ** 2
# C2_error = C_th2 / C_fit2
#
# fig6, ax6 = plt.subplots(figsize=(8, 6))
# th_curve2 = (dis_vec / 2 * pi) * (vf ** 2) / corr_length ** 2
# ax6.plot(dis_vec, Variance_dis, '.r')
# ax6.plot(dis_vec, th_curve2, 'b')
# ax6.plot(dis_vec, C_fit2 * dis_vec, '--r')
#
#
#
# # Dependence on the radius
# # Dependence of the variance with the correlation length
# radius_vec = np.array([10, 30, 60, 100, 150])
# Variance_r = np.zeros(radius_vec.shape)
# for i, radius in enumerate(radius_vec):
#     print(radius)
#     V_radius = gaussian_correlated_potential_2D(x, theta, radius, dis_strength, corr_length, vf, Nq, Nl)
#     aux = np.outer(V_radius[:, fixed_theta], V_radius[:, fixed_theta])
#     Variance_r[i] = np.trace(aux) / Nx
# C3_error = (Vstd_th ** 2) / Variance_r
#
# fig7, ax7 = plt.subplots(figsize=(8, 6))
# ax7.plot(radius_vec, C3_error, '.r')

#%% Figures

# Distribution of correlated potentials for fixed random theta
fig1, ax1 = plt.subplots(figsize=(8, 6))
# for i in range(ncheck): ax1.plot(x, V[i, np.random.randint(theta.shape[0], :)], color='#00BFFF', alpha=0.1)
ax1.plot(x, V[np.random.randint(ncheck), np.random.randint(theta.shape[0]), :], color='#1E90FF', linewidth=3)
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
# ax1.set_ylim(-4 * Vstd_th, 4 * Vstd_th)
ax1.set_xlabel("$L$ [nm]")
ax1.set_ylabel("$V$ [meV]")
ax1.set_title(" Gaussian correlated potential samples along $x$ and random fixed $\\theta$ with \n $\\xi=$ {} nm, $K_V=$ {}, "
              "$N_q=$ {} and $N_\\theta=$ {}".format(corr_length, dis_strength, Nx, Ntheta))
plt.plot()


# Distribution of correlated potentials for fixed random x
fig2, ax2 = plt.subplots(figsize=(8, 6))
# for i in range(ncheck): ax2.plot(r[0] * theta, V[i, :, np.random.randint(x.shape[0])], color='#00BFFF', alpha=0.1)
ax2.plot(r[0] * theta, V[np.random.randint(ncheck), :, np.random.randint(x.shape[0])], color='#1E90FF', linewidth=3)
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
              "$N_q=$ {} and $N_\\theta=$ {}".format(corr_length, dis_strength, Nx, Ntheta))




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



# Spacial correlations
Variance_th = (Vstd_th ** 2) * np.exp(-0.5 * x[: -int(Nx / 2)] ** 2 / corr_length ** 2)
fig4, ax4 = plt.subplots(figsize=(8, 6))
# ax4.plot(x[: -int(Nx / 2)]**2, Variance, '.r')
# ax4.plot(x[: -int(Nx / 2)]**2, Variance_th, '-b')
# ax4.plot(x[: -int(Nx / 2)]**2, (Vstd_th ** 2) * np.ones(int(Nx/2)), '--')
ax4.plot(x[: -int(Nx / 2) - 1] ** 2, Variance_x, '.r')
ax4.plot(x[: -int(Nx / 2)] ** 2, Variance_th, '-b')
ax4.plot(x[: -int(Nx / 2) - 1 ** 2], (Vstd_th ** 2) * np.ones(int(Nx/2)), '--')
ax4.set_ylabel("$\langle V(x) V(x') \\rangle$ ")
ax4.set_xlabel("$x^2$")
ax4.set_yscale('log')
ax4.set_ylim(1, 3000)
ax4.set_xlim(0, 2000)
ax4.set_title("$\langle V(x) V(x') \\rangle$  with $\\xi=$ {} nm, $K_V=$ {} and $N_q=$ {}".format(corr_length, dis_strength, Nx))
ax4.text(1500, 1000, f'$\\xi_{{fit1}} / \\xi_{{th}} = {xi_fit1_x / corr_length:.2f}$')
ax4.text(1500, 500, f'$\\xi_{{fit2}} / \\xi_{{th}} = {xi_fit2_x / corr_length:.2f}$')
ax4.text(1500, 100, f'$Scale_{{fit}} / Scale_{{th}} = {fit_x[0] / Vstd_th ** 2:.2f}$')

# Spacial correlations
Variance_th = (Vstd_th ** 2) * np.exp(-0.5 * (r[0] * theta[: -int(Ntheta / 2)])  ** 2 / corr_length ** 2)
fig5, ax5 = plt.subplots(figsize=(8, 6))
# ax4.plot(x[: -int(Nx / 2)]**2, Variance, '.r')
# ax4.plot(x[: -int(Nx / 2)]**2, Variance_th, '-b')
# ax4.plot(x[: -int(Nx / 2)]**2, (Vstd_th ** 2) * np.ones(int(Nx/2)), '--')
ax5.plot((r[0] * theta[: -int(Ntheta / 2) - 1]) ** 2, Variance_theta, '.r')
ax5.plot((r[0] * theta[: -int(Ntheta / 2)]) ** 2, Variance_th, '-b')
ax5.plot((r[0] * theta[: -int(Ntheta / 2) - 1]) ** 2, (Vstd_th ** 2) * np.ones(int(Ntheta/2)), '--')
ax5.set_ylabel("$\langle V(\\theta) V(\\theta') \\rangle$ ")
ax5.set_xlabel("$r^2\\theta^2$")
ax5.set_yscale('log')
ax5.set_ylim(1, 3000)
ax5.set_xlim(0, 2000)
ax5.set_title("$\langle V(\\theta) V(\\theta')  \\rangle$  with $\\xi=$ {} nm, $K_V=$ {} and $N_q=$ {}".format(corr_length, dis_strength, Nx))
ax5.text(1500, 1000, f'$\\xi_{{fit1}} / \\xi_{{th}} = {xi_fit1_theta / corr_length:.2f}$')
ax5.text(1500, 500, f'$\\xi_{{fit2}} / \\xi_{{th}} = {xi_fit2_theta / corr_length:.2f}$')
ax5.text(1500, 100, f'$Scale_{{fit}} / Scale_{{th}} = {fit_theta[0] / Vstd_th ** 2:.2f}$')