import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from TransportClass import transport, gaussian_correlated_potential_1D_FFT, gaussian_correlated_potential_2D_FFT, get_fileID
import time
import h5py
import os
import sys
from datetime import date
from numpy.fft import fft2, ifft

start_time = time.time()
#%% Parameters

phi0         = 2 * pi * 1e-34 / 1.6e-19                        # Quantum of flux
vf           = 330                                             # Fermi velocity in meV nm
B_perp       = 0                                               # Perpendicular magnetic field in T
n_flux       = 0.0                                             # Number of flux quanta threaded
B_par        = 0                                               # Parallel magnetic field in T
l_cutoff     = 20                                              # Cutoff number modes
corr_length  = 10                                              # Correlation length in nm
dis_strength = 6                                               # Disorder strength in vf / xi scale
radius       = np.array([20])                                  # Radius of the nanowire
Nx           = 501                                             # Number of points in the x grid
Ntheta       = 301                                             # Number of points in the theta grid
L            = 500                                             # Length of the system in nm
x            = np.linspace(0, L, Nx)                           # Discretised position
N_samples    = 1                                               # Number of samples to disorder average
fermi        = np.linspace(-200, 200, 1000)                    # Fermi energy sample
g_index      = ['N_samples', 'fermi', 'radius']                # Flag for saving G
test_run     = True if fermi.shape[0] < 25 else False          # Flag that points to test runs or real runs
load_data    = "Experiment66.h5"  # None                       # File from which to load the data
#%% Transport Calculation

Vstd_th_1d = np.sqrt(dis_strength / np.sqrt(2 * pi)) * (vf / corr_length)
Vstd_th_2d = np.sqrt(dis_strength / (2 * pi)) * vf / corr_length


# Loading possible data
outdir = "Data"
if load_data is None:
    G              = np.zeros((N_samples, fermi.shape[0], radius.shape[0]))
    VFFT_storage   = np.zeros((radius.shape[0], Ntheta, Nx), dtype='complex128')
    Vxy_storage    = np.zeros((radius.shape[0], Ntheta, Nx))
for file in os.listdir(outdir):
    if file == "Experiment66.h5":
        file_path = os.path.join(outdir, file)
        with h5py.File(file_path, 'r') as f:
            datanode1 = f['Potential_xy']
            V_xy = datanode1[()][0, :, :]
            datanode2 = f['Conductance']
            G = datanode2[()][0, :, :]


# V = gaussian_correlated_potential_1D_FFT(L, Nx, dis_strength, corr_length, vf)


# sin_fft = np.zeros((11,))
# sin_fft[1] = 0.5 * 1j / np.sqrt(2 * pi * 20)
# sin_fft[-1] = sin_fft[1].conj()
# theta_sample = np.linspace(0, 2*pi, Ntheta, endpoint=False)
# V_sample = 10 * np.ones((Ntheta, Nx))
# V_sample = np.zeros((Ntheta, Nx))
# for i in range(Nx): V_sample[:, i] = np.sin(theta_sample )
# V_1 = fft2(V_sample) * np.sqrt(L * 2 * pi * 20) / (Nx * Ntheta)
# V_2 = ifft(V_1, axis=1) * (Nx / np.sqrt(L)) * (1 / np.sqrt(2 * pi * 20))
# V_2 = np.repeat(10, Nx)

for i, rad in enumerate(radius):
    r = np.repeat(rad, x.shape[0])

    for n in range(N_samples):
        # Instance of the transport class
        model_gauss = transport(vf, B_perp, B_par, l_cutoff)
        V_FFT, V_xy = gaussian_correlated_potential_2D_FFT(L, rad, Nx, Ntheta, dis_strength, corr_length, vf)
        model_gauss.build_geometry(r, x, V_FFT)
        # model_gauss.build_geometry(r, x, V_2)
        VFFT_storage[i, :, :] = V_FFT
        Vxy_storage[i, :, :] = V_xy

        # Conductance calculation
        for j, E in enumerate(fermi):
            start_iter = time.time()
            G[n, j, i] = model_gauss.get_Landauer_conductance(E)
            print('iter radius: {}/{} | iter sample: {}/{} | iter transport: {}/{} | iter time: {:.3e} s | Fermi level: {:.3f} | Conductance: {:.2f}'.format
                  (i, radius.shape[0]-1, n, N_samples-1, j, fermi.shape[0]-1, time.time() - start_iter, E, G[n, j, i]))

G_avg = np.mean(G, axis=0)
G_avg = G

#%% Data storage
# file_list = os.listdir('Data')
# expID = get_fileID(file_list)
# filename ='{}{}{}'.format('Experiment', expID, '.h5')
# filepath = os.path.join('Data', filename)
# with h5py.File(filepath, 'w') as f:
#     f.create_dataset("Conductance", data=G)
#     f.create_dataset("Potential_FFT", data=VFFT_storage)
#     f.create_dataset("Potential_xy", data=Vxy_storage)
#     f["Conductance"].attrs.create("Date",                 data=str(date.today()))
#     f["Conductance"].attrs.create("Codepath",             data=sys.argv[0])
#     f["Conductance"].attrs.create("vf",                   data=vf)
#     f["Conductance"].attrs.create("B_perp",               data=B_perp)
#     f["Conductance"].attrs.create("n_flux",               data=n_flux)
#     f["Conductance"].attrs.create("B_par",                data=B_par)
#     f["Conductance"].attrs.create("l_cutoff",             data=l_cutoff)
#     f["Conductance"].attrs.create("corr_length",          data=corr_length)
#     f["Conductance"].attrs.create("dis_strength",         data=dis_strength)
#     f["Conductance"].attrs.create("radius",               data=radius)
#     f["Conductance"].attrs.create("x",                    data=x)
#     f["Conductance"].attrs.create("Nx",                   data=Nx)
#     f["Conductance"].attrs.create("Ntheta",               data=Ntheta)
#     f["Conductance"].attrs.create("N_samples",            data=N_samples)
#     f["Conductance"].attrs.create("fermi",                data=fermi)
#     f["Conductance"].attrs.create("fermi0",               data=fermi[0])
#     f["Conductance"].attrs.create("fermi-1",              data=fermi[-1])
#     f["Conductance"].attrs.create("fermi_resolution",     data=fermi.shape[0])
#     f["Conductance"].attrs.create("Gshape",               data=g_index)
#     f["Conductance"].attrs.create("Test_run",             data=test_run)


#%% Figures
# color_list = ['#FF7256', '#00BFFF', '#00C957', '#9A32CD', '#FFC125']
# fig, ax1 = plt.subplots(figsize=(8, 6))
# for i in range(radius.shape[0]):
#     ax1.plot(fermi, G_avg[:, i], color=color_list[i], label='$r=$ {} nm'.format(radius[i]))
# ax1.plot(np.repeat(2 * Vstd_th, 10), np.linspace(0, 10, 10), '--', color='#A9A9A9', alpha=0.5)
# ax1.plot(np.repeat(-2 * Vstd_th, 10), np.linspace(0, 10, 10), '--', color='#A9A9A9', alpha=0.5)
# ax1.text(2 * Vstd_th - 10, 9, '$2\sigma$', fontsize=20, rotation='vertical', color='#A9A9A9', alpha=0.5)
# ax1.text(- 2 * Vstd_th + 3, 9, '$2\sigma$', fontsize=20, rotation='vertical', color='#A9A9A9', alpha=0.5)
# ax1.set_xlim(min(fermi), max(fermi))
# ax1.set_ylim(0, 10)
# ax1.set_xlabel("$E_F$ [meV]")
# ax1.set_ylabel("$G[2e^2/h]$")
# ax1.set_title(" Gaussian correlated: ExpID= {}, $\\xi=$ {} nm, $N_q=$ {}, $N_s=$ {}, $L=$ {} nm, $K_v=$ {}".format(expID, corr_length, Nx, N_samples, x[-1], dis_strength))
# ax1.legend(loc='upper right', ncol=1)
# plt.show()
#
#
#
# left, bottom, width, height = [0.35, 0.65, 0.3, 0.3]
# inset_ax1 = ax1.inset_axes([left, bottom, width, height])
# inset_ax1.plot(x, V, color='#6495ED')
# inset_ax1.plot(x, Vstd_th * np.ones(x.shape), '--k')
# inset_ax1.plot(x, -Vstd_th * np.ones(x.shape), '--k')
# inset_ax1.plot(x, 2 * Vstd_th * np.ones(x.shape), '--k')
# inset_ax1.plot(x, -2 * Vstd_th * np.ones(x.shape), '--k')
# inset_ax1.plot(x, 3 * Vstd_th * np.ones(x.shape), '--k')
# inset_ax1.plot(x, -3 * Vstd_th * np.ones(x.shape), '--k')
# inset_ax1.text(450, 1.1 * Vstd_th, '$1\sigma$', fontsize=20)
# inset_ax1.text(450, 2.1 * Vstd_th, '$2\sigma$', fontsize=20)
# inset_ax1.text(450, 3.1 * Vstd_th, '$3\sigma$', fontsize=20)
# inset_ax1.text(450, -1.5 * Vstd_th, '$1\sigma$', fontsize=20)
# inset_ax1.text(450, -2.5 * Vstd_th, '$2\sigma$', fontsize=20)
# inset_ax1.text(450, -3.5 * Vstd_th, '$3\sigma$', fontsize=20)
# inset_ax1.set_xlim(x[0], x[-1])
# inset_ax1.set_ylim(-4 * Vstd_th, 4 * Vstd_th)
# inset_ax1.set_xlabel("$L$ [nm]")
# inset_ax1.set_ylabel("$V$ [meV]")
# # inset_ax1.set_title(" Gaussian correlated potential samples with $\\xi=$ {} nm, $K_V=$ {} and $N_x=$ {}".format(corr_length, dis_strength, Nx))
# inset_ax1.plot()
# plt.show()




color_list = ['#FF7256', '#00BFFF', '#00C957', '#9A32CD', '#FFC125']
fig, ax1 = plt.subplots(figsize=(8, 6))
for i in range(radius.shape[0]):
    ax1.plot(fermi, G_avg[:, i], color=color_list[i], label='$r=$ {} nm'.format(radius[i]))
ax1.plot(np.repeat(2 * Vstd_th_2d, 10), np.linspace(0, 10, 10), '--', color='#A9A9A9', alpha=0.5)
ax1.plot(np.repeat(-2 * Vstd_th_2d, 10), np.linspace(0, 10, 10), '--', color='#A9A9A9', alpha=0.5)
ax1.text(2 * Vstd_th_2d - 10, 3, '$2\sigma$', fontsize=20, rotation='vertical', color='#A9A9A9', alpha=0.5)
ax1.text(- 2 * Vstd_th_2d + 3, 3, '$2\sigma$', fontsize=20, rotation='vertical', color='#A9A9A9', alpha=0.5)
ax1.set_xlim(min(fermi), max(fermi))
ax1.set_ylim(0, 10)
ax1.set_xlabel("$E_F$ [meV]")
ax1.set_ylabel("$G[2e^2/h]$")
ax1.set_title(" Gaussian correlated: ExpID= {}, $\\xi=$ {} nm, $N_x=$ {}, $N_y=$ {}, $N_s=$ {}, $L=$ {} nm, $K_v=$ {}".format(66, corr_length, Nx, Ntheta, N_samples, x[-1], dis_strength))
ax1.legend(loc='upper right', ncol=1)



left, bottom, width, height = [0.31, 0.55, 0.4, 0.4]
inset_ax1 = ax1.inset_axes([left, bottom, width, height])
density_plot = inset_ax1.imshow(V_xy, origin='lower', vmin=np.min(V_xy), vmax=np.max(V_xy))
divider1 = make_axes_locatable(inset_ax1)
cax1 = divider1.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(density_plot, cax=cax1, orientation='vertical')
inset_ax1.set(xticks=[0, int(Nx/2), Nx - 1], xticklabels=[0, int(L/2), L])
inset_ax1.set(yticks=[0, int(Ntheta/2), Ntheta - 1], yticklabels=[0, int(pi * radius[0]), int(2 * pi * radius[0])])
inset_ax1.set_xlabel("$x$ [nm]")
inset_ax1.set_ylabel("$r\\theta$ [nm]")
cbar.set_label(label='$V(x, \\theta)$', labelpad=10)
# inset_ax1.set_title(" Gaussian correlated potential samples with $\\xi=$ {} nm, $K_V=$ {} and $N_x=$ {}".format(corr_length, dis_strength, Nx))
inset_ax1.plot()
plt.show()
