import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import h5py
import os
import sys
from datetime import date
from TransportClass import transport, gaussian_correlated_potential_1D_FFT, gaussian_correlated_potential_2D_FFT, \
      get_fileID, code_testing

start_time = time.time()
#%% Parameters

phi0                 = 2 * pi * 1e-34 / 1.6e-19                                          # Quantum of flux
vf                   = 330                                                               # Fermi velocity in meV nm
B_perp               = 0                                                                 # Perpendicular magnetic field in T
n_flux               = 0.0                                                               # Number of flux quanta threaded
B_par                = 0                                                                 # Parallel magnetic field in T
l_cutoff             = 20                                                                # Cutoff number modes
corr_length          = 10                                                                # Correlation length in nm
dis_strength         = 6                                                                 # Disorder strength in vf / xi scale
r                    = 20                                                                # Radius of the nanowire
Nx                   = 501                                                               # Number of points in the x grid
Ntheta_fft           = 301                                                               # Number of points in the theta grid for the FFT
Ntheta_grid          = Ntheta_fft if Ntheta_fft > 1 else 301                             # Number of points in the theta grid for plotting states
L                    = 900                                                               # Length of the system in nm
x                    = np.linspace(0, L, Nx)                                             # Discretised position
theta                = np.linspace(0, 2 * pi, Ntheta_grid)                               # Discretised angles
N_samples            = 1                                                                 # Number of samples to disorder average
fermi                = np.linspace(-200, 200, 1000)                                      # Fermi energy sample
E_resonance_index    = 506                                                               # Energy to calculate scatt. states
g_index              = ['N_samples', 'fermi', 'radius']                                  # Flag for saving G
test_run             = True if fermi.shape[0] < 25 else False                            # Flag that points to test runs or real runs
load_file            = 'Experiment90.h5'                                                 # File from which to load the data
save_data            = False                                                             # Flag for saving data
calculate_transport  = False                                                             # Flag for calculating the transport
G                    = np.zeros((N_samples, fermi.shape[0]))                             # Conductance storage
V_real_storage       = np.zeros((N_samples, Ntheta_grid, Nx))                            # Vxy storage
V_fft_storage        = np.zeros((N_samples, Ntheta_grid, Nx), dtype='complex128')        # Vfft storage
scatt_density_up     = np.zeros((N_samples, Ntheta_grid, Nx - 1))                        # Scattering states storage
scatt_states_up      = np.zeros((N_samples, Ntheta_grid, Nx - 1), dtype='complex128')    # Scattering states storage
scatt_density_down   = np.zeros((N_samples, Ntheta_grid, Nx - 1))                        # Scattering states storage
scatt_states_down    = np.zeros((N_samples, Ntheta_grid, Nx - 1), dtype='complex128')    # Scattering states storage
scatt_density        = np.zeros((N_samples, Ntheta_grid, Nx - 1))
S                    = None
T                    = None

#%% Transport Calculation

# Loading possible data
outdir = "Data"
if load_file is not None:
    for file in os.listdir(outdir):
        if file == load_file:
            file_path = os.path.join(outdir, file)
            with h5py.File(file_path, 'r') as f:
                datanode1 = f['Potential_xy']
                V_real_load = datanode1[()]  # [0, :, :]
                datanode2 = f['Potential_FFT']
                V_fft_load = datanode2[()]   # [0, :, :]
                datanode3 = f['Conductance']
                G_load = datanode3[()]

# Transport calculation
if calculate_transport:
    for n in range(N_samples):

        # Load/create potential landscape
        if load_file is not None:
            V_fft, V_real = V_fft_load[n, :, :], V_real_load[n, :, :]
        elif Ntheta_fft == 1:
            V_real = gaussian_correlated_potential_1D_FFT(L, Nx, dis_strength, corr_length, vf); V_fft = V_real
        else:
            V_fft, V_real = gaussian_correlated_potential_2D_FFT(L, r, Nx, Ntheta_fft, dis_strength, corr_length, vf)

        # Create geometry
        model_gauss = transport(vf=vf, B_perp=B_perp, B_par=B_par, l_cutoff=l_cutoff)
        model_gauss.build_geometry(np.repeat(r, x.shape[0]), x, V_fft)

        # Conductance calculation
        for j, E in enumerate(fermi):
            start_iter = time.time()
            G[n, j] = model_gauss.get_Landauer_conductance(E)
            print('iter sample: {}/{} | iter transport: {}/{} | iter time: {:.3e} s | Fermi level: {:.3f} | Conductance: {:.2f}'.format
                (n, N_samples - 1, j, fermi.shape[0] - 1, time.time() - start_iter, E, G[n, j]))

        # Store each disorder realisation
        V_fft_storage[n, :, :] = V_fft
        V_real_storage[n, :, :] = V_real

    # Scattering states at the resonant energies
    scatt_states_up[0, :, :], scatt_states_down[0, :, :]  = model_gauss.get_scattering_states_Tproduct(fermi[E_resonance_index], theta)
    scatt_density_up[0, :, :] = np.conj(scatt_states_up) * scatt_states_up
    scatt_density_down[0, :, :] = np.conj(scatt_states_down) * scatt_states_down

else:

    # Load data
    G = G_load
    V_fft, V_real = V_fft_load[0, :, :], V_real_load[0, :, :]
    V_real_storage, V_fft_storage = V_real_load, V_fft_load

    # Create instance of the transport class
    model_gauss = transport(vf=vf, B_perp=B_perp, B_par=B_par, l_cutoff=l_cutoff)
    model_gauss.build_geometry(np.repeat(r, x.shape[0]), x, V_fft)

    # Scattering states at the resonant energies
    scatt_states_up[0, :, :], scatt_states_down[0, :, :]  = model_gauss.get_scattering_states_S2T(fermi[E_resonance_index], theta)
    scatt_density_up[0, :, :] = np.abs(scatt_states_up[0, :, :]) ** 2
    scatt_density_down[0, :, :] = np.abs(scatt_states_down[0, :, :]) ** 2
    scatt_density[0, :, :] = scatt_density_down[0, :, :] + scatt_density_up[0, :, :]

# Average conductance
G_avg = np.mean(G, axis=0)
time_lapse = time.time() - start_time
print(f'Time elapsed: {time_lapse:.2e}')


#%% Data storage
file_list = os.listdir('Data')
expID = get_fileID(file_list)
filename ='{}{}{}'.format('Experiment', expID, '.h5')
filepath = os.path.join('Data', filename)
if save_data:
    with h5py.File(filepath, 'w') as f:
        f.create_dataset("Conductance",                       data=G)
        f.create_dataset("Potential_FFT",                     data=V_fft_storage)
        f.create_dataset("Potential_xy",                      data=V_real_storage)
        f.create_dataset("scatt_states_up",                   data=scatt_states_up)
        f.create_dataset("scatt_states_down",                 data=scatt_states_down)
        f["Conductance"].attrs.create("Date",                 data=str(date.today()))
        f["Conductance"].attrs.create("Codepath",             data=sys.argv[0])
        f["Conductance"].attrs.create("vf",                   data=vf)
        f["Conductance"].attrs.create("B_perp",               data=B_perp)
        f["Conductance"].attrs.create("n_flux",               data=n_flux)
        f["Conductance"].attrs.create("B_par",                data=B_par)
        f["Conductance"].attrs.create("l_cutoff",             data=l_cutoff)
        f["Conductance"].attrs.create("corr_length",          data=corr_length)
        f["Conductance"].attrs.create("dis_strength",         data=dis_strength)
        f["Conductance"].attrs.create("radius",               data=r)
        f["Conductance"].attrs.create("x",                    data=x)
        f["Conductance"].attrs.create("Nx",                   data=Nx)
        f["Conductance"].attrs.create("Ntheta_fft",           data=Ntheta_fft)
        f["Conductance"].attrs.create("Ntheta_grid",          data=Ntheta_grid)
        f["Conductance"].attrs.create("N_samples",            data=N_samples)
        f["Conductance"].attrs.create("fermi",                data=fermi)
        f["Conductance"].attrs.create("fermi0",               data=fermi[0])
        f["Conductance"].attrs.create("fermi-1",              data=fermi[-1])
        f["Conductance"].attrs.create("fermi_resolution",     data=fermi.shape[0])
        f["Conductance"].attrs.create("Gshape",               data=g_index)
        f["Conductance"].attrs.create("Test_run",             data=test_run)


#%% Figures
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
axcolour = ['#FF7D66', '#FF416D', '#00B5A1',  '#3F6CFF']
color_list = ['#FF7256', '#00BFFF', '#00C957', '#9A32CD', '#FFC125']
Vstd_th_1d = np.sqrt(dis_strength / np.sqrt(2 * pi)) * (vf / corr_length)      # Standard deviation 1d Gauss
Vstd_th_2d = np.sqrt(dis_strength / (2 * pi)) * vf / corr_length               # Standard deviation 2d Gauss


# Rotationally symmetric potential
if Ntheta_grid == 1:

    # Conductance plot
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(fermi, G_avg, color=color_list[1], label='$r=$ {} nm'.format(r))
    ax1.set_xlim(min(fermi), max(fermi))
    ax1.set_ylim(0, 10)

    ax1.plot(np.repeat(2 * Vstd_th_1d, 10), np.linspace(0, 10, 10), '--', color='#A9A9A9', alpha=0.5)
    ax1.plot(np.repeat(-2 * Vstd_th_1d, 10), np.linspace(0, 10, 10), '--', color='#A9A9A9', alpha=0.5)
    ax1.text(2 * Vstd_th_1d - 10, 9, '$2\sigma$', fontsize=20, rotation='vertical', color='#A9A9A9', alpha=0.5)
    ax1.text(- 2 * Vstd_th_1d + 3, 9, '$2\sigma$', fontsize=20, rotation='vertical', color='#A9A9A9', alpha=0.5)

    ax1.tick_params(which='major', width=0.75, labelsize=20)
    ax1.tick_params(which='major', length=14, labelsize=20)
    ax1.set_xlabel("$E_F$ [meV]", fontsize=20)
    ax1.set_ylabel("$G[2e^2/h]$",fontsize=20)
    # ax1.set_title(" Gaussian correlated: ExpID= {}, $\\xi=$ {} nm, $N_q=$ {}, $N_s=$ {}, $L=$ {} nm, $K_v=$ {}".format(expID, corr_length, Nx, N_samples, x[-1], dis_strength))
    ax1.legend(loc='upper right', ncol=1, fontsize=20)



    # Inset showing a potential sample
    sample = 0
    left, bottom, width, height = [0.35, 0.65, 0.3, 0.3]
    inset_ax1 = ax1.inset_axes([left, bottom, width, height])
    inset_ax1.plot(x, V_real_storage[sample, 0, :], color='#6495ED')
    inset_ax1.set_xlim(x[0], x[-1])
    inset_ax1.set_ylim(-4 * Vstd_th_1d, 4 * Vstd_th_1d)

    inset_ax1.plot(x, Vstd_th_1d * np.ones(x.shape), '--k')
    inset_ax1.plot(x, -Vstd_th_1d * np.ones(x.shape), '--k')
    inset_ax1.plot(x, 2 * Vstd_th_1d * np.ones(x.shape), '--k')
    inset_ax1.plot(x, -2 * Vstd_th_1d * np.ones(x.shape), '--k')
    inset_ax1.plot(x, 3 * Vstd_th_1d * np.ones(x.shape), '--k')
    inset_ax1.plot(x, -3 * Vstd_th_1d * np.ones(x.shape), '--k')
    inset_ax1.text(450, 1.1 * Vstd_th_1d, '$1\sigma$', fontsize=20)
    inset_ax1.text(450, 2.1 * Vstd_th_1d, '$2\sigma$', fontsize=20)
    inset_ax1.text(450, 3.1 * Vstd_th_1d, '$3\sigma$', fontsize=20)
    inset_ax1.text(450, -1.5 * Vstd_th_1d,'$1\sigma$', fontsize=20)
    inset_ax1.text(450, -2.5 * Vstd_th_1d, '$2\sigma$', fontsize=20)
    inset_ax1.text(450, -3.5 * Vstd_th_1d, '$3\sigma$', fontsize=20)

    ax1.tick_params(which='major', width=0.75, labelsize=20)
    ax1.tick_params(which='major', length=14, labelsize=20)
    inset_ax1.set_xlabel("$L$ [nm]", fontsize=20)
    inset_ax1.set_ylabel("$V$ [meV]", fontsize=20)
    # inset_ax1.set_title(" Gaussian correlated potential samples with $\\xi=$ {} nm, $K_V=$ {} and $N_x=$ {}".format(corr_length, dis_strength, Nx))
    inset_ax1.plot()
    plt.show()


# Broken rotational symmetry
else:

   # Conductance plot
    color_list = ['#FF7256', '#00BFFF', '#00C957', '#9A32CD', '#FFC125']
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(fermi, G_avg, color=color_list[1], label='$r=$ {} nm'.format(r))
    ax1.set_xlim(min(fermi), max(fermi))
    ax1.set_ylim(0, 10)

    ax1.plot(np.repeat(2 * Vstd_th_2d, 10), np.linspace(0, 10, 10), '--', color='#A9A9A9', alpha=0.5)
    ax1.plot(np.repeat(-2 * Vstd_th_2d, 10), np.linspace(0, 10, 10), '--', color='#A9A9A9', alpha=0.5)
    ax1.text(2 * Vstd_th_2d - 10, 3, '$2\sigma$', fontsize=20, rotation='vertical', color='#A9A9A9', alpha=0.5)
    ax1.text(- 2 * Vstd_th_2d + 3, 3, '$2\sigma$', fontsize=20, rotation='vertical', color='#A9A9A9', alpha=0.5)

    ax1.set_xlabel("$E_F$ [meV]", fontsize=20)
    ax1.set_ylabel("$G[2e^2/h]$", fontsize=20)
    ax1.tick_params(which='major', width=0.75, labelsize=20)
    ax1.tick_params(which='major', length=14, labelsize=20)
    # ax1.set_title(" Gaussian correlated: ExpID= {}, $\\xi=$ {} nm, $N_x=$ {}, $N_y=$ {}, $N_s=$ {}, $L=$ {} nm, $K_v=$ {}".format(expID, corr_length, Nx, Ntheta, N_samples, x[-1], dis_strength))
    ax1.legend(loc='upper right', ncol=1, fontsize=20)



   # Inset showing a potential sample
   #  sample = 0
   #  left, bottom, width, height = [0.31, 0.55, 0.4, 0.4]
   #  inset_ax1 = ax1.inset_axes([left, bottom, width, height])
   #  density_plot = inset_ax1.imshow(V_real_storage[sample, :, :], origin='lower', vmin=np.min(V_real_storage[sample, :, :]), vmax=np.max(V_real_storage[sample, :, :]))
   #
   #  divider1 = make_axes_locatable(inset_ax1)
   #  cax1 = divider1.append_axes("right", size="5%", pad=0.05)
   #  cbar = fig.colorbar(density_plot, cax=cax1, orientation='vertical')
   #  cbar.set_label(label='$V$ [meV]', labelpad=10, fontsize=20)
   #  cbar.ax.tick_params(which='major', length=14, labelsize=15)
   #
   #  inset_ax1.set_xlabel("$x$ [nm]", fontsize=20)
   #  inset_ax1.set_ylabel("$r\\theta$ [nm]", fontsize=20)
   #  inset_ax1.set(xticks=[0, int(Nx/2), Nx - 1], xticklabels=[0, int(L/2), L])
   #  inset_ax1.set(yticks=[0, int(Ntheta_grid/2), Ntheta_grid - 1], yticklabels=[0, int(pi * r), int(2 * pi * r)])
   #  inset_ax1.tick_params(which='major', width=0.75, labelsize=15)
   #  inset_ax1.tick_params(which='major', length=14, labelsize=15)
   #
   #  inset_ax1.set_title(" Gaussian correlated potential samples with $\\xi=$ {} nm, $K_V=$ {} and $N_x=$ {}".format(corr_length, dis_strength, Nx))
   #  inset_ax1.plot()


   # Inset showing the distribution of scattering states
    sample = 0
   #  left, bottom, width, height = [0.31, 0.55, 0.4, 0.4]
   #  inset_ax1 = ax1.inset_axes([left, bottom, width, height])
   #  density_plot = inset_ax1.imshow(scatt_density[sample, :, :] / np.max(scatt_density[sample, :, :]), origin='lower',
   #                                  vmin=0, vmax=1)
   #
   #  divider1 = make_axes_locatable(inset_ax1)
   #  cax1 = divider1.append_axes("right", size="5%", pad=0.05)
   #  cbar = fig.colorbar(density_plot, cax=cax1, orientation='vertical')
   #  cbar.set_label(label='$\\vert \psi \\vert ^2$', labelpad=10, fontsize=20)
   #  cbar.ax.tick_params(which='major', length=14, labelsize=15)
   #
   #  inset_ax1.set_xlabel("$x$ [nm]", fontsize=20)
   #  inset_ax1.set_ylabel("$r\\theta$ [nm]", fontsize=20)
   #  inset_ax1.set(xticks=[0, int(Nx / 2), Nx - 1], xticklabels=[0, int(L / 2), L])
   #  inset_ax1.set(yticks=[0, int(Ntheta / 2), Ntheta - 1], yticklabels=[0, int(pi * r), int(2 * pi * r)])
   #  inset_ax1.tick_params(which='major', width=0.75, labelsize=15)
   #  inset_ax1.tick_params(which='major', length=14, labelsize=15)
   #
   #  inset_ax1.set_title(" Bound state density at energy $E=$ {} nm".format(fermi[E_resonance_index]))
   #  inset_ax1.plot()


   # Distribution of scattering states
    fig, ax2 = plt.subplots(figsize=(8, 6))
    density_plot = ax2.imshow(scatt_density[sample, :, :] / np.max(scatt_density[sample, :, :]), origin='lower',
                                    vmin=0, vmax=1)

    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    cbar2 = fig.colorbar(density_plot, cax=cax2, orientation='vertical')
    cbar2.set_label(label='$\\vert \psi \\vert ^2$', labelpad=10, fontsize=20)
    cbar2.ax.tick_params(which='major', length=14, labelsize=15)

    ax2.set_xlabel("$x$ [nm]", fontsize=20)
    ax2.set_ylabel("$r\\theta$ [nm]", fontsize=20)
    ax2.set(xticks=[0, int(Nx / 2), Nx - 1], xticklabels=[0, int(L / 2), L])
    ax2.set(yticks=[0, int(Ntheta_grid / 2), Ntheta_grid - 1], yticklabels=[0, int(pi * r), int(2 * pi * r)])
    ax2.tick_params(which='major', width=0.75, labelsize=15)
    ax2.tick_params(which='major', length=14, labelsize=15)

    ax2.set_title(" Bound state density at energy $E=$ {:.2f} meV".format(fermi[E_resonance_index]))

   # Distribution of scattering states spin up
    fig, ax3 = plt.subplots(figsize=(8, 6))
    density_plot = ax3.imshow(scatt_density_up[sample, :, :] / np.max(scatt_density_up[sample, :, :]), origin='lower',
                              vmin=0, vmax=1)

    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    cbar3 = fig.colorbar(density_plot, cax=cax3, orientation='vertical')
    cbar3.set_label(label='$\\vert \psi \\vert ^2$', labelpad=10, fontsize=20)
    cbar3.ax.tick_params(which='major', length=14, labelsize=15)

    ax3.set_xlabel("$x$ [nm]", fontsize=20)
    ax3.set_ylabel("$r\\theta$ [nm]", fontsize=20)
    ax3.set(xticks=[0, int(Nx / 2), Nx - 1], xticklabels=[0, int(L / 2), L])
    ax3.set(yticks=[0, int(Ntheta_grid / 2), Ntheta_grid - 1], yticklabels=[0, int(pi * r), int(2 * pi * r)])
    ax3.tick_params(which='major', width=0.75, labelsize=15)
    ax3.tick_params(which='major', length=14, labelsize=15)

    ax3.set_title(" Bound state density (spin up) at energy $E=$ {:.2f} meV".format(fermi[E_resonance_index]))


   # Distribution of scattering states spin down
    fig, ax4 = plt.subplots(figsize=(8, 6))
    density_plot = ax4.imshow(scatt_density_down[sample, :, :] / np.max(scatt_density_down[sample, :, :]), origin='lower',
                              vmin=0, vmax=1)

    divider4 = make_axes_locatable(ax4)
    cax4 = divider4.append_axes("right", size="5%", pad=0.05)
    cbar4 = fig.colorbar(density_plot, cax=cax4, orientation='vertical')
    cbar4.set_label(label='$\\vert \psi \\vert ^2$', labelpad=10, fontsize=20)
    cbar4.ax.tick_params(which='major', length=14, labelsize=15)

    ax4.set_xlabel("$x$ [nm]", fontsize=20)
    ax4.set_ylabel("$r\\theta$ [nm]", fontsize=20)
    ax4.set(xticks=[0, int(Nx / 2), Nx - 1], xticklabels=[0, int(L / 2), L])
    ax4.set(yticks=[0, int(Ntheta_grid / 2), Ntheta_grid - 1], yticklabels=[0, int(pi * r), int(2 * pi * r)])
    ax4.tick_params(which='major', width=0.75, labelsize=15)
    ax4.tick_params(which='major', length=14, labelsize=15)

    ax4.set_title(" Bound state density (spin down) at energy $E=$ {:.2f} meV".format(fermi[E_resonance_index]))
    plt.show()


