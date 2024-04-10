#%% Modules setup

# Math and plotting
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec

# Managing system, data and config files
import h5py
import os
import sys
import config
from TransportClass import store_my_data, attr_my_data, get_fileID

# Tracking time
import time
from datetime import date

# Managing logging
import logging
import colorlog
from colorlog import ColoredFormatter

# External modules
from TransportClass import transport, gaussian_correlated_potential_1D_FFT, gaussian_correlated_potential_2D_FFT,\
    check_imaginary, smooth_circular_quantum_dot

# %% Logging setup
logger_main = logging.getLogger('main')
logger_main.setLevel(logging.INFO)

stream_handler = colorlog.StreamHandler()
formatter = ColoredFormatter(
    '%(black)s%(asctime) -5s| %(blue)s%(name) -10s %(black)s| %(cyan)s %(funcName) '
    '-40s %(black)s|''%(log_color)s%(levelname) -10s | %(message)s',
    datefmt=None,
    reset=True,
    log_colors={
        'TRACE': 'black',
        'DEBUG': 'purple',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)
stream_handler.setFormatter(formatter)
logger_main.addHandler(stream_handler)

# %% Initialise variables
variables = config.variables
for key in variables:
    globals().update(variables[key])

x                     = np.linspace(0, L, Nx)                                             # Discretised position
Ntheta_grid           = Ntheta_fft if dimension=='2d' else 1                              # Grid for theta
Ntheta_plot           = Ntheta_fft if dimension=='2d' else default_Ntheta_plot            # Grid for plotting theta
theta                 = np.linspace(0, 2 * pi, Ntheta_plot)                               # Discretised angles
fermi                 = np.linspace(fermi_0, fermi_end, fermi_length)                     # Fermi energy sample
G                     = np.zeros((N_samples, fermi.shape[0]))                             # Conductance storage
V_real_storage        = np.zeros((N_samples, Ntheta_grid, Nx))                            # Vxy storage
V_fft_storage         = np.zeros((N_samples, Ntheta_grid, Nx), dtype=np.complex128)       # Vfft storage
scatt_states_up       = np.zeros((N_samples, Ntheta_plot, Nx - 1), dtype=np.complex128)   # Scattering states storage
scatt_density_up      = np.zeros((N_samples, Ntheta_plot, Nx - 1), dtype=np.complex128)   # Scattering states storage
scatt_states_down     = np.zeros((N_samples, Ntheta_plot, Nx - 1), dtype=np.complex128)   # Scattering states storage
scatt_density_down    = np.zeros((N_samples, Ntheta_plot, Nx - 1), dtype=np.complex128)   # Scattering states storage


# %% Loading possible data
outdir = "Data"
if load_data:
    logger_main.info('Loading data from {}'.format(load_file))
    for file in os.listdir(outdir):
        if file == load_file:
            file_path = os.path.join(outdir, file)
            with h5py.File(file_path, 'r') as f:
                datanode1     = f['Potential_xy']
                datanode2     = f['Potential_FFT']
                datanode3     = f['Conductance']
                V_real_load   = datanode1[()]
                V_fft_load    = datanode2[()]
                G_load        = datanode3[()]


# %% Transport Calculation
start_time = time.time()

# Transport calculation
if calculate_transport:
    logger_main.info('Performing transport calculation...')
    for n in range(N_samples):

        # Load/create potential landscape
        logger_main.trace('Generating potential...')
        L_lead, L_trans, L_well = L / 5, L / 5, L / 5
        gap = vf / (2 * r)
        V0, V1 = 20, - 20    # 20 - 2 * gap
        amplitude, period = 30, L / 3

        if load_data:
            V_fft, V_real = V_fft_load[n, :, :], V_real_load[n, :, :]
        elif dimension=='1d':
            V_real = gaussian_correlated_potential_1D_FFT(L, Nx, dis_strength, corr_length, vf)
            # V_real = potential_well_1D_smooth(x, L_lead, L_trans, L_well, V0, V1, smoothing_factor=1)
            # V_real = potential_well_1D(L, Nx, V0, V0, L/3, L/3)
            # V_real = potential_barrier(x, 30, -60, L/2, smoothing=0.1)
            # V_real = sin_potential(x, amplitude, period)
            V_fft = V_real
        else:
            V_fft, V_real = gaussian_correlated_potential_2D_FFT(L, r, Nx, Ntheta_fft, dis_strength, corr_length, vf)
            # V_real, V_fft = sin_potential_theta(theta, Nx, amplitude, pi / 3)
            # V_real, V_fft = constant_2D_potential(Nx, Ntheta_grid, 20, L, r)
            # V_real, V_fft = circular_quantum_dot_potential(x, theta, r, V0, V1, 100)
            # V_real, V_fft = smooth_circular_quantum_dot(x, theta, 75, 25, V1, V0, r, smoothing=0.1)
            # V_real, V_fft = lead_connecting_channel_potential(Nx, Ntheta_grid, x, theta, V0, V1, 10, 200, r)

        # Create geometry
        logger_main.trace(' Creating model...')
        logger_main.info('Potential shape: {}'.format(np.shape(V_real)))
        model_gauss = transport(vf=vf, L=L, rad=r, B_perp=B_perp, n_flux=n_flux, l_cutoff=l_cutoff)
        model_gauss.build_geometry(np.repeat(r, x.shape[0]), x, V_fft)

        # Conductance calculation
        logger_main.trace('Calculating conductance:')
        for j, E in enumerate(fermi):
            start_iter = time.time()
            G[n, j] = model_gauss.get_Landauer_conductance(E)
            logger_main.info(
                'iter sample: {}/{} | iter transport: {}/{} | iter time: {:.3e} s | Fermi level: {:.3f} | Conductance: {:.2f}'.format
                (n, N_samples - 1, j, fermi.shape[0] - 1, time.time() - start_iter, E, G[n, j]))

        # Store each disorder realisation
        V_fft_storage[n, :, :] = V_fft
        V_real_storage[n, :, :] = V_real

    # Scattering states and transmission eigenvalues at the resonant energies
    logger_main.info('Calculating scattering states...')
    trans_eigenvalues = model_gauss.get_transmission_eigenvalues(fermi[E_resonance_index])[0]
    scatt_states_up[0, :, :], scatt_states_down[0, :, :] = model_gauss.get_scattering_states(fermi[E_resonance_index], theta, initial_state=transmission_eigenval)
    scatt_density_up[0, :, :] = np.real(scatt_states_up[0, :, :] * scatt_states_up[0, :, :].conj())
    scatt_density_down[0, :, :] = np.real(scatt_states_down[0, :, :] * scatt_states_down[0, :, :].conj())

else:

    # Load data
    G = G_load
    V_fft, V_real = V_fft_load[0, :, :], V_real_load[0, :, :]
    V_real_storage, V_fft_storage = V_real_load, V_fft_load

    # Create instance of the transport class
    logger_main.trace('Creating model...')
    model_gauss = transport(vf=vf, L=L, rad=r, B_perp=B_perp, n_flux=n_flux, l_cutoff=l_cutoff)
    model_gauss.build_geometry(np.repeat(r, x.shape[0]), x, V_fft)

    # Scattering states and transmission eigenvalues at the resonant energies
    logger_main.info('Calculating scattering states...')
    trans_eigenvalues = model_gauss.get_transmission_eigenvalues(fermi[E_resonance_index])[0]
    scatt_states_up[0, :, :], scatt_states_down[0, :, :] = model_gauss.get_scattering_states(fermi[E_resonance_index], theta, initial_state=transmission_eigenval)
    scatt_density_up[0, :, :] = scatt_states_up[0, :, :] * scatt_states_up[0, :, :].conj()
    scatt_density_down[0, :, :] = scatt_states_down[0, :, :] * scatt_states_down[0, :, :].conj()

# Average conductance
G_avg = np.mean(G, axis=0)
time_lapse = time.time() - start_time
print(f'Time elapsed: {time_lapse:.2e}')

# %% Data storage
file_list = os.listdir('Data')
expID = get_fileID(file_list)
filename = '{}{}{}'.format('Experiment', expID, '.h5')
filepath = os.path.join('Data', filename)
if save_data:
    logger_main.info('Storing data...')
    with h5py.File(filepath, 'w') as f:
        store_my_data(f,                        "Conductance",             G)
        store_my_data(f,                        "Potential_FFT",           V_fft_storage)
        store_my_data(f,                        "Potential_xy",            V_real_storage)
        store_my_data(f,                        "scatt_states_up",         scatt_states_up)
        store_my_data(f,                        "scatt_states_down",       scatt_states_down)
        attr_my_data(f["Conductance"],          "Date",                    str(date.today()))
        attr_my_data(f["Conductance"],          "Code_path",               sys.argv[0])
        attr_my_data(f["Conductance"],          "vf",                      vf)
        attr_my_data(f["Conductance"],          "B_perp",                  B_perp)
        attr_my_data(f["Conductance"],          "n_flux",                  n_flux)
        attr_my_data(f["Conductance"],          "l_cutoff",                l_cutoff)
        attr_my_data(f["Conductance"],          "corr_length",             corr_length)
        attr_my_data(f["Conductance"],          "dis_strength",            dis_strength)
        attr_my_data(f["Conductance"],          "radius",                  r)
        attr_my_data(f["Conductance"],          "x0",                      x[0])
        attr_my_data(f["Conductance"],          "xf",                      x[-1])
        attr_my_data(f["Conductance"],          "Nx",                      Nx)
        attr_my_data(f["Conductance"],          "Ntheta_fft",              Ntheta_fft)
        attr_my_data(f["Conductance"],          "Ntheta_grid",             Ntheta_grid)
        attr_my_data(f["Conductance"],          "N_samples",               N_samples)
        attr_my_data(f["Conductance"],          "fermi0",                  fermi[0])
        attr_my_data(f["Conductance"],          "fermif",                  fermi[-1])
        attr_my_data(f["Conductance"],          "Nfermi",                  fermi.shape[0])
        attr_my_data(f["scatt_states_up"],      "resIndex",                E_resonance_index)


# %% Figures
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
axcolour = ['#FF7D66', '#FF416D', '#00B5A1', '#3F6CFF']
color_list = ['#FF7256', '#00BFFF', '#00C957', '#9A32CD', '#FFC125']
Vstd_th_1d = np.sqrt(dis_strength / np.sqrt(2 * pi)) * (vf / corr_length)  # Standard deviation 1d Gauss
Vstd_th_2d = np.sqrt(dis_strength / (2 * pi)) * vf / corr_length  # Standard deviation 2d Gauss

# Rotationally symmetric potential
if dimension=='1d':
    fig = plt.figure(figsize=(6, 8))
    gs = GridSpec(2, 6, figure=fig, wspace=0.5, hspace=0.5)
    ax31 = fig.add_subplot(gs[0, 0:2])
    ax32 = fig.add_subplot(gs[1, 0:2])
    ax33 = fig.add_subplot(gs[0, 2:4])
    ax34 = fig.add_subplot(gs[1, 2:4])
    ax35 = fig.add_subplot(gs[0, 4:])
    ax36 = fig.add_subplot(gs[1, 4:])

    # Conductance vs Fermi level
    ax31.plot(fermi, G_avg, color=color_list[1], label='$r=$ {} nm'.format(r))
    ax31.set_xlim(-200, 200)
    ax31.set_ylim(0, 12.5)
    ax31.set_xlabel("$E_F$ [meV]", fontsize=20)
    ax31.set_ylabel("$G[2e^2/h]$", fontsize=20)
    ax31.tick_params(which='major', width=0.75, labelsize=20)
    ax31.tick_params(which='major', length=14, labelsize=20)
    ax31.set_title(" Conductance: ExpID= {}, $L=$ {} nm, $r=$ {} nm".format(expID, x[-1], r))
    ax31.arrow(fermi[E_resonance_index], 3.5, 0, -1, width=0.4, head_length=0.1)
    # ax31.legend(loc='upper right', ncol=1, fontsize=20)

    # Conductance vs Fermi level (no background region)
    ax32.plot(fermi, G_avg, color=color_list[1], label='$r=$ {} nm'.format(r))
    ax32.set_xlim(-50, 50)
    ax32.set_ylim(0, 15)
    ax32.arrow(fermi[E_resonance_index], 3.5, 0, -1, width=0.4, head_length=0.1)
    ax32.set_xlabel("$E_F$ [meV]", fontsize=20)
    ax32.set_ylabel("$G[2e^2/h]$", fontsize=20)
    ax32.tick_params(which='major', width=0.75, labelsize=20)
    ax32.tick_params(which='major', length=14, labelsize=20)
    ax32.set_title("Conductance in the resonant region ")

    gap = vf / (2 * r)
    # ax32.plot((amplitude - gap) * np.ones((10, )), np.linspace(0, 15, 10), '--b')
    # ax32.plot((amplitude + gap) * np.ones((10,)), np.linspace(0, 15, 10), '--b')
    # ax32.plot((-amplitude - gap) * np.ones((10, )), np.linspace(0, 15, 10), '--b')
    # ax32.plot((-amplitude + gap) * np.ones((10,)), np.linspace(0, 15, 10), '--b')
    # ax32.plot((V1 - gap) * np.ones((10, )), np.linspace(0, 15, 10), '--b')
    # ax32.plot((V1 + gap) * np.ones((10,)), np.linspace(0, 15, 10), '--b')


    # Potential sample
    sample = 0
    ax33.plot(x, V_real_storage[sample, 0, :], color='#6495ED')
    ax33.set_xlim(x[0], x[-1])
    ax33.set_ylim(-4 * Vstd_th_1d, 4 * Vstd_th_1d)
    ax33.set_xlabel("$x$ [nm]", fontsize=20)
    ax33.set_ylabel('$V$ [meV]', fontsize=20)
    ax33.set(xticks=[0, int(L / 4), int(L / 2), int(3 * L / 4), L - 1], xticklabels=[0, int(L / 4), int(L / 2), int(3 * L / 4), L])
    ax33.tick_params(which='major', width=0.75, labelsize=15)
    ax33.tick_params(which='major', length=14, labelsize=15)
    ax33.set_title(" Gaussian correlated potential samples with $\\xi=$ {} nm, $K_V=$ {} and $N_x=$ {}".format
                                                                                      (corr_length, dis_strength, Nx))
    # peaks = np.arange(1, 20, 2) * period / 4
    # for pos in peaks:
    #     ax33.plot(pos * np.ones((10, )), np.linspace(-300, 300, 10), '--m')

    # Distribution of scattering states
    sample = 0
    # N_lead, N_trans, N_well = int(Nx * (L_lead / L)), int(Nx * (L_trans / L)), int(Nx * (L_well / L))
    check_imaginary(scatt_density_up[sample, :, :])
    density_plot = ax34.imshow(np.real(scatt_density_up[sample, :, :]) / np.max(np.real(scatt_density_up[sample, :, :])),
                                                         origin='lower', cmap='plasma', vmin=0, vmax=1, aspect='auto')
    divider34 = make_axes_locatable(ax34)
    cax34 = divider34.append_axes("right", size="5%", pad=0.05)
    cbar34 = fig.colorbar(density_plot, cax=cax34, orientation='vertical')
    cbar34.set_label(label='$\\vert \psi \\vert ^2$', labelpad=10, fontsize=20)
    cbar34.ax.tick_params(which='major', length=14, labelsize=15)
    ax34.set_xlabel("$x$ [nm]", fontsize=20)
    ax34.set_ylabel("$r\\theta$ [nm]", fontsize=20)
    ax34.set(xticks=[0, int(Nx / 4), int(Nx / 2), int(3 * Nx / 4), Nx - 1], xticklabels=[0, int(L / 4), int(L / 2), int(3 * L / 4), L])
    ax34.set(yticks=[0, int(Ntheta_plot / 2), Ntheta_plot - 1], yticklabels=[0, int(pi * r), int(2 * pi * r)])

    ax34.tick_params(which='major', width=0.75, labelsize=15)
    ax34.tick_params(which='major', length=14, labelsize=15)
    ax34.set_title(" Bound state density at energy $E=$ {:.2f} meV".format(fermi[E_resonance_index]))
    ax34.set_xlim(0, Nx)
    # ax34.plot(N_lead * np.ones((10, )), np.linspace(0, 300, 10), '--m')
    # ax34.plot((N_lead + N_trans) * np.ones((10,)), np.linspace(0, 300, 10), '--m')
    # ax34.plot((N_lead + N_trans + N_well) * np.ones((10,)), np.linspace(0, 300, 10), '--m')
    # ax34.plot((N_lead + 2 * N_trans + N_well) * np.ones((10,)), np.linspace(0, 300, 10), '--m')
    # peaks = np.arange(1, 20, 2) * period / 4
    # for pos in peaks:
    #     Npos = int(Nx * (pos / L))
    #     ax34.plot(Npos * np.ones((10, )), np.linspace(0, 300, 10), '--c')



    # Transmission eigenvalues
    ax35.plot(np.arange(len(trans_eigenvalues)), np.sort(trans_eigenvalues), 'o', color=color_list[2])
    ax35.plot(len(trans_eigenvalues) - transmission_eigenval - 1,
              np.sort(trans_eigenvalues)[len(trans_eigenvalues) - transmission_eigenval - 1], 'o', color='red')
    ax35.set_yscale('log')
    ax35.yaxis.set_label_position("right")
    ax35.yaxis.tick_right()
    ax35.set_ylim(10e-16, 10)
    ax35.set_xlabel("Transmission eigenvalues", fontsize=20)
    ax35.set_ylabel("eig$(t^\dagger t)$", fontsize=20)
    ax35.tick_params(which='major', width=0.75, labelsize=20)
    ax35.tick_params(which='major', length=14, labelsize=20)
    ax35.set_title('Transmission eigenvalues for $E=$ {:.2f} meV'.format(fermi[E_resonance_index]), fontsize=20)

    # Distribution of scattering states in logscale
    sample = 0
    check_imaginary(scatt_density_up[sample, :, :])
    density_plot = ax36.imshow(
        np.log(np.real(scatt_density_up[sample, :, :]) / np.max(np.real(scatt_density_up[sample, :, :]))),
        origin='lower', cmap='plasma', aspect='auto', vmax=0, vmin=-8)
    divider36 = make_axes_locatable(ax36)
    cax36 = divider36.append_axes("right", size="5%", pad=0.05)
    cbar36 = fig.colorbar(density_plot, cax=cax36, orientation='vertical')
    cbar36.set_label(label='$log \\vert \psi \\vert ^2$', labelpad=10, fontsize=20)
    cbar36.ax.tick_params(which='major', length=14, labelsize=15)
    ax36.set_xlabel("$x$ [nm]", fontsize=20)
    # ax36.set_ylabel("$r\\theta$ [nm]", fontsize=20)
    ax36.set(xticks=[0, int(Nx / 4), int(Nx / 2), int(3 * Nx / 4), Nx - 1],
             xticklabels=[0, int(L / 4), int(L / 2), int(3 * L / 4), L])
    ax36.set(yticks=[])
    ax36.tick_params(which='major', width=0.75, labelsize=15)
    ax36.tick_params(which='major', length=14, labelsize=15)
    ax36.set_title(" Bound state density at energy $E=$ {:.2f} meV".format(fermi[E_resonance_index]))

    plt.show()


# Broken rotational symmetry
else:

    fig = plt.figure(figsize=(6, 8))
    gs = GridSpec(2, 6, figure=fig, wspace=0.5, hspace=0.5)
    ax31 = fig.add_subplot(gs[0, 0:2])
    ax32 = fig.add_subplot(gs[1, 0:2])
    ax33 = fig.add_subplot(gs[0, 2:4])
    ax34 = fig.add_subplot(gs[1, 2:4])
    ax35 = fig.add_subplot(gs[0, 4:])
    ax36 = fig.add_subplot(gs[1, 4:])

    # Conductance vs Fermi level
    ax31.plot(fermi, G_avg, color=color_list[1], label='$r=$ {} nm'.format(r))
    ax31.set_xlim(-200, 200)
    ax31.set_ylim(0, 12.5)
    ax31.set_xlabel("$E_F$ [meV]", fontsize=20)
    ax31.set_ylabel("$G[2e^2/h]$", fontsize=20)
    ax31.tick_params(which='major', width=0.75, labelsize=20)
    ax31.tick_params(which='major', length=14, labelsize=20)
    ax31.set_title(" Conductance: ExpID= {}, $L=$ {} nm, $r=$ {} nm".format(expID, x[-1], r))
    ax31.arrow(fermi[E_resonance_index], 3.5, 0, -1, width=0.4, head_length=0.1)
    # ax31.legend(loc='upper right', ncol=1, fontsize=20)

    # Conductance vs Fermi level (no background region)
    ax32.plot(fermi, G_avg, color=color_list[1], label='$r=$ {} nm'.format(r))
    ax32.set_xlim(-50, 50)
    ax32.set_ylim(0, 15)
    ax32.arrow(fermi[E_resonance_index], 3.5, 0, -1, width=0.4, head_length=0.1)
    # ax32.plot(np.repeat(2 * Vstd_th_2d, 10), np.linspace(0, 10, 10), '--', color='#A9A9A9', alpha=0.5)
    # ax32.plot(np.repeat(-2 * Vstd_th_2d, 10), np.linspace(0, 10, 10), '--', color='#A9A9A9', alpha=0.5)
    # ax32.text(2 * Vstd_th_2d - 10, 3, '$2\sigma$', fontsize=20, rotation='vertical', color='#A9A9A9', alpha=0.5)
    # ax32.text(- 2 * Vstd_th_2d + 3, 3, '$2\sigma$', fontsize=20, rotation='vertical', color='#A9A9A9', alpha=0.5)
    ax32.set_xlabel("$E_F$ [meV]", fontsize=20)
    ax32.set_ylabel("$G[2e^2/h]$", fontsize=20)
    ax32.tick_params(which='major', width=0.75, labelsize=20)
    ax32.tick_params(which='major', length=14, labelsize=20)
    ax32.set_title("Conductance in the resonant region ")
    ax32.legend(loc='upper right', ncol=1, fontsize=20)

    # Potential sample
    sample = 0
    auxV = V_real_storage[sample, :, :] + np.abs(np.min(V_real_storage[sample, :, :]))
    density_plot = ax33.imshow(np.log(auxV) / np.max(auxV), origin='lower', aspect='auto')
    divider33 = make_axes_locatable(ax33)
    cax33 = divider33.append_axes("right", size="5%", pad=0.05)
    cbar33 = fig.colorbar(density_plot, cax=cax33, orientation='vertical')
    cbar33.set_label(label='$V$ [meV]', labelpad=-5, fontsize=20)
    cbar33.ax.tick_params(which='major', length=14, labelsize=15)
    ax33.set_xlabel("$x$ [nm]", fontsize=20)
    ax33.set_ylabel("$r\\theta$ [nm]", fontsize=20)
    ax33.set(xticks=[0, int(Nx / 4), int(Nx / 2), int(3 * Nx / 4), Nx - 1],
             xticklabels=[0, int(L / 4), int(L / 2), int(3 * L / 4), L])
    ax33.set(yticks=[0, int(Ntheta_grid / 2), Ntheta_grid - 1], yticklabels=[0, int(pi * r), int(2 * pi * r)])
    ax33.tick_params(which='major', width=0.75, labelsize=15)
    ax33.tick_params(which='major', length=14, labelsize=15)
    ax33.set_title(
        " Gaussian correlated potential samples with $\\xi=$ {} nm, $K_V=$ {} and $N_x=$ {}".format(corr_length,
                                                                                                    dis_strength, Nx))

    gap = vf / (2 * r)
    # contours = ax33.contour(V_real_storage[sample, :, :], levels=[fermi[E_resonance_index] - 2 * gap,
    #                                      fermi[E_resonance_index] + 2 * gap], colors="black", linewidths=2)

    # Distribution of scattering states
    sample = 0
    check_imaginary(scatt_density_up[sample, :, :])
    density_plot = ax34.imshow(
        np.real(scatt_density_up[sample, :, :]) / np.max(np.real(scatt_density_up[sample, :, :])),
        cmap='plasma', vmin=0, vmax=1, aspect='auto')
    divider34 = make_axes_locatable(ax34)
    cax34 = divider34.append_axes("right", size="5%", pad=0.05)
    cbar34 = fig.colorbar(density_plot, cax=cax34, orientation='vertical')
    cbar34.set_label(label='$\\vert \psi \\vert ^2$', labelpad=10, fontsize=20)
    cbar34.ax.tick_params(which='major', length=14, labelsize=15)
    ax34.set_xlabel("$x$ [nm]", fontsize=20)
    ax34.set_ylabel("$r\\theta$ [nm]", fontsize=20)
    ax34.set(xticks=[0, int(Nx / 4), int(Nx / 2), int(3 * Nx / 4), Nx - 1],
             xticklabels=[0, int(L / 4), int(L / 2), int(3 * L / 4), L])
    ax34.set(yticks=[0, int(Ntheta_grid / 2), Ntheta_grid - 1], yticklabels=[0, int(pi * r), int(2 * pi * r)])
    ax34.tick_params(which='major', width=0.75, labelsize=15)
    ax34.tick_params(which='major', length=14, labelsize=15)
    ax34.set_title(" Bound state density at energy $E=$ {:.2f} meV".format(fermi[E_resonance_index]))
    gap = vf / (2 * r)
    # contours = ax34.contour(V_real_storage[sample, :, :], levels=[fermi[E_resonance_index] - 2 * gap,
    #                                                               fermi[E_resonance_index] + 2 * gap], colors="black",
    #                         linewidths=2)

    # Transmission eigenvalues
    ax35.plot(np.arange(len(trans_eigenvalues)), np.sort(trans_eigenvalues), 'o', color=color_list[2])
    ax35.plot(len(trans_eigenvalues) - transmission_eigenval - 1,
              np.sort(trans_eigenvalues)[len(trans_eigenvalues) - transmission_eigenval - 1], 'o', color='red')
    ax35.set_yscale('log')
    ax35.set_ylim(10e-16, 10)
    ax35.yaxis.set_label_position("right")
    ax35.yaxis.tick_right()
    ax35.set_xlabel("Transmission eigenvalues", fontsize=20)
    ax35.set_ylabel("eig$(t^\dagger t)$", fontsize=20)
    ax35.tick_params(which='major', width=0.75, labelsize=20)
    ax35.tick_params(which='major', length=14, labelsize=20)
    ax35.set_title('Transmission eigenvalues for $E=$ {:.2f} meV'.format(fermi[E_resonance_index]), fontsize=20)


    # Distribution of scattering states in logscale
    sample = 0
    check_imaginary(scatt_density_up[sample, :, :])
    density_plot = ax36.imshow(np.log(np.real(scatt_density_up[sample, :, :]) / np.max(np.real(scatt_density_up[sample, :, :]))),
                                                                        cmap='plasma', aspect='auto', vmax=0, vmin=-8)
    divider36 = make_axes_locatable(ax36)
    cax36 = divider36.append_axes("right", size="5%", pad=0.05)
    cbar36 = fig.colorbar(density_plot, cax=cax36, orientation='vertical')
    cbar36.set_label(label='$log \\vert \psi \\vert ^2$', labelpad=10, fontsize=20)
    cbar36.ax.tick_params(which='major', length=14, labelsize=15)
    ax36.set_xlabel("$x$ [nm]", fontsize=20)
    # ax36.set_ylabel("$r\\theta$ [nm]", fontsize=20)
    ax36.set(xticks=[0, int(Nx / 4), int(Nx / 2), int(3 * Nx / 4), Nx - 1],
             xticklabels=[0, int(L / 4), int(L / 2), int(3 * L / 4), L])
    ax36.set(yticks=[])
    ax36.tick_params(which='major', width=0.75, labelsize=15)
    ax36.tick_params(which='major', length=14, labelsize=15)
    ax36.set_title(" Bound state density at energy $E=$ {:.2f} meV".format(fermi[E_resonance_index]))

    plt.show()

