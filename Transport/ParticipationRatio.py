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

# Managing logging
import logging
import colorlog
from colorlog import ColoredFormatter

# Tracking time
import time
from datetime import date

# External modules
from TransportClass import transport

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


#%% Loading data

# Importing global variables from toml.file
variables = config.variables
for key in variables:
    globals().update(variables[key])

# Loading data
outdir = "Data"
if load_data_IPR:
    logger_main.info('Loading data from {}'.format(load_file_IPR))
    for file in os.listdir(outdir):
        if file == load_file_IPR:
            file_path = os.path.join(outdir, file)
            with h5py.File(file_path, 'r') as f:
                if file.startswith('IPR'):
                    variable_setup     = False
                    IPR_up             = f['IPR_up'][()]
                    IPR_down           = f['IPR_down'][()]
                    V_real_load        = f['Potential_xy'][()]
                    V_fft_load         = f['Potential_FFT'][()]
                    res_index          = f['res_index'][()]
                    delta_x            = f['delta_x'][()]
                    delta_theta        = f['delta_theta'][()]
                    fermi              = f['fermi'][()]
                    x                  = f['x'][()]
                    theta              = f['theta'][()]
                    scatt_states_up    = f['scatt_states_up'][()]
                    scatt_states_down  = f['scatt_states_down'][()]

                else:
                    variable_setup     = True
                    V_real_load        = f['Potential_xy'][()]
                    V_fft_load         = f['Potential_FFT'][()]

# %% Initialise variables
if variable_setup:

    # Global variables set up in this script
    delta_x           = np.array([L, L / 2, L / 3, L / 4])
    delta_theta       = np.array([2 * pi, 2 * pi / 2, 2 * pi / 3, 2 * pi / 4])
    res_index         = np.array([100])  #, 2000, 3000, 4000, 5453, 5542, 5599, 5620, 5672, 6107, 6584, 6619, 6542, 9000, 10000, 10772])

    # Global variables set up at the .toml file
    x                 = np.linspace(0, L, Nx)                                                                # Discretised position
    Ntheta_grid       = Ntheta_fft if dimension=='2d' else 1                                                 # Grid for theta
    Ntheta_plot       = Ntheta_fft if dimension=='2d' else default_Ntheta_plot                               # Grid for plotting theta
    theta             = np.linspace(0, 2 * pi, Ntheta_plot)                                                  # Discretised angles
    fermi             = np.linspace(fermi_0, fermi_end, fermi_length)                                        # Fermi energy sample
    scatt_states_up   = np.zeros((len(res_index), len(delta_x), Ntheta_plot, Nx - 1), dtype=np.complex128)   # Scattering states storage
    scatt_states_down = np.zeros((len(res_index), len(delta_x), Ntheta_plot, Nx - 1), dtype=np.complex128)   # Scattering states storage
    IPR_up            = np.zeros((len(res_index), len(delta_x)))
    IPR_down          = np.zeros((len(res_index), len(delta_x)))

Vstd_th_2d        = np.sqrt(dis_strength / (2 * pi)) * vf / corr_length


#%% Main

# Load/create potential landscape
logger_main.trace('Generating potential...')
if load_data_IPR:
    try:
        V_fft, V_real = V_fft_load[0, :, :], V_real_load[0, :, :]
    except:
        V_fft, V_real = V_fft_load, V_real_load
elif dimension=='1d':
    V_real = gaussian_correlated_potential_1D_FFT(L, Nx, dis_strength, corr_length, vf)
    V_fft = V_real
else:
    V_fft, V_real = gaussian_correlated_potential_2D_FFT(L, r, Nx, Ntheta_fft, dis_strength, corr_length, vf)
logger_main.info('Potential shape: {}'.format(np.shape(V_real)))


# Calculate participation ratio
if calculate_IPR:
    logger_main.trace(' Creating model...')
    model_gauss = transport(vf=vf, L=L, rad=r, B_perp=B_perp, n_flux=n_flux, l_cutoff=l_cutoff)
    model_gauss.build_geometry(np.repeat(r, x.shape[0]), x, V_fft)

    logger_main.info('Calculating scattering states...')
    for i, (size_x, size_theta) in enumerate(zip(delta_x, delta_theta)):
        for j, Ef_index in enumerate(res_index):
            scatt_states_up[j, i, :, :], scatt_states_down[j, i, :, :] = model_gauss.get_scattering_states(fermi[Ef_index], theta, initial_state=transmission_eigenval)
            IPR_up[j, i] = model_gauss.get_participation_ratio(scatt_states_up[j, i, :, :], x, theta, delta_x=size_x, delta_theta=size_theta)
            IPR_down[j, i] = model_gauss.get_participation_ratio(scatt_states_down[j, i, :, :], x, theta, delta_x=size_x, delta_theta=size_theta)
            logger_main.info('size sample: {}/{} | iter sample: {}/{} | IPR_up: {} | IPR_down: {}'.format(i, len(delta_x) - 1, j, len(res_index) - 1, IPR_up[j, i], IPR_down[j, i]))


#%% Saving data
file_list = os.listdir('Data')
expID = get_fileID(file_list, common_name='IPR')
filename = '{}{}{}'.format('IPR', expID, '.h5')
filepath = os.path.join('Data', filename)
if save_data_IPR:
    logger_main.info('Storing data...')
    with h5py.File(filepath, 'w') as f:
        store_my_data(f,                   "IPR_up",                IPR_up)
        store_my_data(f,                   "IPR_down",              IPR_down)
        store_my_data(f,                   "Potential_FFT",         V_fft)
        store_my_data(f,                   "Potential_xy",          V_real)
        store_my_data(f,                   "delta_x",               delta_x)
        store_my_data(f,                   "delta_theta",           delta_theta)
        store_my_data(f,                   "x",                     x)
        store_my_data(f,                   "theta",                 theta)
        store_my_data(f,                   "res_index",             res_index)
        store_my_data(f,                   "scatt_states_up",       scatt_states_up)
        store_my_data(f,                   "scatt_states_down",     scatt_states_down)
        store_my_data(f,                   "fermi",                 fermi)
        attr_my_data(f["IPR_up"],          "Date",                  str(date.today()))
        attr_my_data(f["IPR_up"],          "Code_path",             sys.argv[0])
        attr_my_data(f["IPR_up"],          "load_file",             load_file_IPR)

#%% Figures
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
axcolour = ['#FF7D66', '#FF416D', '#00B5A1', '#3F6CFF']
color_list = ['#FF7256', '#00BFFF', '#00C957', '#9A32CD', '#FFC125', '#3F6CFF']


# Inverse participation ratio vs fermi level
fig1 = plt.figure(figsize=(5, 4))
gs = GridSpec(1, 1, figure=fig1)
ax1 = fig1.add_subplot(gs[0, 0])
for i in range(len(delta_x)):
    ax1.plot(fermi[res_index], IPR_up[:, i], '.', color=color_list[i], label='$(\delta x, \delta_\\theta)=$ {}, {:.2f}'.format(delta_x[i], delta_theta[i]))
ax1.set_xlim(np.min(fermi), np.max(fermi))
ax1.set_ylim(1e-6, 1)
ax1.set_yscale('log')
ax1.set_xlabel("$E_F$ [meV]", fontsize=10)
ax1.set_ylabel("Inverse participation ratio",fontsize=10)
ax1.tick_params(which='major', width=0.75, labelsize=10)
ax1.tick_params(which='major', length=6, labelsize=10)
ax1.legend(loc='best')

ax1.plot(np.repeat(2 * Vstd_th_2d, 10), np.linspace(0, 10, 10), '--', color='#00B5A1', alpha=0.5)
ax1.plot(np.repeat(-2 * Vstd_th_2d, 10), np.linspace(0, 10, 10), '--', color='#00B5A1', alpha=0.5)
ax1.text(2 * Vstd_th_2d - 10, 0.5, '$2\sigma$', fontsize=10, rotation='vertical', color='#00B5A1', alpha=0.5)
ax1.text(- 2 * Vstd_th_2d + 3, 0.5, '$2\sigma$', fontsize=10, rotation='vertical', color='#00B5A1', alpha=0.5)
plt.show()



# Inverse participation ratio vs L
fig2 = plt.figure(figsize=(5, 4))
gs = GridSpec(1, 1, figure=fig2)
ax2 = fig2.add_subplot(gs[0, 0])
ax2.plot(delta_x, IPR_up[0, :], '.', color=color_list[0], label='$E_f=$ {:.2f} [meV]'.format(fermi[res_index[0]]))
# ax2.plot(delta_x, IPR_up[4, :], '.', color=color_list[1], label='$E_f=$ {:.2f} [meV]'.format(fermi[res_index[4]]))
# ax2.plot(delta_x, IPR_up[7, :], '.', color=color_list[2], label='$E_f=$ {:.2f} [meV]'.format(fermi[res_index[7]]))
# ax2.plot(delta_x, IPR_up[11, :], '.', color=color_list[3], label='$E_f=$ {:.2f} [meV]'.format(fermi[res_index[11]]))
# ax2.plot(delta_x, IPR_up[-2, :], '.', color=color_list[4], label='$E_f=$ {:.2f} [meV]'.format(fermi[res_index[-2]]))
# ax2.plot(delta_x, IPR_up[-1, :], '.', color=color_list[5], label='$E_f=$ {:.2f} [meV]'.format(fermi[res_index[-1]]))
ax2.set_xlim(np.min(delta_x) - 100, np.max(delta_x) + 100)
ax2.set_ylim(1e-6, 1)
ax2.set_yscale('log')
ax2.set_xlabel("$L$ [nm]", fontsize=10)
ax2.set_ylabel("Inverse participation ratio",fontsize=10)
ax2.tick_params(which='major', width=0.75, labelsize=10)
ax2.tick_params(which='major', length=6, labelsize=10)
ax2.legend(loc='best')
plt.show()
