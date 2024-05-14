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
from functions import store_my_data, attr_my_data, get_fileID
from dataclasses import fields

# Tracking time
import time
from datetime import date

# Managing logging
import logging
import colorlog
from colorlog import ColoredFormatter

# External modules
from MyStaticVariables import MyStaticVariables
from MyStatusVariablesTransport import MyStatusVariablesTransport
from Nanostructure import Nanostructure, gaussian_correlated_potential_2D_FFT, constant_2D_potential, transport_mode
from functions import check_imaginary

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


#%% Variables

# Static variables (should not be edited)
msv = MyStaticVariables()
msv = msv.pop_vars(config.variables)

# Status variables (can be accessed and edited)
sim = MyStatusVariablesTransport(msv)
sim = sim.pop_vars()

# Loading data
outdir = "Data"
if msv.load_data_transport:
    logger_main.info('Loading data from {}'.format(msv.load_file_transport))

    for file in os.listdir(outdir):
        if file == msv.load_file_transport:
            file_path = os.path.join(outdir, file)
            with h5py.File(file_path, 'r') as f:

                # Loading parameters
                try:
                    msv.load_data_to_var(file_path)
                except KeyError:
                    logger_main.warning('Unable to load parameters from {} into msv due to old storage format.'.format
                                        (msv.load_file_transport))

                # Loading previous simulation results
                V_real, V_fft, sim.G   = f['Potential_xy'][()], f['Potential_FFT'][()], f['Conductance'][()]
                if len(V_real.shape) > 2:
                    V_real, V_fft = V_real[0, :, :], V_fft[0, :, :]
                if len(sim.G.shape) > 1:
                    sim.G  = sim.G[0, :]
                logger_main.info('Potential shape: {}'.format(np.shape(V_real)))

else:
    logger_main.info('Generating potential...')
    if msv.dimension == '1d':
        V_real = gaussian_correlated_potential_1D_FFT(msv.L, msv.Nx, msv.dis_strength, msv.corr_length, msv.vf)
        V_fft = V_real
        logger_main.info('Potential shape: {}'.format(np.shape(V_real)))
    else:
        V_fft, V_real = constant_2D_potential(msv.Nx, msv.Ntheta_fft, 0, msv.L, msv.r)
        # V_fft, V_real = gaussian_correlated_potential_2D_FFT(msv.L, msv.r, msv.Nx, msv.Ntheta_fft, msv.dis_strength, msv.corr_length, msv.vf)
        logger_main.info('Potential shape: {}'.format(np.shape(V_real)))

# %% Transport Calculation
start_time = time.time()

# Create model
logger_main.trace(' Creating model...')
nw = Nanostructure(L=msv.L, vf=msv.vf, rad=msv.r, B_perp=msv.B_perp, n_flux=msv.n_flux, l_cutoff=msv.l_cutoff,
                         r_vec=sim.r_vec, x_vec=sim.x, theta_vec=sim.theta, V_vec=V_fft)

# Transport calculation
if msv.calculate_transport:
    logger_main.info('Performing transport calculation...')

    # Conductance calculation
    logger_main.trace('Calculating conductance:')
    for i, E in enumerate(sim.fermi):
        start_iter = time.time()
        sim.G[i] = nw.get_Landauer_conductance(E)
        logger_main.info('iter transport: {}/{} | iter time: {:.3e} s | Fermi level: {:.3f} | G: {:.2f}'.format
                         (i, sim.fermi.shape[0] - 1, time.time() - start_iter, E, sim.G[i]))

# Scattering states and transmission eigenvalues at the resonant energies
logger_main.info('Calculating scattering states...')
sim.trans_eigenvalues = nw.get_transmission_eigenvalues(sim.fermi[msv.E_resonance_index])[0]
sim.scatt_states_up, sim.scatt_states_down = nw.get_scattering_states(sim.fermi[msv.E_resonance_index],
                                                               initial_state=msv.transmission_eigenval)
sim.scatt_density_up = sim.scatt_states_up * sim.scatt_states_up.conj()
sim.scatt_density_down = sim.scatt_states_down * sim.scatt_states_down.conj()

time_lapse = time.time() - start_time
print(f'Time elapsed: {time_lapse:.2e}')

#%% Current analysis
n = 0
scatt_up_analytic   = np.zeros((nw.theta_vec.shape[0], nw.x_vec.shape[0]), dtype=np.complex128)
scatt_down_analytic = np.zeros((nw.theta_vec.shape[0], nw.x_vec.shape[0]), dtype=np.complex128)

for i, x in enumerate(nw.x_vec):
    for j, theta in enumerate(nw.theta_vec):
        scatt_up_analytic[j, i] = transport_mode(x, theta, nw.rad, n, sim.fermi[0], nw.vf)
        scatt_down_analytic[j, i] = transport_mode(x, theta, nw.rad, n, sim.fermi[0], nw.vf, spin='down')

# norm = np.sqrt(np.sum(scatt_up_analytic * scatt_up_analytic.conj() + scatt_down_analytic * scatt_down_analytic.conj()))

dens_up_analytic   = scatt_up_analytic * scatt_up_analytic.conj()
dens_down_analytic = scatt_down_analytic * scatt_down_analytic.conj()

print(dens_up_analytic[1, 1], 1 / (2 * pi * msv.r))
print(sim.scatt_density_up[1, 1] + sim.scatt_density_down[1, 1])


check_imaginary(dens_up_analytic)
check_imaginary(sim.scatt_density_up)
dens_up_analytic = np.real(dens_up_analytic)
dens_up_num = np.real(sim.scatt_density_up)
print('hey')

fig1 = plt.figure(figsize=(5, 4))
gs = GridSpec(1, 1, figure=fig1)
ax34 = fig1.add_subplot(gs[0, 0])


density_plot = ax34.imshow(np.real(dens_up_analytic) / np.max(np.real(dens_up_analytic)),
                                                     origin='lower', cmap='plasma', vmin=0, vmax=1, aspect='auto')
divider34 = make_axes_locatable(ax34)
cax34 = divider34.append_axes("right", size="5%", pad=0.05)
cbar34 = fig1.colorbar(density_plot, cax=cax34, orientation='vertical')
cbar34.set_label(label='$\\vert \psi \\vert ^2$', labelpad=10, fontsize=20)
cbar34.ax.tick_params(which='major', length=14, labelsize=15)
ax34.set_xlabel("$x$ [nm]", fontsize=20)
ax34.set_ylabel("$r\\theta$ [nm]", fontsize=20)
ax34.set_xlim(0, msv.Nx)



#%% Saving data
file_list = os.listdir('Data')
expID = get_fileID(file_list, common_name='Experiment')
filename = '{}{}{}'.format('Experiment', expID, '.h5')
filepath = os.path.join('Data', filename)

if msv.save_data_transport:
    logger_main.info('Storing data...')
    with h5py.File(filepath, 'w') as f:
        # Simulation folder
        simulation = f.create_group('Simulation')
        store_my_data(simulation, "Potential_xy", V_real)
        store_my_data(simulation, "Potential_FFT", V_fft)
        for field in fields(sim):
            store_my_data(simulation, field.name, getattr(sim, field.name))

        # Parameters folder
        parameters = f.create_group('Parameters')
        for field in fields(msv):
            store_my_data(parameters, field.name, getattr(msv, field.name))

        # Attributes
        attr_my_data(parameters, "Date",       str(date.today()))
        attr_my_data(parameters, "Code_path",  sys.argv[0])
        attr_my_data(parameters, "load_file",  msv.load_file_IPR)

plt.show()