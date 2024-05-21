#%% Modules setup

# Math and plotting
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn

# Managing system, data and config files
import h5py
import os
import sys
import config
from functions import store_my_data, attr_my_data, get_fileID
from dataclasses import fields

# Managing logging
import logging
import colorlog
from colorlog import ColoredFormatter

# Tracking time
from datetime import date

# External modules
import functions
from MyStaticVariables import MyStaticVariables
from MyStatusVariablesIPR import MyStatusVariablesIPR
from Nanostructure import Nanostructure, gaussian_correlated_potential_2D_FFT

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
sim = MyStatusVariablesIPR(msv)
sim = sim.pop_vars()

# Loading potential landscape or generating it
logger_main.trace('Generating potential...')
outdir = "Data"
if msv.load_data_IPR:
    logger_main.info('Loading data from {}'.format(msv.load_file_IPR))
    for file in os.listdir(outdir):
        if file == msv.load_file_IPR:
            file_path = os.path.join(outdir, file)
            with h5py.File(file_path, 'r') as f:
                V_real = f['Simulation/Potential_xy'][()][0, :, :]
                V_fft  = f['Simulation/Potential_FFT'][()][0, :, :]
                logger_main.info('Potential shape: {}'.format(np.shape(V_real)))
else:
    if msv.dimension=='1d':
        V_real = gaussian_correlated_potential_1D_FFT(msv.L, msv.Nx, msv.dis_strength, msv.corr_length, msv.vf)
        V_fft = V_real
        logger_main.info('Potential shape: {}'.format(np.shape(V_real)))
    else:
        V_fft, V_real = sym_potential_well_2D(msv.Nx, msv.Ntheta_fft, 0, 0, msv.L, msv.r)
        # V_fft, V_real = gaussian_correlated_potential_2D_FFT(msv.L, msv.r, msv.Nx, msv.Ntheta_fft, msv.dis_strength, msv.corr_length, msv.vf)
        logger_main.info('Potential shape: {}'.format(np.shape(V_real)))

#%% Main

# Calculate participation ratio
if msv.calculate_IPR:
    logger_main.info('Performing IPR calculation...')

    # Create model
    logger_main.trace(' Creating model...')
    nanowire = Nanostructure(L=msv.L, vf=msv.vf, rad=msv.r, B_perp=msv.B_perp, n_flux=msv.n_flux, l_cutoff=msv.l_cutoff,
                                r_vec=sim.r_vec, x_vec=sim.x, theta_vec=sim.theta, V_vec=V_fft)

    logger_main.info('Calculating scattering states...')
    for i, (size_x, size_theta) in enumerate(zip(sim.delta_x, sim.delta_theta)):
        for j, (Ef_index, x_start, theta_start) in enumerate(zip(msv.res_index, msv.x0, msv.theta0)):

            # Get scattering states for each resonance
            sim.scatt_states_up[j, i, :, :], sim.scatt_states_down[j, i, :, :] = nanowire.get_scattering_states(sim.fermi[Ef_index])

            # Get inverse participation ratio
            sim.IPR_up[j, i] = nanowire.get_participation_ratio(sim.scatt_states_up[j, i, :, :],
                                                delta_x=size_x, delta_theta=size_theta, x0=x_start, theta0=theta_start)
            sim.IPR_down[j, i] = nanowire.get_participation_ratio(sim.scatt_states_down[j, i, :, :],
                                                delta_x=size_x, delta_theta=size_theta, x0=x_start, theta0=theta_start)
            logger_main.info('size sample: {}/{} | iter sample: {}/{} | IPR_up: {} | IPR_down: {}'.format(i,
                                len(sim.delta_x) - 1, j, len(msv.res_index) - 1, sim.IPR_up[j, i], sim.IPR_down[j, i]))

# Fits to the IPR curves
def funcfit(y, C1, C2): return C1 / y ** C2
for i in range(len(msv.res_index)):
    fit_params, _ = curve_fit(funcfit, sim.delta_x * sim.delta_theta * msv.r, sim.IPR_up[i, :])
    sim.fit_params1[i], sim.fit_params2[i] = fit_params[0], fit_params[1]

#%% Saving data
file_list = os.listdir('Data')
expID = get_fileID(file_list, common_name='IPR')
filename = '{}{}{}'.format('IPR', expID, '.h5')
filepath = os.path.join('Data', filename)

if msv.save_data_IPR:
    logger_main.info('Storing data...')

    with h5py.File(filepath, 'w') as f:
        # Simulation folder
        simulation = f.create_group('Simulation')
        store_my_data(simulation, "V_real", V_real)
        store_my_data(simulation, "V_fft", V_fft)
        for field in fields(sim):
            store_my_data(simulation, field.name, getattr(sim, field.name))

        # Parameters folder
        parameters = f.create_group('Parameters')
        for field in fields(msv):
            store_my_data(parameters, field.name, getattr(msv, field.name))

        # Attributes
        attr_my_data(simulation, "Date",       str(date.today()))
        attr_my_data(simulation, "Code_path",  sys.argv[0])
        attr_my_data(simulation, "load_file",  msv.load_file_IPR)


