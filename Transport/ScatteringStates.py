#%% Modules setup

# Math and plotting
import numpy as np

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
from MyStatusVariablesScatt import MyStatusVariablesScatt
from Nanostructure import Nanostructure, gaussian_correlated_potential_2D_FFT, gaussian_correlated_potential_1D_FFT

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

# Setting up dataclasses for variables
msv = MyStaticVariables()
msv = msv.pop_vars(config.variables)

if msv.load_data_scatt:
    logger_main.info('Loading data from {}'.format(msv.load_file_scatt))
    file_path = os.path.join('Data', msv.load_file_scatt)

    # Load static variables and create status variables
    with h5py.File(file_path, 'r') as f:
        not_load = ['Nx', 'l_cutoff', 'E_resonance_index', 'transmission_eigenval', 'load_pot_scatt']
        msv.load_data_to_static_var(file_path, not_load_var=not_load)
        sim = MyStatusVariablesScatt(msv)
        sim = sim.pop_vars()

        # Load specific potential
        if msv.load_pot_scatt is None:
            load_V = f['Simulation']['FT_matrix'][()]
        else:
            file_path_pot = os.path.join('Data', msv.load_pot_scatt)
            with h5py.File(file_path_pot, 'r') as f2:
                logger_main.info('Loading potential from {}'.format(msv.load_pot_scatt))
                load_V = f2['Simulation']['FT_matrix'][()]

        sim.V_fft, sim.V_real, sim.FT_matrix = gaussian_correlated_potential_2D_FFT(msv.L, msv.r, msv.Nx, msv.Ntheta_fft,
                                                                                    msv.dis_strength, msv.corr_length,
                                                                                    msv.vf, from_potential=load_V)
else:
    # Status variables
    sim = MyStatusVariablesScatt(msv)
    sim = sim.pop_vars()

    # Generating potential
    logger_main.info('Generating potential...')
    if msv.dimension_scatt == '1d':
        sim.V_real = gaussian_correlated_potential_1D_FFT(msv.L, msv.Nx, msv.dis_strength, msv.corr_length, msv.vf)
        sim.V_fft = sim.V_real
    else:
        sim.V_fft, sim.V_real, sim.FT_matrix = gaussian_correlated_potential_2D_FFT(msv.L, msv.r, msv.Nx, msv.Ntheta_fft,
                                                                                    msv.dis_strength, msv.corr_length, msv.vf)

logger_main.info('Potential shape: {}'.format(np.shape(sim.V_real)))

# %% Scattering states Calculation
start_time = time.time()

# Create model
logger_main.trace(' Creating model...')
nw = Nanostructure(L=msv.L, vf=msv.vf, rad=msv.r, B_perp=msv.B_perp, n_flux=msv.n_flux, l_cutoff=msv.l_cutoff,
                         r_vec=sim.r_vec, x_vec=sim.x, theta_vec=sim.theta, V_vec=sim.V_fft)


# Scattering states and transmission eigenvalues at the resonant energies
logger_main.info('Calculating scattering states...')
sim.trans_eigenvalues = nw.get_transmission_eigenvalues(sim.fermi[msv.E_resonance_index])[0]
sim.scatt_states_up, sim.scatt_states_down = nw.get_scattering_states(sim.fermi[msv.E_resonance_index], initial_state=msv.transmission_eigenval)
sim.scatt_density_up = sim.scatt_states_up * sim.scatt_states_up.conj()
sim.scatt_density_down = sim.scatt_states_down * sim.scatt_states_down.conj()

time_lapse = time.time() - start_time
print(f'Time elapsed: {time_lapse:.2e}')


#%% Saving data
file_list = os.listdir('Data')
expID = get_fileID(file_list, common_name='Experiment')
filename = '{}{}{}'.format('Experiment', expID, '.h5')
filepath = os.path.join('Data', filename)

if msv.save_data_scatt:
    logger_main.info('Storing data...')
    with h5py.File(filepath, 'w') as f:

        # Simulation folder
        simulation = f.create_group('Simulation')
        for field in fields(sim):
            try:
                store_my_data(simulation, field.name, getattr(sim, field.name))
            except AttributeError as err:
                logger_main.warning(f'Certain field in sim could not be stored: {err}')

        # Parameters folder
        parameters = f.create_group('Parameters')
        for field in fields(msv):
            try:
                store_my_data(parameters, field.name, getattr(msv, field.name))
            except AttributeError:
                logger_main.warning(f'Certain field in msv could not be stored: {err}')

        # Attributes
        attr_my_data(parameters, "Date",       str(date.today()))
        attr_my_data(parameters, "Code_path",  sys.argv[0])
        attr_my_data(parameters, "load_file",  msv.load_file_IPR)
