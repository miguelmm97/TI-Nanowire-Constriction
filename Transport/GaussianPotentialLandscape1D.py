import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from TransportClass import transport, gaussian_correlated_potential_1D, get_fileID
import time
import h5py
import os
import sys
from datetime import date

start_time = time.time()
#%% Parameters

# Constants and set up of the model
phi0         = 2 * pi * 1e-34 / 1.6e-19                                 # Quantum of flux
vf           = 330                                                      # Fermi velocity in meV nm
B_perp       = 0                                                        # Perpendicular magnetic field in T
n_flux       = 0.0                                                      # Number of flux quanta threaded
B_par        = 0  # n_flux * phi0 / (120 * 20 * 1e-9 * 1e-9)            # Parallel magnetic field in T
l_cutoff     = 30                                                       # Cutoff number modes
corr_length  = 10                                                       # Correlation length in nm
dis_strength = 6                                                        # Disorder strength in vf / xi scale
radius       = np.array([19])                                           # Radius of the nanowire
x            = np.linspace(0, 500, 4)                                   # Discretised position
Nq           = 100                                                      # Number of points to take the FFT
N_samples    = 2                                                        # Number of samples to disorder average
fermi        = np.linspace(-100, 100, 2)                                # Fermi energy sample
g_index      = ['N_samples', 'fermi', 'radius']                         # Flag for saving G
test_run    = True if fermi.shape[0] < 10 else False                    # Flag that points to test runs or real runs
G            = np.zeros((N_samples, fermi.shape[0], radius.shape[0]))
Vstd_th = np.sqrt((dis_strength / (corr_length * np.sqrt(2 * pi))) * (vf / corr_length) ** 2)
#%% Calculations

# Transport calculation
for i, rad in enumerate(radius):
    r = np.repeat(rad, x.shape[0])

    for n in range(N_samples):

        # Instance of the transport class
        model_gauss = transport(vf, B_perp, B_par, l_cutoff)
        V = gaussian_correlated_potential_1D(x, dis_strength, corr_length, vf, Nq)[0]
        model_gauss.build_geometry(r, x, V)

        # Conductance calculation
        for j, E in enumerate(fermi):
            start_iter = time.time()
            G[n, j, i] = model_gauss.get_Landauer_conductance(E)
            print('iter radius: {}/{} | iter sample: {}/{} | iter transport: {}/{} | iter time: {:.3e} s'.format
                  (i, radius.shape[0]-1, n, N_samples-1, j, fermi.shape[0]-1, time.time() - start_iter))


G_avg = np.mean(G, axis=0)


#%% Data storage
file_list = os.listdir('Data')
expID = get_fileID(file_list)
filename ='{}{}{}'.format('Experiment', expID, '.h5')
filepath = os.path.join('Data', filename)
with h5py.File(filepath, 'w') as f:
    f.create_dataset("data", data=G)
    f["data"].attrs.create("Date",                 data=str(date.today()))
    f["data"].attrs.create("Codepath",             data=sys.argv[0])
    f["data"].attrs.create("vf",                   data=vf)
    f["data"].attrs.create("B_perp",               data=B_perp)
    f["data"].attrs.create("n_flux",               data=n_flux)
    f["data"].attrs.create("B_par",                data=B_par)
    f["data"].attrs.create("l_cutoff",             data=l_cutoff)
    f["data"].attrs.create("corr_length",          data=corr_length)
    f["data"].attrs.create("dis_strength",         data=dis_strength)
    f["data"].attrs.create("radius",               data=radius)
    f["data"].attrs.create("x",                    data=x)
    f["data"].attrs.create("Nq",                   data=Nq)
    f["data"].attrs.create("N_samples",            data=N_samples)
    f["data"].attrs.create("fermi",                data=fermi)
    f["data"].attrs.create("fermi0",               data=fermi[0])
    f["data"].attrs.create("fermi-1",              data=fermi[-1])
    f["data"].attrs.create("fermi_resolution",     data=fermi.shape[0])
    f["data"].attrs.create("Gshape",               data=g_index)
    f["data"].attrs.create("Test_run",             data=test_run)


#%% Figures
color_list = ['#FF7256', '#00BFFF', '#00C957', '#9A32CD', '#FFC125']
fig, ax1 = plt.subplots(figsize=(8, 6))
for i in range(radius.shape[0]):
    ax1.plot(fermi, G_avg[:, i], color=color_list[i], label='$r=$ {} nm'.format(radius[i]))
ax1.plot(np.repeat(2 * Vstd_th, 10), np.linspace(0, 10, 10), '--', color='#A9A9A9', alpha=0.5)
ax1.plot(np.repeat(-2 * Vstd_th, 10), np.linspace(0, 10, 10), '--', color='#A9A9A9', alpha=0.5)
ax1.text(2 * Vstd_th - 10, 9, '$2\sigma$', fontsize=20, rotation='vertical', color='#A9A9A9', alpha=0.5)
ax1.text(- 2 * Vstd_th + 3, 9, '$2\sigma$', fontsize=20, rotation='vertical', color='#A9A9A9', alpha=0.5)
ax1.set_xlim(min(fermi), max(fermi))
ax1.set_ylim(0, 10)
ax1.set_xlabel("$E_F$ [meV]")
ax1.set_ylabel("$G[2e^2/h]$")
ax1.set_title(" Gaussian correlated: ExpID= {}, $\\xi=$ {} nm, $N_q=$ {}, $N_s=$ {}, $L=$ {} nm, $K_v=$ {}".format(expID, corr_length, Nq, N_samples, x[-1], dis_strength))
ax1.legend(loc='upper right', ncol=1)
plt.show()
