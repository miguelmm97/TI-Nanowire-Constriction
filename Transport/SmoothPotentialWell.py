import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from ..PackageFolder.TransportClass import transport, geom_cons
import time

start_time = time.time()
#%% Parameters

# Constants and set up of the model
phi0      = 2 * pi * 1e-34 / 1.6e-19                                 # Quantum of flux
vf        = 330                                                      # Fermi velocity in meV nm
B_perp    = 0                                                        # Perpendicular magnetic field in T
n_flux    = 0.0                                                      # Number of flux quanta threaded
B_par     = 0  # n_flux * phi0 / (120 * 20 * 1e-9 * 1e-9)            # Parallel magnetic field in T
l_cutoff  = 30                                                       # Cutoff number modes


# Geometry

L = 50
Npoints = 200
dx = L / Npoints                                                  # Disorder amplitude

Vmax = 50
Vmin = 0
L_well = 20
x0 = 0
x_barrier1 = x0 + 10
x_barrier2 = x_barrier1 + L_well
xf = x_barrier2 + 10
x = np.linspace(x0, xf, Npoints)
V = geom_cons(x, x0, x_barrier1, x_barrier2, Vmax, Vmin, 10)



fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(x, V, color='#FF7256')
ax1.set_xlim(x0, xf)
ax1.set_ylim(Vmin - 10, Vmax + 10)
ax1.set_xlabel("$L$ [nm]")
ax1.set_ylabel("$V$ [meV]")
plt.plot()
