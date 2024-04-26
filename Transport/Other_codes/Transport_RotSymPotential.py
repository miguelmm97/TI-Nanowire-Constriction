import numpy as np
import argparse
import time
import random
import matplotlib.pyplot as plt
from numpy import pi
from Nanostructure import transport


parser = argparse.ArgumentParser(description='Argument parser for Transport_RotSymPotential.py')

# Constants of the model
parser.add_argument('-vf',       '--fermi_velocity',              type=float,    default=330,       help='Fermi velocity in meV nm')
parser.add_argument('-Bperp',    '--perpendicular_field',         type=float,    default=0,         help='Perpendicular magnetic field in T')
parser.add_argument('-Bpar',     '--parallel_field',              type=float,    default=0,         help='Parallel magnetic field in T')
parser.add_argument('-flux',     '--flux_quanta',                 type=int,      default=0,         help='Number of flux quanta threaded')
parser.add_argument('-lcut',     '--angular_momentum_cutoff',     type=int,      default=30,        help='Cutoff on the number of angular momentum modes')

# Set up of the geometry
parser.add_argument('-L',        '--total_length',                type=float,    default=500,       help='Total length of the nanostructure in nm')
parser.add_argument('-Nreg',     '--number_regions',              type=int,      default=10,        help='Number of regions in the nanostructure')
parser.add_argument('-Vmax',     '--maximum_potential',           type=float,    default=0,         help='Maximum possible value of the potential')

# Known potential landscape
parser.add_argument('-V',        '--potential',                   type=float,    default=0,         help='Potential for the different regions',             action='append', nargs='+')

# Random Potential Landscape
parser.add_argument('-disV',     '--disorder_potential',          type=bool,     default=False,     help='Implements disorder in the potential')
parser.add_argument('-disL',     '--disorder_length',             type=bool,     default=False,     help='Implements disorder in the region length')

# Radii and confinement gaps
parser.add_argument('-Ncalc',    '--number_calculations',         type=int,      default=1,         help='Number of different transport calculations' )
parser.add_argument('-rmin',     '--minimum_radius',              type=float,    default=8,         help='Minimum radius considered in nm')
parser.add_argument('-rmax',     '--minimum_radius',              type=float,    default=8,         help='Maximum radius considered in nm')
parser.add_argument('-Egmin',    '--minimum_gap',                 type=float,    default=20,        help='Minimum confinement gap considered in meV')
parser.add_argument('-Egmax',    '--maximum_gap',                 type=float,    default=20,        help='Maximum confinement gap considered in meV')

# Transport calculation
parser.add_argument('-Emin',     '--minimum_fermi',               type=float,    default=-100,      help='Initial Fermi energy')
parser.add_argument('-Emax',     '--maximum_fermi',               type=float,    default=100,       help='Final Fermi energy')


args = parser.parse_args()


start_time = time.time()
#%% Parameters

# Constants and set up of the model
phi0      = 2 * pi * 1e-34 / 1.6e-19                                 # Quantum of flux
vf        = 330                                                      # Fermi velocity in meV nm
B_perp    = 0                                                        # Perpendicular magnetic field in T
n_flux    = 0.0                                                      # Number of flux quanta threaded
B_par     = 0  # n_flux * phi0 / (120 * 20 * 1e-9 * 1e-9)            # Parallel magnetic field in T
l_cutoff  = 30                                                       # Cutoff number modes


# Random potential landscape parameters
L         = 500                                                      # Total length of the nanostructure
Nregions  = 50                                                       # Number of regions in the nanostructure
Vmax      = 60                                                       # Maximum possible value of the random potential
fermi     = np.linspace(-100, 100, 1000)                            # Range of Fermi energies for the conductance

#%% Transport calculation

# Case with disorder
Ldis = np.random.uniform(0, 1, size=Nregions)
Ldis = L * Ldis / np.sum(Ldis)
Vdis = np.random.uniform(0, Vmax, size=Nregions)

# Case without disorder
# Ldis = (L/Nregions) * np.ones((Nregions, ))
# Vdis = Vmax * np.tile(np.array([1, 0]), int(Nregions/2))

# Radius and confinement gap
radius = np.linspace(8, 30, 5)
# conf_gap = np.linspace(8, 30, 5)
# radius = vf / (2 * conf_gap)
conf_gap = vf / (2 * radius)
G = np.zeros((radius.shape[0], fermi.shape[0]))


# Transport calculation
for i, r in enumerate(radius):
    # Instance of the transport class
    l0 = 0
    random_model = transport(vf, B_perp, B_par, l_cutoff)
    for l, V in zip(Ldis, Vdis):
        random_model.add_nw(l0, l0 + l, r=r, Vnm=V * np.eye(2 * l_cutoff + 1))
        l0 += l

    # Conductance calculation
    for j, E in enumerate(fermi):
        start_iter = time.time()
        G[i, j] = random_model.get_Landauer_conductance(E)
        print('iter: {}/{} {}/{}| time: {:.3e} s | G: {:.2e}'.format(i, G.shape[0], j, fermi.shape[0], time.time() - start_iter, G[i, j]))

print('Time elapsed: {:.2e} s'.format(time.time() - start_time))


#%% Figures
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 13, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=15)
color_list = ['#FF7256', '#00BFFF', '#00C957', '#9A32CD', '#FFC125']


# Conductance as a function of the fermi level
fig1, ax1 = plt.subplots(figsize=(8, 6))
for i in range(G.shape[0]):
    hex_color = ["#" + ''.join([random.choice('ABCDEF0123456789') for j in range(6)])]
    ax1.plot(fermi, G[i, :], color=color_list[i], label='$r$ = {} nm, $E_g=$ {0:.1f} meV'.format(radius[i], conf_gap[i]))
ax1.set_xlim(min(fermi), max(fermi))
ax1.set_ylim(0, 10)
ax1.set_xlabel("$E_F$ [meV]")
ax1.set_ylabel("$G[2e^2/h]$")
ax1.set_title('$N=$ {}, $L=$ {} nm, $V\in(0,${}$)$ meV'.format(Nregions, L, Vmax))
ax1.legend(loc='upper right', ncol=1)
plt.show()

# Potential landscape
# l0, V0 = 0, 0
# fig2, ax2 = plt.subplots(figsize=(8, 6))
# for l, V in zip(Ldis, Vdis):
#     ax2.plot(np.linspace(l0, l0 + l, 10), V * np.ones((10, )), '--', color='#6495ED')
#     ax2.plot(l0 * np.ones((10, )), np.linspace(V0, V, 10), '--', color='#6495ED')
#     ax2.plot(np.linspace(l0, l0 + l, 10), (V + conf_gap/2) * np.ones((10,)), '-', color="#00CD66")
#     ax2.plot(l0 * np.ones((10,)), np.linspace((V0 + conf_gap/2), (V + conf_gap/2), 10), '-', color="#00CD66")
#     ax2.plot(np.linspace(l0, l0 + l, 10), (V - conf_gap/2) * np.ones((10,)), '-', color="#00CD66")
#     ax2.plot(l0 * np.ones((10,)), np.linspace((V0 - conf_gap/2), (V - conf_gap/2), 10), '-', color="#00CD66")
#     ax2.fill_between(np.linspace(l0, l0 + l, 10), (V - conf_gap/2) * np.ones((10, )), y2 = -conf_gap, color="#00CD66", alpha=0.5)
#     ax2.fill_between(np.linspace(l0, l0 + l, 10), (V + conf_gap / 2) * np.ones((10,)), y2=Vmax + conf_gap, color="#00CD66", alpha=0.5)
#     l0, V0 = l0 + l, V
#
# ax2.set_xlim(0, L)
# ax2.set_ylim(-conf_gap, Vmax + conf_gap)
# ax2.set_xlabel("$L$ [nm]")
# ax2.set_ylabel("$V$ [meV]")
# ax2.set_title('Random potential landscape')
# plt.show()
