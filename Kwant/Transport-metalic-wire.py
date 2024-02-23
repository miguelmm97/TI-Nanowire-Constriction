"""
Creating a metalic nanowire with Kwant. Part of the tutorial at the website
https://kwant-project.org/doc/1/tutorial/first_steps
"""
import kwant
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
import scipy.sparse.linalg as sla


#%% Parameters and set up of the system
"""
Note about units: Usually we set a = t = 1. This means that all energies that we calculate will be given in units
of t. Conductance calculations become dimensionless in the process as usual and are given in terms of e²/h. If we want to
go back to physical units we just need to multiply the results with the unit energy factor that we get when discretising.
"""


hbar = 4.13e-15                                 # hbar in eV s
a = 1                                           # Lattice constant in nm
c = 3e8 * 1e9                                   # Speed of light in nm/s
m = 0.5e6 / c ** 2                              # Electron mass in eV
unit_E = hbar**2 / (2 * m * a**2)               # Unit energy for the problem (passes every E to meV)
hopping = 4.                                    # Hopping amplitudes (in units of discrete factor)
width = 30                                      # Width of the scattering region in nm
L = 100                                         # Length of the scattering region in nm
mu_leads = -0                                   # Chemical potential of the leads (in units of discrete factor)
fermi = np.linspace(0, 2, 100)                  # Fermi energy for conductance calculation (in units of discrete factor)
G = np.zeros(fermi.shape)                       # Conductance array






#%% Scattering region
"""
Kwant always uses a tight binding model. If we have a continuous Hamiltonian, we have to discretise it. This can be done
by approximating the different differential operators as hopping and onsite terms. The system object takes points in the
lattice and connects them through the hopping amplitudes given (these can also be matrices if there are inner degrees of
freedom). Automatically Kwant takes into account hermiticity so that we only need to specify the hoppings along one 
direction.
"""

syst = kwant.Builder()              # Builds a system class
lat = kwant.lattice.square(a)       # Creates a lattice for the tight binding model

# Chemical potential
def step(x1, x2, sigma=None):
    # Smoothened step function theta(x1-x2)
    if sigma is None:
        return np.heaviside(x1 - x2, 1)
    else:
        return 0.5 + (1 / pi) * np.arctan(sigma * (x1 - x2))
def geom_nc(x, x1, x2, r1, r2, sigma=None):
    # x1, r1: Initial widest part
    # x2, r2: Final narrow part
    return r1 + (r2 - r1) * step(x, x2, sigma) + ((r2 - r1) / (x2 - x1)) * (x - x1) * (
            step(x, x1, sigma) - step(x, x2, sigma))
def geom_cons(x, x1, x2, x3, r1, r2, sigma):
    # x1, r1: Initial lead
    # x2, r2: Constriction
    # x3: Start of the second cone
    return geom_nc(x, x1, x2, r1, r2, sigma) + geom_nc(-x + x2 + x3, x1, x2, r1, r2, sigma) - r2
mu = geom_cons(np.arange(0, 100), 0, 20, 80, mu_leads, 0, sigma=0.1)

for i in range(L):
    for j in range(width):
        syst[lat(i, j)] = 4 * hopping + geom_cons(i, 0, 20, 80, mu_leads, 0, sigma=0.1)
        if j > 0: syst[lat(i, j), lat(i, j - 1)] = - hopping    # Hopping along y
        if i > 0: syst[lat(i, j), lat(i - 1, j)] = - hopping    # Hopping along x

#%% Leads
"""
In order to do transport calculations we need to attach leads to the lattice. The way to do so is to specify that certain
regions of the system are translation invariant. After this, we give the same structure (or other if the leads are different
from the scattering region) to the hopping between sites. In particular we only need to define the hopping from the 0th
site to the 1st site of our lattice in the direction of translation invariance.

Physics note: As they are by default in this construction, the leads are just merely the same system as the scattering
region but with infinite translation invariance. This means that we won't get Fabry-Perot oscillations in the conductance
because there will be perfect transmission from the leads to the scattering region for every mode (since they are the 
same system essentially). In order to see these oscillations we have to calibrate the onsite energy of the leads so that
there are many modes (bands) in the regions where we calculate conductance. Note that this comes with an increase of w, 
and hence it is a more expensive computation. Furthermore, these oscillations become smoother the smoother the change of 
onsite energy is between the leads and the scattering region.
"""
# Left lead
sym_left_lead = kwant.TranslationalSymmetry((-a, 0))   # Indicates translational symmetry
left_lead = kwant.Builder(sym_left_lead)               # Creates the lead

for j in range(width):
    left_lead[lat(0, j)] = 4 * hopping + mu_leads
    left_lead[lat(1, j), lat(0, j)] = - hopping
    if j > 0: left_lead[lat(0, j), lat(0, j - 1)] = - hopping

syst.attach_lead(left_lead)                            # Attaches the lead to the scattering region

# Right lead
sym_right_lead = kwant.TranslationalSymmetry((a, 0))
right_lead = kwant.Builder(sym_right_lead)

for j in range(width):
    right_lead[lat(0, j)] = 4 * hopping + mu_leads
    right_lead[lat(1, j), lat(0, j)] = - hopping
    if j > 0:
        right_lead[lat(0, j), lat(0, j - 1)] = - hopping

syst.attach_lead(right_lead)                           # Attaches the lead to the scattering region
syst = syst.finalized()                                # Locks the system to perform calculations on it


#%% Transport calculations
"""
Once the system is initialised we can start performing calculations. We can specify the direction in which we calculate
conductance, namely if we calculate G from t or t' in the scattering matrix, that is from lead 0 to 1 or the other way
around.

Physics note: Note that we work with a tight binding model, that means we will have as many transversal modes, and hence
bands, as sites in y direction (set by w). As a result, the conductance will reach a point where it will start decreasing
monotonically, exactly when the first band is fully occupied as there are no more upper bands at some point. 
This is an artifact of the tight binding model, because bands will only be linear/ parabolic near E=0, then they start 
combing until they become flat, which makes them able to be fully occupied.
"""

for i, E in enumerate(fermi):
    S = kwant.smatrix(syst, E)     # Scattering matrix at energy E
    G[i] = S.transmission(1, 0)    # Conductance from lead 0 to lead 1
    print('iter transport: {}/{} | Fermi level: {:.3f} | Conductance: {:.2f}'.format(i, fermi.shape[0] - 1, E, G[i]))


#%% Band structure
"""
In order to calculate band structures we can just define a translation invariant system the same way as we treat the 
leads for transport calculations. We just need to initialise the object directly without attaching it to any previous
scattering region.
"""
trans_inv = kwant.TranslationalSymmetry((a, 0))
closed_syst= kwant.Builder(trans_inv)

for j in range(width):
    closed_syst[lat(0, j)] = 4 * hopping
    closed_syst[lat(1, j), lat(0, j)] = - hopping
    if j > 0: closed_syst[lat(0, j), lat(0, j - 1)] = - hopping

closed_syst = closed_syst.finalized()
temp1 = kwant.physics.Bands(closed_syst)
momenta = np.linspace(-np.pi, np.pi, 101)
temp2 = [temp1(k) for k in momenta]
energy_bands = np.zeros((width, 101))
for i, E in enumerate(temp2): energy_bands[:, i] = E

# Get the eigenstates
H = closed_syst.hamiltonian_submatrix(sparse=True)
evals, evecs = sla.eigsh(H.tocsc(), k=5)

#%% Local density of states
local_dos = kwant.ldos(syst, energy=0.4)


#%% Figures

# System
fig0, ax0 = plt.subplots(figsize=(8, 2))
kwant.plot(syst, ax=ax0, show=False)

# Conductance
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(fermi * unit_E, G, color='#00BFFF')
ax1.set_xlabel("$E$ [eV]")
ax1.set_ylabel("$G$ [$e²/h$]")
ax1.set_xlim(0, fermi[-1] * unit_E)
ax1.set_ylim(0, G[-1])
ax1.set_title('Conductance vs Fermi level')

# Bands
fig2, ax2 = plt.subplots(figsize=(8, 6))
for i in range(energy_bands.shape[0] - 1):
    ax2.plot(momenta, energy_bands[i, :] * unit_E, color='#00BFFF')
ax2.plot(momenta, -mu_leads * np.ones(momenta.shape), '--', color='#9A32CD', alpha=0.5)
ax2.set_xlabel("$k$ [$a$]")
ax2.set_ylabel("$E$ [eV]")
ax2.set_xlim(-np.pi, np.pi)
ax2.set_ylim(0, 8 * unit_E)
ax2.set_title('Band structure')
ax2.text(-3, -mu_leads + 0.3, '$\mu_{\\text{leads}}$', fontsize=20, color='#A9A9A9', alpha=0.5)

# Local density of states
fig3, ax3 = plt.subplots(figsize=(8, 6))
kwant.plotter.map(syst, local_dos, num_lead_cells=10, ax=ax3, show=False)
ax3.set_title('Local density of states')

# Chemical potential
fig4, ax4 = plt.subplots(figsize=(8, 6))
ax4.plot(np.arange(L), mu * unit_E, color='#9A32CD')
ax4.set_xlabel("$x$ [nm]")
ax4.set_ylabel("$\\mu$ [eV]")
ax4.set_xlim(0, L)
# ax4.set_ylim(0, mu_leads + 0.2)
ax4.set_title('Chemical potential')

# Eigenstates
# fig5, ax5 = plt.subplots(figsize=(8, 6))
# kwant.plotter.map(closed_syst, np.abs(evecs[:, 0])**2, colorbar=False, oversampling=1, ax=ax5, show=False)
# ax5.set_title('Eigenstates')

plt.show()