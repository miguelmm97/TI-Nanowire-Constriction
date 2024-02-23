import kwant
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
import scipy.sparse.linalg as sla
import tinyarray
from Functions import geom_cons, get_bands

"""
We build a kwant model for a topological insulator nanowire in the simplest case, that is no variation on the radius,
no magnetic field and no external potential.
"""

#%% Parameters and set up of the system

vf = 330                                            # Fermi velocity in meV nm
a = 1                                               # Lattice constant in nm
unit_E = vf / (2 * a)                               # Unit energy for the problem (passes every E to meV)
hopping = 1.                                        # Hopping amplitudes (in units of discrete factor)
width = 50                                          # Width of the scattering region in nm
L = 100                                             # Length of the scattering region in nm
mu_leads = -0                                       # Chemical potential of the leads (in units of discrete factor)
fermi = np.linspace(0, 2, 100)                      # Fermi energy for conductance calculation (in units of discrete factor)
G = np.zeros(fermi.shape)                           # Conductance array

sigma_0 = tinyarray.array([[1, 0], [0, 1]])         # Pauli 0
sigma_x = tinyarray.array([[0, 1], [1, 0]])         # Pauli x
sigma_y = tinyarray.array([[0, -1j], [1j, 0]])      # Pauli y
sigma_z = tinyarray.array([[1, 0], [0, -1]])        # Pauli z

#%% Scattering region and leads
"""
In the tight binding approximation used by kwant, the Dirac Hamiltonian becomes just simple hopping terms in both 
directions, with the energy scale set by unit_E. The anti-periodic boundary conditions due to the spin rotation are
implemented by an additional hopping from j=0 to j=width.
"""

syst = kwant.Builder()
lat = kwant.lattice.square(a)

# Chemical potential
mu = geom_cons(np.arange(0, 100), 0, 20, 80, mu_leads, 0, sigma=0.1)

# Scattering region
for i in range(L):
    for j in range(width):
        syst[lat(i, j)] = geom_cons(i, 0, 20, 80, mu_leads, 0, sigma=0.1) * sigma_0
        if j > 0: syst[lat(i, j), lat(i, j - 1)] = - 1j * hopping * sigma_y
        if i > 0: syst[lat(i, j), lat(i - 1, j)] = - 1j * hopping * sigma_x

    # Anti-periodic conditions
    syst[lat(i, 0), lat(i, width - 1)] = 1j * hopping * sigma_y

# Left lead
sym_left_lead = kwant.TranslationalSymmetry((-a, 0))
left_lead = kwant.Builder(sym_left_lead)
for j in range(width):
    left_lead[lat(0, j)] = mu_leads * sigma_0
    left_lead[lat(1, j), lat(0, j)] = - 1j * hopping * sigma_x
    if j > 0: left_lead[lat(0, j), lat(0, j - 1)] = - 1j * hopping * sigma_y
left_lead[lat(0, 0), lat(0, width - 1)] = 1j * hopping * sigma_y
syst.attach_lead(left_lead)

# Right lead
sym_right_lead = kwant.TranslationalSymmetry((a, 0))
right_lead = kwant.Builder(sym_right_lead)
for j in range(width):
    right_lead[lat(0, j)] = mu_leads * sigma_0
    right_lead[lat(1, j), lat(0, j)] = - 1j * hopping * sigma_x
    if j > 0: right_lead[lat(0, j), lat(0, j - 1)] = - 1j * hopping * sigma_y
right_lead[lat(0, 0), lat(0, width - 1)] = 1j * hopping * sigma_y
syst.attach_lead(right_lead)
syst = syst.finalized()




#%% Conductance

for i, E in enumerate(fermi):
    S = kwant.smatrix(syst, E)
    G[i] = 0.25 * S.transmission(1, 0)
    print('iter transport: {}/{} | Fermi level: {:.3f} | Conductance: {:.2f}'.format(i, fermi.shape[0] - 1, E, G[i]))

#%% Band structure

trans_inv = kwant.TranslationalSymmetry((a, 0))
closed_syst= kwant.Builder(trans_inv)

for j in range(width):
    closed_syst[lat(0, j)] = mu_leads * sigma_0
    closed_syst[lat(1, j), lat(0, j)] = - 1j * hopping * sigma_x
    if j > 0: closed_syst[lat(0, j), lat(0, j - 1)] = - 1j * hopping * sigma_y
closed_syst[lat(0, 0), lat(0, width - 1)] = 1j * hopping * sigma_y

closed_syst = closed_syst.finalized()
temp1 = kwant.physics.Bands(closed_syst)
momenta = np.linspace(-np.pi, np.pi, 101)
temp2 = [temp1(k) for k in momenta]
energy_bands = np.zeros((width * 2, 101))
for i, E in enumerate(temp2): energy_bands[:, i] = E

k_exact, E_exact =  get_bands(width * a, 0, 0, vf)
#%% Figures

# System
fig0, ax0 = plt.subplots(figsize=(8, 2))
kwant.plot(syst, show=False, ax=ax0)


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
    if i == 0: label1, label2 = 'TB approx.', 'Surface theory'
    else: label1, label2 = None, None
    ax2.plot(momenta, energy_bands[i, :] * unit_E, color='#00BFFF', label=label1)
    ax2.plot(k_exact, E_exact[i, :], color='#3F6CFF', label=label2)
ax2.plot(momenta, -mu_leads * np.ones(momenta.shape), '--', color='#9A32CD', alpha=0.5)
ax2.set_xlabel("$k$ [$a$]")
ax2.set_ylabel("$E$ [eV]")
ax2.set_xlim(-np.pi, np.pi)
ax2.set_ylim(0, 3 * unit_E)
ax2.set_title('Band structure')
ax2.text(-3, -mu_leads + 0.3, '$\mu_{\\text{leads}}$', fontsize=20, color='#A9A9A9', alpha=0.5)
ax2.legend(loc='best', ncol=1, fontsize=15)




# Chemical potential
fig4, ax4 = plt.subplots(figsize=(8, 6))
ax4.plot(np.arange(L), mu * unit_E, color='#9A32CD')
ax4.set_xlabel("$x$ [nm]")
ax4.set_ylabel("$\\mu$ [eV]")
ax4.set_xlim(0, L)
# ax4.set_ylim(0, mu_leads + 0.2)
ax4.set_title('Chemical potential')

plt.show()