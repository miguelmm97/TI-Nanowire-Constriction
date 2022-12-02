import matplotlib.pyplot as plt
import numpy as np
from functions import spectrum, Ham_ThinFilm_Bi2Se3_bulkXY, xtranslation

# %%  Global definitions

# Parameters of the model
a = 10                                  # Lattice constant in Å
flux = 0.0                              # Flux through the cross-section in units of the flux quantum
kx = np.linspace(-0.4, 0.4, 300)      # kx in 1 / Å
ky = np.linspace(-0.4, 0.4, 300)      # ky in 1 / Å

# Parameters
A1 = 2.2                                # eV Å
A2 = 4.1                                # eV Å
B1 = 10                                 # eV Å^2
B2 = 56.6                               # eV Å^2
D1 = 1.3                                # eV Å^2
D2 = 19.6                               # eV Å^2
C = - 6.8e-3                            # eV
M = 0.28                                # eV

# Lattice definition
L_z = 100                                # In units of a (average bond length)
n_states = L_z * 4                      # Number of basis states
z = np.arange(0, L_z)                   # Array with the number of each site

# Definitions
E = np.zeros((n_states, len(kx)))

# %% Diagonalisation

# Band structure
for j in range(len(kx)):
    print(str(j) + "/" + str(len(kx)))
    H = Ham_ThinFilm_Bi2Se3_bulkXY(L_z, z, kx[j], ky[j], A1, A2, B1, B2, C, D1, D2, M, a)
    E[:, j] = spectrum(H)[0]


# %% Figures

# Band Structure
for j in range(n_states):
    plt.plot(np.sqrt(2) * kx, E[j, :], 'b')
# Axis labels and limits
plt.ylabel("$E$[eV]", fontsize=15)
plt.xlabel("$k_{\perp}[1/Å]$", fontsize=15)
# plt.ylim(-2, 2)
plt.xlim(-0.4, 0.4)
plt.title("Bi$_2$Se$_3$ thin film, $L_z=$" + str(L_z) + " nm")
plt.show()


