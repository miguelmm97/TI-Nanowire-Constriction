import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from numpy import pi
from functions import spectrum, Ham_bulk_Bi2Se3, Ham_bulk_LowEnergy_Bi2Se3

# %%  Global definitions

# Parameters of the model
a = 10                                          # Lattice constant in Å
flux = 0.0                                      # Flux through the cross-section in units of the flux quantum
kx = np.linspace(-0.4, 0.4, 300)                # kx in 1 / Å
ky = np.linspace(-0.4, 0.4, 300)                # ky in 1 / Å
kz = np.linspace(-0.4, 0.4, 300)                # kz in 1 / Å

# Parameters
A1 = 2.2e3                                      # meV Å
A2 = 4.1e3                                      # meV Å
B1 = 10e3                                       # meV Å^2
B2 = 56.6e3                                     # meV Å^2
D1 = 1.3e3                                      # meV Å^2
D2 = 19.6e3                                     # meV Å^2
C = - 6.8                                       # meV
M = 280                                         # meV

# Definitions
Econt = np.zeros((4, len(kx)))
Elatt = np.zeros((4, len(kx)))

# %% Diagonalisation

# Bulk band structure low energy theory
for j in range(len(kx)):
    print(str(j) + "/" + str(len(kx)))
    Hcont = Ham_bulk_LowEnergy_Bi2Se3(kx[j], ky[j], 0, C, M, D1, D2, B1, B2, A1, A2)
    Econt[:, j] = spectrum(Hcont)[0]
    Hlatt = Ham_bulk_Bi2Se3(kx[j], ky[j], 0, C, M, D1, D2, B1, B2, A1, A2, a)
    Elatt[:, j] = spectrum(Hlatt)[0]



# %% Figures


fig1 = plt.figure()
for j in range(4):
    plt.plot(kx * np.sqrt(2), Econt[j, :], 'b')
    plt.plot(kx * np.sqrt(2), Elatt[j, :], '--r')
# Axis labels and limits
plt.ylabel("$E$[meV]", fontsize=15)
plt.xlabel("$\\vert k \\vert[1/Å]$", fontsize=15)
plt.ylim(-2000, 2000)
plt.xlim(-0.4, 0.4)
plt.show()


