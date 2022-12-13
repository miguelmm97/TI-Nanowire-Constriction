import matplotlib.pyplot as plt
import numpy as np
from functions import spectrum, Ham_bulk_Bi2Se3, Ham_bulk_LowEnergy_Bi2Se3, Ham_bulk_FB3dTI

# %%  Global definitions

# Parameters of the model
a = 10                                       # Lattice constant in Å
flux = 0.0                                   # Flux through the cross-section in units of the flux quantum
kx = np.linspace(-0.4, 0.4, 300)             # kx in 1 / Å
ky = np.linspace(-0.4, 0.4, 300)             # ky in 1 / Å
kz = np.linspace(-0.4, 0.4, 300)             # kz in 1 / Å

# Parameters
A1 = 2.2                                     # eV Å
A2 = 4.1                                     # eV Å
B1 = 10                                      # eV Å^2
B2 = 56.6                                    # eV Å^2
D1 = 1.3                                     # eV Å^2
D2 = 19.6                                    # eV Å^2
C = - 6.8e-3                                 # eV
M = 0.28                                     # eV
lamb = 0.15                                  # eV
eps = 4 * lamb                               # eV
lamb_z = 2 * lamb                            # eV
t = lamb                                     # eV

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


font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 13, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=15)


fig1 = plt.figure()
for j in range(4):
    plt.plot(np.sqrt(2) * kx, Econt[j, :], '--r')
    plt.plot(np.sqrt(2) * kz, Elatt[j, :], 'b')
# Axis labels and limits
plt.ylabel("$E$[eV]", fontsize=15)
plt.xlabel("$\\vert k \\vert[1/Å]$", fontsize=15)
plt.ylim(-1.5, 1.5)
plt.xlim(-0.4, 0.4)
plt.legend(("Continuum", "Lattice"))
plt.title("Bi$_2$Se$_3$ bulk (110)" )
plt.show()


