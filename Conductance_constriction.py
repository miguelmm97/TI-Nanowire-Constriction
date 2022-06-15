# Transport calculation for a clean TI nanocone

import numpy as np
from numpy import pi
import os
import matplotlib.pyplot as plt
import h5py
from scipy.linalg import expm
from functions import transfer_to_scattering, scat_product, transport_checks, thermal_average, finite_voltage_bias


#%% Parameters

# Constants and set up of the model
hbar = 1                            # Planck's constant
e = 1.6e-19                         # Electron charge in C
G_q = ((e ** 2) / hbar)             # Conductance quantum
vf = 330                            # Fermi velocity in meV nm
E_F = np.linspace(-120, 120, 1200)  # Fermi energy
dE = E_F[1] - E_F[0]                # Separation in energies
B_par, B_perp = 0, 6                # Parallel and perpendicular magnetic fields in T

# Geometry of the nanostructure (MUST BE SYMMETRIC)
l_lead, l_cone, l_constriction = 100, 100, 100  # Length of the leads, cones, and the constriction (nm)
w1, h1 = 150, 15                   # Width and height of the wire in the leads (nm)
w2, h2 = 15, 15                    # Width and height of the wire in the constriction (nm)
a1 = (w1 + h1) / pi                # Radius at x1
a2 = (w2 + h2) / pi                # Radius at x2
x0 = 0                             # Initial length
x1 = x0 + l_lead                   # Final position first lead
x2 = x1 + l_cone                   # Initial position of the constriction
x3 = x2 + l_constriction           # Final position of the constriction
x4 = x3 + l_cone                   # Start of the final lead
x5 = x4 + l_lead                   # End of the nanostructure
print(x0, x1, x2, x3, x4, x5)
L = x5 - x0                        # Length of the nanowire in nm
slope_w = (w2 - w1) / (x2 - x1)    # Slope of the width
slope_h = (h2 - h1) / (x2 - x1)    # Slope of the height
slope_a = (a2 - a1) / (x2 - x1)    # Slope of the radius
def height(x):
    if x0 <= x < x1 or x4 < x <= x5:
        return h1
    if x1 <= x <= x2:
        return h1 + slope_h * (x - x1)
    if x3 <= x <= x4:
        return h2 - slope_h * (x - x3)
    if x2 < x < x3:
        return h2
def width(x):
    if x0 <= x < x1 or x4 < x <= x5:
        return w1
    if x1 <= x <= x2:
        return w1 + slope_w * (x - x1)
    if x3 <= x <= x4:
        return w2 - slope_w * (x - x3)
    if x2 < x < x3:
        return w2
def radius(x):
    if x0 <= x < x1 or x4 < x <= x5:
        return a1
    if x1 <= x <= x2:
        return a1 + slope_a * (x - x1)
    if x3 <= x <= x4:
        return a2 - slope_a * (x - x3)
    if x2 < x < x3:
        return a2
def slope(x):
    if x0 <= x < x1 or x4 < x <= x5:
        return 0
    if (x1 <= x <= x2) or (x3 <= x <= x4):
        return slope_a
    elif x2 < x < x3:
        return 0

# Declarations
l_cutoff = 50                                      # Cutoff for the number of angular momentum modes that we consider
modes = np.arange(-l_cutoff, l_cutoff+1)           # Angular momentum modes
n_modes = int(len(modes))                          # Number of l modes
n_s = 2                                            # Spin components
n_x = 300                                          # Number of x intervals
L_grid = np.linspace(x0, x5, n_x)                  # Grid for x direction
dx = L / n_x                                       # Transfer step
aux1 = 0.5 * 1e16 * e * B_par                      # Auxiliary amplitude for the parallel vector potential
aux2 = 1e16 * e * B_perp                           # Auxiliary amplitude for the perpendicular vector potential
G = np.zeros((len(E_F), ))                         # Conductance vector

# We use 1e16 cause T nm = nm/m² so we need 1/1e18, and we need to divide by hbar because we are pulling a factor
# hbar over all but the minimal coupling says p --> p+eA, so to pull the hbar factor out we need to write
# eA/hbar so an added 1e34.

# Pauli matrices
sigma_0 = np.eye(2)                      # Pauli 0
sigma_x = np.array([[0, 1], [1, 0]])     # Pauli x
sigma_y = np.array([[0, -1j], [1j, 0]])  # Pauli y
sigma_z = np.array([[1, 0], [0, -1]])    # Pauli z


# Thermal average parameters
T = 4                                                          # Temperature in K
thermal_interval = 20                                           # Energy range above and below mu that we include on the thermal average in meV
sample_points_therm = int(thermal_interval / dE)                # Points included in the integration above and below
E_F_thermal = E_F[sample_points_therm: -sample_points_therm]    # Range over which we calculate the thermal conductance
G_therm = np.zeros((len(E_F_thermal), ))                        # Thermal conductance vector


# Finite voltage bias parameters
eVb = 10                                                  # Finite voltage bias
sample_points_bias = int(eVb / dE)                        # Points included in the integration above and below
E_F_bias = E_F[sample_points_bias: -sample_points_bias]   # Range over which we calculate the thermal conductance
G_bias = np.zeros((len(E_F_bias), ))                      # Thermal conductance vector


# Loading possible data
file_name ="G_" + str(B_perp) + str(B_par) + str(w1) + str(w2) + str(h1) + str(h2) + ".h5"
outdir = "Data"
calculate_G = 1  # 1 means we need to calculate in the code
for file in os.listdir(outdir):
    if file == file_name:
        calculate_G = 0  # 0 means we have uploaded it from the data we already have
        file_path = os.path.join(outdir, file)
        with h5py.File(file_path, 'r') as f:
            datanode = f['data']
            G = datanode[0, :]  # Conductance at T=0K

#%% Transport calculation at 0K and Vb=0 meV

if calculate_G == 1:
    M_modes1 = np.eye(n_modes)                       # Exponent of the mode transfer matrix for the Ef term
    M_modes2 = np.diag(modes - 0.5)                  # Exponent of the mode transfer matrix for the p_y term
    M1_aux = (1j / vf) * np.kron(sigma_z, M_modes1)  # Exponent of the transfer matrix for the Ef term (wo Ef)
    M2_aux = np.kron(sigma_x, M_modes2)              # Exponent of the transfer matrix for the p_y term (wo 1/a)
    M3_aux = aux1 * np.kron(sigma_x, M_modes1)       # Exponent of the transfer matrix for the eA_y term (wo a)
    # Mode mixing matrix for eA_x
    def M_offdiag(x):

        M_eA = np.zeros([n_modes, n_modes], )  # Mode mixing matrix for the vector potential
        r = width(x) / (width(x) + height(x))  # Aspect ratio at point x in the geometry
        P = 2 * (width(x) + height(x))         # Perimeter at point x in the geometry

        for i, n1 in enumerate(modes):
            for j, n2 in enumerate(modes):

                if (n1 - n2) % 2 != 0:
                    m = n1 - n2
                    M_eA[i, j] = aux2 * P * vf * ((-1) ** ((m + 1) / 2)) * np.sin(m * pi * r / 2) / (m * m * pi * pi)

        return M_eA

    # Sweep the Fermi level
    for i, fermi_level in enumerate(E_F):

        print(str(i) + "/" + str(len(E_F)-1))

        # Propagation of the scattering matrix
        for j, x in enumerate(L_grid):

            M1 = fermi_level * np.sqrt(1 + slope(x) ** 2) * M1_aux  # Exponent of the transfer matrix for the Ef term
            M2 = M2_aux * np.sqrt(1 + slope(x) ** 2) / radius(x)    # Exponent of the transfer matrix for the p_y term
            M3 = M3_aux * radius(x) * np.sqrt(1 + slope(x) ** 2)    # Exponent of the transfer matrix for the eA_x term
            M4 = (-1j / vf) * np.kron(sigma_0, M_offdiag(x))        # Exponent of the transfer matrix for the eA_y term
            M = M1 + M2 + M3 + M4                                   # Exponent of the transfer matrix
            transfer_matrix = expm(M * dx)                          # Transfer matrix

            if j == 0:
                scat_matrix = transfer_to_scattering(transfer_matrix, n_modes)     # Initial scattering matrix
            else:
                scat_matrix_dx = transfer_to_scattering(transfer_matrix, n_modes)  # Scattering matrix for dx
                scat_matrix = scat_product(scat_matrix, scat_matrix_dx, n_modes)   # Propagating the scattering matrix


        t = scat_matrix[n_modes:, 0: n_modes]      # Transmission matrix
        t_dagger = np.conj(t.T)                    # Conjugate transmission matrix
        G[i] = np.trace(t_dagger @ t)              # Conductance / Gq
        print(G[i])


#%% Thermal-averaged conductance at low temperatures

for i, energy in enumerate(E_F_thermal):
    print(str(i) + "/" + str(len(E_F_thermal)))
    j = i + sample_points_therm                                                       # Index in the complete E_f vector
    integration_interval = E_F[j - sample_points_therm: j + sample_points_therm]      # Energy range for integration
    G_interval = G[j - sample_points_therm: j + sample_points_therm]                  # Conductance range for integration
    G_therm[i] = thermal_average(T, E_F[j], integration_interval, G_interval)         # Thermal averaged conductance


#%% Finite voltage bias conductance

for i, energy in enumerate(E_F_bias):
    print(str(i) + "/" + str(len(E_F_bias)))
    j = i + sample_points_bias                                                       # Index in the complete E_f vector
    mu1, mu2 = E_F[j] + (0.5 * eVb), E_F[j] - (0.5 * eVb)                            # Chemical potentials
    integration_interval = E_F[j - sample_points_bias: j + sample_points_bias]       # Energy range for integration
    G_interval = G[j - sample_points_bias: j + sample_points_bias]                   # Conductance range for integration
    G_bias[i] = finite_voltage_bias(0, mu1, mu2, integration_interval, G_interval)   # Thermal averaged conductance


#%% Output data

if calculate_G == 1:
    with h5py.File(file_name, 'w') as f:
        f.create_dataset("data", data=[G])
        f["data"].attrs.create("B_perp", data=B_perp)
        f["data"].attrs.create("B_par", data=B_par)
        f["data"].attrs.create("w1", data=w1)
        f["data"].attrs.create("h1", data=h1)
        f["data"].attrs.create("w2", data=w2)
        f["data"].attrs.create("h2", data=h2)

#%% Figures

# Conductance
plt.plot(E_F, G, '-b', linewidth=1)
plt.plot(E_F_thermal, G_therm, '.r', markersize=2)
plt.plot(E_F_bias, G_bias, '.m', markersize=2)
plt.plot(E_F, np.repeat(1, len(E_F)), '-.k', linewidth=1, alpha=0.3)
plt.plot(E_F, np.repeat(3, len(E_F)), '-.k', linewidth=1, alpha=0.3)
plt.plot(E_F, np.repeat(5, len(E_F)), '-.k', linewidth=1, alpha=0.3)
plt.plot(E_F, np.repeat(7, len(E_F)), '-.k', linewidth=1, alpha=0.3)
plt.legend(("T=0K", "Thermal Average " + str(T) + "K")) # , "Vb=" + str(eVb) + " meV"))

plt.xlabel("$E_F$ (meV)")
plt.ylabel("$G/G_Q$")
plt.xlim([-100, 100])
plt.ylim([0, max(G)])
plt.title("$B_\perp =$" + str(B_perp) + "$, B =$" + str(B_par) + ", $L_{leads}=$" + str(l_lead) + ", $L_{cones}=$" + str(l_cone) +
          ", $L_{constriction}=$" + str(l_constriction)  +", $w_1=$" + str(w1) +", $h_1=$" + str(h1) + ", $w_2=$" + str(w2) +
          ", $h_2=$" + str(h2) )
plt.savefig(file_name + "_T0" + ".pdf", bbox_inches="tight")
plt.show()



# Only thermal averged conductance
plt.plot(E_F_thermal, G_therm, '-b', linewidth=1)
plt.plot(E_F, np.repeat(1, len(E_F)), '-.k', linewidth=1, alpha=0.3)
plt.plot(E_F, np.repeat(3, len(E_F)), '-.k', linewidth=1, alpha=0.3)
plt.plot(E_F, np.repeat(5, len(E_F)), '-.k', linewidth=1, alpha=0.3)
plt.plot(E_F, np.repeat(7, len(E_F)), '-.k', linewidth=1, alpha=0.3)

plt.xlabel("$E_F$ (meV)")
plt.ylabel("$G/G_Q$")
plt.xlim([-100, 100])
plt.ylim([0, max(G)])
plt.title("$T=$" + str(T) + "$, B_\perp =$" + str(B_perp) + "$, B =$" + str(B_par) + ", $L_{leads}=$" + str(l_lead) + ", $L_{cones}=$" + str(l_cone) +
          ", $L_{constriction}=$" + str(l_constriction)  +", $w_1=$" + str(w1) +", $h_1=$" + str(h1) + ", $w_2=$" + str(w2) +
          ", $h_2=$" + str(h2))
plt.savefig(file_name + "_Thermal" + ".pdf", bbox_inches="tight")
plt.show()
