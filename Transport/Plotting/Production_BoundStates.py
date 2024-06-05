#%% Modules setup

# Math and plotting
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn

# Managing system, data and config files
from functions import load_my_data, load_my_attr, check_imaginary


#%% Loading data
file_list = ['G_zoom.h5', 'G_zoom.h5', 'Resonance1.h5', 'Resonance2.h5', 'Resonance3.h5', 'Resonance4.h5']
data_dict = load_my_data(file_list, '../ProductionData/Gaussian_20nm_50000pts')

# Parameters
dimension              = data_dict[file_list[0]]['Parameters']['dimension_transport']
r                      = data_dict[file_list[0]]['Parameters']['r']
L                      = data_dict[file_list[0]]['Parameters']['L']
vf                     = data_dict[file_list[0]]['Parameters']['vf']
Nx                     = data_dict[file_list[0]]['Parameters']['Nx']
corr_length            = data_dict[file_list[0]]['Parameters']['corr_length']
dis_strength           = data_dict[file_list[0]]['Parameters']['dis_strength']
Ntheta_plot            = data_dict[file_list[0]]['Parameters']['Ntheta_fft']
Vstd_th_2d             = data_dict[file_list[0]]['Simulation']['Vstd_th_2d']
x                      = data_dict[file_list[0]]['Simulation']['x']

# Figure 1.1: Conductance over Fermi energy
G                      = data_dict[file_list[1]]['Simulation']['G']
fermi                  = data_dict[file_list[1]]['Simulation']['fermi']
V_real                 = data_dict[file_list[1]]['Simulation']['V_real']

# Figure 1.2: Close up conductance over Fermi energy
G_zoom                 = data_dict[file_list[0]]['Simulation']['G']
fermi_zoom             = data_dict[file_list[0]]['Simulation']['fermi']

# Figure 2.1: Scattering states for the conductance resonances
scatt_density_up1       = data_dict[file_list[2]]['Simulation']['scatt_density_up']
scatt_density_down1     = data_dict[file_list[2]]['Simulation']['scatt_density_down']
trans_eigenvalues1      = data_dict[file_list[2]]['Simulation']['trans_eigenvalues']
x_res                   = data_dict[file_list[2]]['Simulation']['x']
index1                  = data_dict[file_list[2]]['Parameters']['E_resonance_index']
scatt_density1          = scatt_density_up1[:, ::10] + scatt_density_down1[:, ::10]

# Figure 2.2: Scattering states for the conductance resonances
scatt_density_up2       = data_dict[file_list[3]]['Simulation']['scatt_density_up']
scatt_density_down2     = data_dict[file_list[3]]['Simulation']['scatt_density_down']
trans_eigenvalues2      = data_dict[file_list[3]]['Simulation']['trans_eigenvalues']
index2                  = data_dict[file_list[3]]['Parameters']['E_resonance_index']
scatt_density2          = scatt_density_up2[:, ::10] + scatt_density_down2[:, ::10]

# Figure 2.1: Scattering states for the conductance resonances
scatt_density_up3       = data_dict[file_list[4]]['Simulation']['scatt_density_up']
scatt_density_down3     = data_dict[file_list[4]]['Simulation']['scatt_density_down']
trans_eigenvalues3      = data_dict[file_list[4]]['Simulation']['trans_eigenvalues']
index3                  = data_dict[file_list[4]]['Parameters']['E_resonance_index']
scatt_density3          = scatt_density_up3[:, ::10] + scatt_density_down3[:, ::10]

# Figure 2.1: Scattering states for the conductance resonances
scatt_density_up4       = data_dict[file_list[5]]['Simulation']['scatt_density_up']
scatt_density_down4     = data_dict[file_list[5]]['Simulation']['scatt_density_down']
trans_eigenvalues4      = data_dict[file_list[5]]['Simulation']['trans_eigenvalues']
index4                  = data_dict[file_list[5]]['Parameters']['E_resonance_index']
scatt_density4          = scatt_density_up4[:, ::10] + scatt_density_down4[:, ::10]



#%% Conductance over Fermi energy

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
axcolour = ['#FF7D66', '#FF416D', '#00B5A1', '#3F6CFF']
color_list = ['#FF7256', '#00BFFF', '#00C957', '#9A32CD', '#FFC125']


# Figure 1.1: Conductance over Fermi energy
fig1 = plt.figure(figsize=(6, 6))
gs = GridSpec(2, 4, figure=fig1, wspace=0.5, hspace=0.35)
ax1_1 = fig1.add_subplot(gs[0:1, 0:4])
ax2_1 = fig1.add_subplot(gs[1:2, 0:4])

ax1_1.plot(fermi, G, color='#3F6CFF')
ax1_1.set_xlim(fermi[0], fermi[-1])
ax1_1.set_ylim(0, 10)
ax1_1.set_xlabel("$E_F$ [meV]", fontsize=10)
ax1_1.set_ylabel("$G[2e^2/h]$",fontsize=10)
ax1_1.tick_params(which='major', width=0.75, labelsize=10)
ax1_1.tick_params(which='major', length=6, labelsize=10)
ax1_1.plot(np.repeat(2 * Vstd_th_2d, 10), np.linspace(0, 10, 10), '--', color='#00B5A1', alpha=0.5)
ax1_1.plot(np.repeat(-2 * Vstd_th_2d, 10), np.linspace(0, 10, 10), '--', color='#00B5A1', alpha=0.5)
ax1_1.text(2 * Vstd_th_2d - 10, 3, '$2\sigma$', fontsize=10, rotation='vertical', color='#00B5A1', alpha=0.5)
ax1_1.text(- 2 * Vstd_th_2d + 3, 3, '$2\sigma$', fontsize=10, rotation='vertical', color='#00B5A1', alpha=0.5)

check_imaginary(V_real)
V_real = np.real(V_real)
left, bottom, width, height = [0.31, 0.55, 0.4, 0.4]
inset_ax1 = ax1_1.inset_axes([left, bottom, width, height])
density_plot = inset_ax1.imshow(V_real, origin='lower', vmin=np.min(V_real), vmax=np.max(V_real))
divider1 = make_axes_locatable(inset_ax1)
cax1 = divider1.append_axes("right", size="5%", pad=0.05)
cbar = fig1.colorbar(density_plot, cax=cax1, orientation='vertical')
cbar.set_label(label='$V$ [meV]', labelpad=10, fontsize=10)
cbar.ax.tick_params(which='major', length=6, labelsize=10)
inset_ax1.set_xlabel("$x$ [nm]", fontsize=10)
inset_ax1.set_ylabel("$r\\theta$ [nm]", fontsize=10)
inset_ax1.set(xticks=[0, int(Nx/2), Nx - 1], xticklabels=[0, int(x[-1]/2), x[-1]])
inset_ax1.set(yticks=[0, int(Ntheta_plot/2), Ntheta_plot - 1], yticklabels=[0, int(pi * r), int(2 * pi * r)])
inset_ax1.tick_params(which='major', width=0.75, labelsize=10)
inset_ax1.tick_params(which='major', length=6, labelsize=10)
inset_ax1.set_aspect('auto')


# Figure 1.2: Close up conductance over Fermi energy
ax2_1.plot(fermi_zoom, G_zoom, color='#3F6CFF')
ax2_1.set_xlim(fermi_zoom[0], fermi_zoom[-1])
ax2_1.set_ylim(0, 3)
ax2_1.tick_params(which='major', width=0.75, labelsize=10)
ax2_1.tick_params(which='major', length=6, labelsize=10)
ax2_1.set_xlabel("$E_F$ [meV]", fontsize=10)
ax2_1.set_ylabel("$G[2e^2/h]$",fontsize=10)
ax2_1.plot(fermi_zoom[index1], G_zoom[index1], '.', markersize=10, alpha=0.9, color='#00B5A1')
ax2_1.text(fermi_zoom[index1], G_zoom[index1] + 0.1, '$  a)$', fontsize=10)
ax2_1.plot(fermi_zoom[index2], G_zoom[index2], '.', markersize=10, alpha=0.9, color='#00B5A1')
ax2_1.text(fermi_zoom[index2], G_zoom[index2] + 0.1, '$b)$', fontsize=10)
ax2_1.plot(fermi_zoom[index3], G_zoom[index3], '.', markersize=10, alpha=0.9, color='#00B5A1')
ax2_1.text(fermi_zoom[index3], G_zoom[index3] + 0.1, '$c)$', fontsize=10)
ax2_1.plot(fermi_zoom[index4], G_zoom[index4], '.', markersize=10, alpha=0.9, color='#00B5A1')
ax2_1.text(fermi_zoom[index4], G_zoom[index4] + 0.1, '$d)$', fontsize=10)

plt.show()
fig1.savefig('fig1.pdf', format='pdf', backend='pgf')
#%% Scattering states


fig2 = plt.figure(figsize=(6, 4.5))
gs = GridSpec(4, 4, figure=fig2, wspace=0.5, hspace=0.5)
ax1_2 = fig2.add_subplot(gs[0:2, 0:2])
ax2_2 = fig2.add_subplot(gs[0:2, 2:4])
ax3_2 = fig2.add_subplot(gs[2:4, 0:2])
ax4_2 = fig2.add_subplot(gs[2:4, 2:4])


# Resonance 1
check_imaginary(scatt_density1)
density_plot1 = ax1_2.imshow(np.real(scatt_density1) / np.max(np.real(scatt_density1)), cmap='plasma', vmin=0, vmax=1, aspect='auto')
# ax1_2.set_xlabel("$x$ [nm]", fontsize=10)
ax1_2.set_ylabel("$r\\theta$ [nm]", fontsize=10)
ax1_2.set(xticks=[0, int(Nx / 4), int(Nx / 2), int(3 * Nx / 4), Nx - 1], xticklabels=[])
ax1_2.set(yticks=[Ntheta_plot - 1, int(Ntheta_plot / 2), 0], yticklabels=[0, int(pi * r), int(2 * pi * r)])
ax1_2.tick_params(which='major', width=0.75, labelsize=10)
ax1_2.tick_params(which='major', length=6, labelsize=10)
ax1_2.text(15, 30, '$a)$', fontsize=10, color='white')

# Resonance 2
check_imaginary(scatt_density2)
density_plot2 = ax2_2.imshow(np.real(scatt_density2) / np.max(np.real(scatt_density2)), cmap='plasma', vmin=0, vmax=1, aspect='auto')
# ax2_2.set_xlabel("$x$ [nm]", fontsize=10)
# ax2_2.set_ylabel("$r\\theta$ [nm]", fontsize=10)
ax2_2.set(xticks=[0, int(Nx / 4), int(Nx / 2), int(3 * Nx / 4), Nx - 1], xticklabels=[])
ax2_2.set(yticks=[Ntheta_plot - 1, int(Ntheta_plot / 2), 0], yticklabels=[])
ax2_2.tick_params(which='major', width=0.75, labelsize=10)
ax2_2.tick_params(which='major', length=6, labelsize=10)
ax2_2.text(15, 30, '$b)$', fontsize=10, color='white')

# Resonance 3
check_imaginary(scatt_density3)
density_plot3 = ax3_2.imshow(np.real(scatt_density3) / np.max(np.real(scatt_density3)), cmap='plasma', vmin=0, vmax=1, aspect='auto')
ax3_2.set_xlabel("$x$ [nm]", fontsize=10)
ax3_2.set_ylabel("$r\\theta$ [nm]", fontsize=10)
ax3_2.set(xticks=[0, int(Nx / 4), int(Nx / 2), int(3 * Nx / 4), Nx - 1], xticklabels=[0, int(x[-1] / 4), int(x[-1] / 2), int(3 * x[-1] / 4), x[-1]])
ax3_2.set(yticks=[Ntheta_plot - 1, int(Ntheta_plot / 2), 0], yticklabels=[0, int(pi * r), int(2 * pi * r)])
ax3_2.tick_params(which='major', width=0.75, labelsize=10)
ax3_2.tick_params(which='major', length=6, labelsize=10)
ax3_2.text(15, 30, '$c)$', fontsize=10, color='white')

# Resonance 4
check_imaginary(scatt_density4)
density_plot4 = ax4_2.imshow(np.real(scatt_density4) / np.max(np.real(scatt_density4)), cmap='plasma', vmin=0, vmax=1, aspect='auto')
ax4_2.set_xlabel("$x$ [nm]", fontsize=10)
# ax4_2.set_ylabel("$r\\theta$ [nm]", fontsize=10)
ax4_2.set(xticks=[0, int(Nx / 4), int(Nx / 2), int(3 * Nx / 4), Nx - 1], xticklabels=[0, int(x[-1] / 4), int(x[-1] / 2), int(3 * x[-1] / 4), x[-1]])
ax4_2.set(yticks=[Ntheta_plot - 1, int(Ntheta_plot / 2), 0], yticklabels=[])
ax4_2.tick_params(which='major', width=0.75, labelsize=10)
ax4_2.tick_params(which='major', length=6, labelsize=10)
ax4_2.text(15, 30, '$d)$', fontsize=10, color='white')

plt.show()
fig2.savefig('fig2.pdf', format='pdf', backend='pgf')


