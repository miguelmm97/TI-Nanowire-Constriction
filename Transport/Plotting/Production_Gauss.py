import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec

# Managing system, data and config files
import h5py
import os

# External modules
from functions import check_imaginary



#%% Loading data
outdir = "../ProductionData"
for file in os.listdir(outdir):
    if file == 'G_gauss.h5':
        with h5py.File(os.path.join(outdir, file), 'r') as f:
            datanode1          = f['Potential_xy']
            datanode2          = f['Conductance']
            V                  = datanode1[()][0, :]
            G                  = datanode2[()][0, :]
            datanode3          = f['scatt_states_up']
            scatt_states_up1   = datanode3[()][0, :, :]
            vf                 = datanode2.attrs['vf']
            dis_strength       = datanode2.attrs['dis_strength']
            corr_length        = datanode2.attrs['corr_length']
            fermi0             = datanode2.attrs['fermi0']
            fermiN             = datanode2.attrs['fermif']
            Nfermi             = datanode2.attrs['Nfermi']
            x0                 = datanode2.attrs['x0']
            xN                 = datanode2.attrs['xf']
            Nx                 = datanode2.attrs['Nx']
            Ntheta             = datanode2.attrs['Ntheta_grid']
            r                  = datanode2.attrs['radius']
            fermi              = np.linspace(fermi0, fermiN, Nfermi)
            x                  = np.linspace(x0, xN, Nx)
            scatt_density_up1  = scatt_states_up1 * scatt_states_up1.conj()
            index1             = 2000

    if file == 'Gzoom_gauss.h5':
        with h5py.File(os.path.join(outdir, file), 'r') as f:
            datanode1           = f['Conductance']
            datanode2           = f['scatt_states_up']
            G_zoom              = datanode1[()][0, :]
            fermi0_zoom         = datanode1.attrs['fermi0']
            fermiN_zoom         = datanode1.attrs['fermif']
            Nfermi_zoom         = datanode1.attrs['Nfermi']
            x0_zoom             = datanode1.attrs['x0']
            xN_zoom             = datanode1.attrs['xf']
            Nx_zoom             = datanode1.attrs['Nx']
            Ntheta_zoo          = datanode1.attrs['Ntheta_grid']
            fermi_zoom          = np.linspace(fermi0_zoom, fermiN_zoom, Nfermi_zoom)
            x_zoom              = np.linspace(x0_zoom, xN_zoom, Nx_zoom)


    if file == 'Resonance2_gauss.h5':
        with h5py.File(os.path.join(outdir, file), 'r') as f:
            datanode            = f['scatt_states_up']
            scatt_states_up2    = datanode[()][0, :, :]
            scatt_density_up2   = scatt_states_up2 * scatt_states_up2.conj()
            index2 = 1882

    if file == 'Resonance3_gauss.h5':
        with h5py.File(os.path.join(outdir, file), 'r') as f:
            datanode            = f['scatt_states_up']
            scatt_states_up3    = datanode[()][0, :, :]
            scatt_density_up3   = scatt_states_up3 * scatt_states_up3.conj()
            index3 = 6059

    if file == 'Resonance4_gauss.h5':
        with h5py.File(os.path.join(outdir, file), 'r') as f:
            datanode            = f['scatt_states_up']
            scatt_states_up4    = datanode[()][0, :, :]
            scatt_density_up4   = scatt_states_up4 * scatt_states_up4.conj()
            index4 = 7821


Vstd_th_2d = np.sqrt(dis_strength / (2 * pi)) * vf / corr_length
#%% Plotting

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
axcolour = ['#FF7D66', '#FF416D', '#00B5A1', '#3F6CFF']
color_list = ['#FF7256', '#00BFFF', '#00C957', '#9A32CD', '#FFC125']



# Conductance and potential figure
fig1 = plt.figure(figsize=(6, 6))
gs = GridSpec(2, 4, figure=fig1, wspace=0.5, hspace=0.35)
ax1_1 = fig1.add_subplot(gs[0:1, 0:4])
ax2_1 = fig1.add_subplot(gs[1:2, 0:4])


# Conductance plot
ax1_1.plot(fermi, G, color='#3F6CFF')
ax1_1.set_xlim(fermi0, fermiN)
ax1_1.set_ylim(0, 10)
ax1_1.tick_params(which='major', width=0.75, labelsize=10)
ax1_1.tick_params(which='major', length=6, labelsize=10)
ax1_1.set_xlabel("$E_F$ [meV]", fontsize=10)
ax1_1.set_ylabel("$G[2e^2/h]$",fontsize=10)
ax1_1.plot(np.repeat(2 * Vstd_th_2d, 10), np.linspace(0, 10, 10), '--', color='#00B5A1', alpha=0.5)
ax1_1.plot(np.repeat(-2 * Vstd_th_2d, 10), np.linspace(0, 10, 10), '--', color='#00B5A1', alpha=0.5)
ax1_1.text(2 * Vstd_th_2d - 10, 3, '$2\sigma$', fontsize=10, rotation='vertical', color='#00B5A1', alpha=0.5)
ax1_1.text(- 2 * Vstd_th_2d + 3, 3, '$2\sigma$', fontsize=10, rotation='vertical', color='#00B5A1', alpha=0.5)


# Inset for the potential
left, bottom, width, height = [0.31, 0.55, 0.4, 0.4]
inset_ax1 = ax1_1.inset_axes([left, bottom, width, height])
density_plot = inset_ax1.imshow(V, origin='lower', vmin=np.min(V), vmax=np.max(V))
divider1 = make_axes_locatable(inset_ax1)
cax1 = divider1.append_axes("right", size="5%", pad=0.05)
cbar = fig1.colorbar(density_plot, cax=cax1, orientation='vertical')
cbar.set_label(label='$V$ [meV]', labelpad=10, fontsize=10)
cbar.ax.tick_params(which='major', length=6, labelsize=10)
inset_ax1.set_xlabel("$x$ [nm]", fontsize=10)
inset_ax1.set_ylabel("$r\\theta$ [nm]", fontsize=10)
inset_ax1.set(xticks=[0, int(Nx/2), Nx - 1], xticklabels=[0, int(x[-1]/2), x[-1]])
inset_ax1.set(yticks=[0, int(Ntheta/2), Ntheta - 1], yticklabels=[0, int(pi * r), int(2 * pi * r)])
inset_ax1.tick_params(which='major', width=0.75, labelsize=10)
inset_ax1.tick_params(which='major', length=6, labelsize=10)


# Zoomed-in conductance plot
ax2_1.plot(fermi_zoom, G_zoom, color='#3F6CFF')
ax2_1.set_xlim(fermi0_zoom, fermiN_zoom)
ax2_1.set_ylim(0, 3)
ax2_1.tick_params(which='major', width=0.75, labelsize=10)
ax2_1.tick_params(which='major', length=6, labelsize=10)
ax2_1.set_xlabel("$E_F$ [meV]", fontsize=10)
ax2_1.set_ylabel("$G[2e^2/h]$",fontsize=10)
ax2_1.plot(fermi_zoom[100], G_zoom[100], '.', markersize=10, alpha=0.9, color='#00B5A1')
ax2_1.text(fermi_zoom[100], G_zoom[100] + 2.6, '$ \leftarrow a)$', fontsize=10)
ax2_1.plot(fermi_zoom[index2], G_zoom[index2], '.', markersize=10, alpha=0.9, color='#00B5A1')
ax2_1.text(fermi_zoom[index2], G_zoom[index2] + 0.1, '$b)$', fontsize=10)
ax2_1.plot(fermi_zoom[index3], G_zoom[index3], '.', markersize=10, alpha=0.9, color='#00B5A1')
ax2_1.text(fermi_zoom[index3], G_zoom[index3] + 0.1, '$c)$', fontsize=10)
ax2_1.plot(fermi_zoom[index4], G_zoom[index4], '.', markersize=10, alpha=0.9, color='#00B5A1')
ax2_1.text(fermi_zoom[index4], G_zoom[index4] + 0.1, '$d)$', fontsize=10)

plt.show()
fig1.savefig('fig.pdf', format='pdf', backend='pgf')





# Scattering states figure
fig2 = plt.figure(figsize=(6, 4.5))
gs = GridSpec(4, 4, figure=fig2, wspace=0.5, hspace=0.5)
ax1_2 = fig2.add_subplot(gs[0:2, 0:2])
ax2_2 = fig2.add_subplot(gs[0:2, 2:4])
ax3_2 = fig2.add_subplot(gs[2:4, 0:2])
ax4_2 = fig2.add_subplot(gs[2:4, 2:4])


# Resonance 1
check_imaginary(scatt_density_up1)
density_plot = ax1_2.imshow(np.real(scatt_density_up1) / np.max(np.real(scatt_density_up1)), cmap='plasma', vmin=0, vmax=1, aspect='auto')
# ax1_2.set_xlabel("$x$ [nm]", fontsize=10)
ax1_2.set_ylabel("$r\\theta$ [nm]", fontsize=10)
ax1_2.set(xticks=[0, int(Nx / 4), int(Nx / 2), int(3 * Nx / 4), Nx - 1], xticklabels=[])
ax1_2.set(yticks=[Ntheta - 1, int(Ntheta / 2), 0], yticklabels=[0, int(pi * r), int(2 * pi * r)])
ax1_2.tick_params(which='major', width=0.75, labelsize=10)
ax1_2.tick_params(which='major', length=6, labelsize=10)
ax1_2.text(15, 30, '$a)$', fontsize=10, color='white')


# Resonance 2
check_imaginary(scatt_density_up2)
density_plot = ax2_2.imshow(np.real(scatt_density_up2) / np.max(np.real(scatt_density_up2)), cmap='plasma', vmin=0, vmax=1, aspect='auto')
divider1_2 = make_axes_locatable(ax2_2)
cax2_2 = divider1_2.append_axes("right", size="5%", pad=0.05)
cbar2_2 = fig2.colorbar(density_plot, cax=cax2_2, orientation='vertical')
cbar2_2.set_label(label='$\\vert \psi \\vert ^2$', labelpad=10, fontsize=10)
cbar2_2.ax.tick_params(which='major', length=6, labelsize=10)
# ax2_2.set_xlabel("$x$ [nm]", fontsize=10)
# ax2_2.set_ylabel("$r\\theta$ [nm]", fontsize=10)
ax2_2.set(xticks=[0, int(Nx / 4), int(Nx / 2), int(3 * Nx / 4), Nx - 1], xticklabels=[])
ax2_2.set(yticks=[Ntheta - 1, int(Ntheta / 2), 0], yticklabels=[])
ax2_2.tick_params(which='major', width=0.75, labelsize=10)
ax2_2.tick_params(which='major', length=6, labelsize=10)
ax2_2.text(15, 30, '$b)$', fontsize=10, color='white')


# Resonance 3
check_imaginary(scatt_density_up3)
density_plot = ax3_2.imshow(np.real(scatt_density_up3) / np.max(np.real(scatt_density_up3)), cmap='plasma', vmin=0, vmax=1, aspect='auto')
ax3_2.set_xlabel("$x$ [nm]", fontsize=10)
ax3_2.set_ylabel("$r\\theta$ [nm]", fontsize=10)
ax3_2.set(xticks=[0, int(Nx / 4), int(Nx / 2), int(3 * Nx / 4), Nx - 1], xticklabels=[0, int(x[-1] / 4), int(x[-1] / 2), int(3 * x[-1] / 4), x[-1]])
ax3_2.set(yticks=[Ntheta - 1, int(Ntheta / 2), 0], yticklabels=[0, int(pi * r), int(2 * pi * r)])
ax3_2.tick_params(which='major', width=0.75, labelsize=10)
ax3_2.tick_params(which='major', length=6, labelsize=10)
ax3_2.text(15, 30, '$c)$', fontsize=10, color='white')


# Resonance 4
check_imaginary(scatt_density_up4)
density_plot = ax4_2.imshow(np.real(scatt_density_up4) / np.max(np.real(scatt_density_up4)), cmap='plasma', vmin=0, vmax=1, aspect='auto')
divider4_2 = make_axes_locatable(ax4_2)
cax4_2 = divider4_2.append_axes("right", size="5%", pad=0.05)
cbar4_2 = fig2.colorbar(density_plot, cax=cax4_2, orientation='vertical')
cbar4_2.set_label(label='$\\vert \psi \\vert ^2$', labelpad=10, fontsize=10)
cbar4_2.ax.tick_params(which='major', length=6, labelsize=10)
ax4_2.set_xlabel("$x$ [nm]", fontsize=10)
# ax4_2.set_ylabel("$r\\theta$ [nm]", fontsize=10)
ax4_2.set(xticks=[0, int(Nx / 4), int(Nx / 2), int(3 * Nx / 4), Nx - 1], xticklabels=[0, int(x[-1] / 4), int(x[-1] / 2), int(3 * x[-1] / 4), x[-1]])
ax4_2.set(yticks=[Ntheta - 1, int(Ntheta / 2), 0], yticklabels=[])
ax4_2.tick_params(which='major', width=0.75, labelsize=10)
ax4_2.tick_params(which='major', length=6, labelsize=10)
ax4_2.text(15, 30, '$d)$', fontsize=10, color='white')


plt.show()
fig2.savefig('fig2.pdf', format='pdf', backend='pgf')