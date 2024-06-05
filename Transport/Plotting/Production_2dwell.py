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
    if file == 'Experiment168.h5':
        with h5py.File(os.path.join(outdir, file), 'r') as f:
            datanode1     = f['Potential_xy']
            datanode2     = f['Conductance']
            V             = datanode1[()][0, :]
            G             = datanode2[()][0, :]
            vf            = datanode2.attrs['vf']
            fermi0        = datanode2.attrs['fermi0']
            fermiN        = datanode2.attrs['fermif']
            Nfermi        = datanode2.attrs['Nfermi']
            x0            = datanode2.attrs['x0']
            xN            = datanode2.attrs['xf']
            Nx            = datanode2.attrs['Nx']
            Ntheta        = datanode2.attrs['Ntheta_grid']
            r             = datanode2.attrs['radius']
            fermi         = np.linspace(fermi0, fermiN, Nfermi)
            x             = np.linspace(x0, xN, Nx)

    if file == 'Resonance1_well_2d.h5':
        with h5py.File(os.path.join(outdir, file), 'r') as f:
            datanode            = f['scatt_states_up']
            scatt_states_up1    = datanode[()][0, :, :]
            scatt_density_up1   = scatt_states_up1 * scatt_states_up1.conj()
            index1 = 1848

    if file == 'Resonance2_well_2d.h5':
        with h5py.File(os.path.join(outdir, file), 'r') as f:
            datanode            = f['scatt_states_up']
            scatt_states_up2    = datanode[()][0, :, :]
            scatt_density_up2   = scatt_states_up2 * scatt_states_up2.conj()
            index2 = 2114

    if file == 'Resonance3_well_2d.h5':
        with h5py.File(os.path.join(outdir, file), 'r') as f:
            datanode            = f['scatt_states_up']
            scatt_states_up3    = datanode[()][0, :, :]
            scatt_density_up3   = scatt_states_up3 * scatt_states_up3.conj()
            index3 = 2190

    if file == 'Resonance4_well_2d.h5':
        with h5py.File(os.path.join(outdir, file), 'r') as f:
            datanode            = f['scatt_states_up']
            scatt_states_up4    = datanode[()][0, :, :]
            scatt_density_up4   = scatt_states_up4 * scatt_states_up4.conj()
            index4 = 2199

gap = vf / (2 * r)
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
ax1_1.plot(20 - gap * np.ones(10, ), np.linspace(0, 10, 10), '--', color='#00B5A1', alpha=0.5)
ax1_1.plot(20 + gap * np.ones(10, ), np.linspace(0, 10, 10), '--', color='#00B5A1', alpha=0.5)
ax1_1.text(20 - gap - 3, 3, '$\Delta E_g$', fontsize=10, rotation='vertical', color='#00B5A1', alpha=0.5)
ax1_1.text(20 + gap + 1, 3,  '$\Delta E_g$', fontsize=10, rotation='vertical', color='#00B5A1', alpha=0.5)


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
ax2_1.plot(fermi, G, color='#3F6CFF')
ax2_1.set_xlim(15, 25)
ax2_1.set_ylim(0, 3)
ax2_1.tick_params(which='major', width=0.75, labelsize=10)
ax2_1.tick_params(which='major', length=6, labelsize=10)
ax2_1.set_xlabel("$E_F$ [meV]", fontsize=10)
ax2_1.set_ylabel("$G[2e^2/h]$",fontsize=10)
ax2_1.plot(fermi[index1], G[index1], '.', markersize=10, alpha=0.75, color='#00B5A1')
ax2_1.text(fermi[index1], G[index1] + 0.1, '$a)$', fontsize=10)
ax2_1.plot(fermi[index2], G[index2], '.', markersize=10, alpha=0.75, color='#00B5A1')
ax2_1.text(fermi[index2], G[index2] + 0.1, '$b)$', fontsize=10)
ax2_1.plot(fermi[index3], G[index3], '.', markersize=10, alpha=0.75, color='#00B5A1')
ax2_1.text(fermi[index3] + 0.1, G[index3] + 0.1, '$e)$', fontsize=10)
ax2_1.plot(fermi[index4], G[index4], '.', markersize=10, alpha=0.75, color='#00B5A1')
ax2_1.text(fermi[index4], G[index4] + 0.1, '$f)$', fontsize=10)

plt.show()
fig1.savefig('fig5.pdf', format='pdf', backend='pgf')




# Scattering states figure
fig2 = plt.figure(figsize=(6, 6.5))
gs = GridSpec(6, 4, figure=fig2, wspace=0.5, hspace=0.5)
ax1_2 = fig2.add_subplot(gs[0:2, 0:2])
ax2_2 = fig2.add_subplot(gs[0:2, 2:4])
ax3_2 = fig2.add_subplot(gs[4:6, 0:2])
ax4_2 = fig2.add_subplot(gs[4:6, 2:4])
ax5_2 = fig2.add_subplot(gs[2:4, 0:2])
ax6_2 = fig2.add_subplot(gs[2:4, 2:4])


# Resonance 1
check_imaginary(scatt_density_up1)
density_plot = ax1_2.imshow(np.real(scatt_density_up1) / np.max(np.real(scatt_density_up1)), cmap='plasma', vmin=0, vmax=1, aspect='auto')
# ax1_2.set_xlabel("$x$ [nm]", fontsize=10)
ax1_2.set_ylabel("$r\\theta$ [nm]", fontsize=10)
ax1_2.set(xticks=[0, int(Nx / 4), int(Nx / 2), int(3 * Nx / 4), Nx - 1], xticklabels=[])
ax1_2.set(yticks=[Ntheta - 1, int(Ntheta / 2), 0], yticklabels=[0, int(pi * r), int(2 * pi * r)])
ax1_2.tick_params(which='major', width=0.75, labelsize=10)
ax1_2.tick_params(which='major', length=6, labelsize=10)
ax1_2.text(15, 40, '$a)$', fontsize=10, color='white')


# Resonance 2
check_imaginary(scatt_density_up2)
density_plot = ax2_2.imshow(np.real(scatt_density_up2) / np.max(np.real(scatt_density_up2)), cmap='plasma', vmin=0, vmax=1, aspect='auto')
divider2_2 = make_axes_locatable(ax2_2)
cax2_2 = divider2_2.append_axes("right", size="5%", pad=0.05)
cbar2_2 = fig2.colorbar(density_plot, cax=cax2_2, orientation='vertical')
cbar2_2.set_label(label='$\\vert \psi \\vert ^2$', labelpad=10, fontsize=10)
cbar2_2.ax.tick_params(which='major', length=6, labelsize=10)
# ax2_2.set_xlabel("$x$ [nm]", fontsize=10)
# ax2_2.set_ylabel("$r\\theta$ [nm]", fontsize=10)
ax2_2.set(xticks=[0, int(Nx / 4), int(Nx / 2), int(3 * Nx / 4), Nx - 1], xticklabels=[])
ax2_2.set(yticks=[Ntheta - 1, int(Ntheta / 2), 0], yticklabels=[])
ax2_2.tick_params(which='major', width=0.75, labelsize=10)
ax2_2.tick_params(which='major', length=6, labelsize=10)
ax2_2.text(15, 40, '$b)$', fontsize=10, color='white')


# Resonance 3
check_imaginary(scatt_density_up3)
density_plot = ax3_2.imshow(np.real(scatt_density_up3) / np.max(np.real(scatt_density_up3)), cmap='plasma', vmin=0, vmax=1, aspect='auto')
ax3_2.set_xlabel("$x$ [nm]", fontsize=10)
ax3_2.set_ylabel("$r\\theta$ [nm]", fontsize=10)
ax3_2.set(xticks=[0, int(Nx / 4), int(Nx / 2), int(3 * Nx / 4), Nx - 1], xticklabels=[0, int(x[-1] / 4), int(x[-1] / 2), int(3 * x[-1] / 4), x[-1]])
ax3_2.set(yticks=[Ntheta - 1, int(Ntheta / 2), 0], yticklabels=[0, int(pi * r), int(2 * pi * r)])
ax3_2.tick_params(which='major', width=0.75, labelsize=10)
ax3_2.tick_params(which='major', length=6, labelsize=10)
ax3_2.text(15, 40, '$e)$', fontsize=10, color='white')


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
ax4_2.text(15, 40, '$f)$', fontsize=10, color='white')


# Logarithmic plot resonance 1
check_imaginary(scatt_density_up3)
density_plot = ax5_2.imshow(np.log(np.real(scatt_density_up1) / np.max(np.real(scatt_density_up1))), cmap='plasma', vmax=0, vmin=-8, aspect='auto')
# ax5_2.set_xlabel("$x$ [nm]", fontsize=10)
ax5_2.set_ylabel("$r\\theta$ [nm]", fontsize=10)
ax5_2.set(xticks=[0, int(Nx / 4), int(Nx / 2), int(3 * Nx / 4), Nx - 1], xticklabels=[])
ax5_2.set(yticks=[Ntheta - 1, int(Ntheta / 2), 0], yticklabels=[0, int(pi * r), int(2 * pi * r)])
ax5_2.tick_params(which='major', width=0.75, labelsize=10)
ax5_2.tick_params(which='major', length=6, labelsize=10)
ax5_2.text(15, 40, '$c)$', fontsize=10, color='white')



# Logarithmic plot resonance 2
check_imaginary(scatt_density_up4)
density_plot = ax6_2.imshow(np.log(np.real(scatt_density_up2) / np.max(np.real(scatt_density_up2))), cmap='plasma', vmax=0, vmin=-8, aspect='auto')
divider6_2 = make_axes_locatable(ax6_2)
cax6_2 = divider6_2.append_axes("right", size="5%", pad=0.05)
cbar6_2 = fig2.colorbar(density_plot, cax=cax6_2, orientation='vertical')
cbar6_2.set_label(label='$log \\vert \psi \\vert ^2$', labelpad=10, fontsize=10)
cbar6_2.ax.tick_params(which='major', length=6, labelsize=10)
# ax6_2.set_xlabel("$x$ [nm]", fontsize=10)
# ax6_2.set_ylabel("$r\\theta$ [nm]", fontsize=10)
ax6_2.set(xticks=[0, int(Nx / 4), int(Nx / 2), int(3 * Nx / 4), Nx - 1], xticklabels=[])
ax6_2.set(yticks=[Ntheta - 1, int(Ntheta / 2), 0], yticklabels=[])
ax6_2.tick_params(which='major', width=0.75, labelsize=10)
ax6_2.tick_params(which='major', length=6, labelsize=10)
ax6_2.text(15, 40, '$d)$', fontsize=10, color='white')


plt.show()
fig2.savefig('fig6.pdf', format='pdf', backend='pgf')