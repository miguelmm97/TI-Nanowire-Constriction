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
file_list = ['Experiment202.h5']

# Old storage scheme
# data_dict = load_my_data(file_list, '../Data')
# attr_dict = load_my_attr(file_list, '../Data', 'Conductance')
#
# dimension            = '2d'
# r                    = attr_dict[file_list[0]]['radius']
# vf                   = attr_dict[file_list[0]]['vf']
# corr_length          = attr_dict[file_list[0]]['corr_length']
# dis_strength         = attr_dict[file_list[0]]['dis_strength']
# Ntheta_plot          = attr_dict[file_list[0]]['Ntheta_grid']
# G                    = data_dict[file_list[0]]['Conductance'][0, :]
# V_real               = data_dict[file_list[0]]['Potential_xy'][0, :, :]
# scatt_states_up      = data_dict[file_list[0]]['scatt_states_up'][0, :, :]
# scatt_states_down    = data_dict[file_list[0]]['scatt_states_down'][0, :, :]
# # trans_eigenvalues  = data_dict[file_list[0]]['trans_eigenvalues']
# Nf, f0, f1           = [attr_dict[file_list[0]][key] for key in ('Nfermi', 'fermi0', 'fermif')]
# Nx, x0, x1           = [attr_dict[file_list[0]][key] for key in ('Nx', 'x0', 'xf')]
#
# L = x1; x = np.linspace(x0, x1, Nx)
# fermi = np.linspace(f0, f1, Nf); E_resonance_index = 100
# scatt_density_up = scatt_states_up * scatt_states_up.conj()
# scatt_density_down = scatt_states_down * scatt_states_down.conj()


# New storage scheme
data_dict = load_my_data(file_list, '../Data')

dimension              = data_dict[file_list[0]]['Parameters']['dimension']
r                      = data_dict[file_list[0]]['Parameters']['r']
L                      = data_dict[file_list[0]]['Parameters']['L']
vf                     = data_dict[file_list[0]]['Parameters']['vf']
Nx                     = data_dict[file_list[0]]['Parameters']['Nx']
corr_length            = data_dict[file_list[0]]['Parameters']['corr_length']
dis_strength           = data_dict[file_list[0]]['Parameters']['dis_strength']
E_resonance_index      = data_dict[file_list[0]]['Parameters']['E_resonance_index']
Ntheta_plot            = data_dict[file_list[0]]['Parameters']['Ntheta_fft']
G                      = data_dict[file_list[0]]['Simulation']['G']
V_real                 = data_dict[file_list[0]]['Simulation']['V_real']
scatt_density_up       = data_dict[file_list[0]]['Simulation']['scatt_density_up']
scatt_density_down     = data_dict[file_list[0]]['Simulation']['scatt_density_down']
trans_eigenvalues      = data_dict[file_list[0]]['Simulation']['trans_eigenvalues']
transmission_eigenval  = data_dict[file_list[0]]['Parameters']['transmission_eigenval']
fermi                  = data_dict[file_list[0]]['Simulation']['fermi']
x                      = data_dict[file_list[0]]['Simulation']['x']

scatt_density_tot = scatt_density_up + scatt_density_down

# %% Figures
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
axcolour = ['#FF7D66', '#FF416D', '#00B5A1', '#3F6CFF']
color_list = ['#FF7256', '#00BFFF', '#00C957', '#9A32CD', '#FFC125']
Vstd_th_1d = np.sqrt(dis_strength / np.sqrt(2 * pi)) * (vf / corr_length)  # Standard deviation 1d Gauss
Vstd_th_2d = np.sqrt(dis_strength / (2 * pi)) * vf / corr_length  # Standard deviation 2d Gauss

# Rotationally symmetric potential
if dimension=='1d':
    fig = plt.figure(figsize=(6, 8))
    gs = GridSpec(2, 6, figure=fig, wspace=0.5, hspace=0.5)
    ax31 = fig.add_subplot(gs[0, 0:2])
    ax32 = fig.add_subplot(gs[1, 0:2])
    ax33 = fig.add_subplot(gs[0, 2:4])
    ax34 = fig.add_subplot(gs[1, 2:4])
    ax35 = fig.add_subplot(gs[0, 4:])
    ax36 = fig.add_subplot(gs[1, 4:])

    # Conductance vs Fermi level
    ax31.plot(fermi, G, color=color_list[1], label='$r=$ {} nm'.format(r))
    ax31.set_xlim(-200, 200)
    ax31.set_ylim(0, 12.5)
    ax31.set_xlabel("$E_F$ [meV]", fontsize=20)
    ax31.set_ylabel("$G[2e^2/h]$", fontsize=20)
    ax31.tick_params(which='major', width=0.75, labelsize=20)
    ax31.tick_params(which='major', length=14, labelsize=20)
    ax31.arrow(fermi[E_resonance_index], 3.5, 0, -1, width=0.4, head_length=0.1)

    # Conductance vs Fermi level (no background region)
    ax32.plot(fermi, G, color=color_list[1], label='$r=$ {} nm'.format(r))
    ax32.set_xlim(-50, 50)
    ax32.set_ylim(0, 15)
    ax32.arrow(fermi[E_resonance_index], 3.5, 0, -1, width=0.4, head_length=0.1)
    ax32.set_xlabel("$E_F$ [meV]", fontsize=20)
    ax32.set_ylabel("$G[2e^2/h]$", fontsize=20)
    ax32.tick_params(which='major', width=0.75, labelsize=20)
    ax32.tick_params(which='major', length=14, labelsize=20)
    ax32.set_title("Conductance in the resonant region ")

    gap = vf / (2 * r)
    # ax32.plot((amplitude - gap) * np.ones((10, )), np.linspace(0, 15, 10), '--b')
    # ax32.plot((amplitude + gap) * np.ones((10,)), np.linspace(0, 15, 10), '--b')
    # ax32.plot((-amplitude - gap) * np.ones((10, )), np.linspace(0, 15, 10), '--b')
    # ax32.plot((-amplitude + gap) * np.ones((10,)), np.linspace(0, 15, 10), '--b')
    # ax32.plot((V1 - gap) * np.ones((10, )), np.linspace(0, 15, 10), '--b')
    # ax32.plot((V1 + gap) * np.ones((10,)), np.linspace(0, 15, 10), '--b')


    # Potential sample
    ax33.plot(x, V_real, color='#6495ED')
    ax33.set_xlim(x[0], x[-1])
    ax33.set_ylim(-4 * Vstd_th_1d, 4 * Vstd_th_1d)
    ax33.set_xlabel("$x$ [nm]", fontsize=20)
    ax33.set_ylabel('$V$ [meV]', fontsize=20)
    ax33.set(xticks=[0, int(L / 4), int(L / 2), int(3 * L / 4), L - 1], xticklabels=[0, int(L / 4), int(L / 2), int(3 * L / 4), L])
    ax33.tick_params(which='major', width=0.75, labelsize=15)
    ax33.tick_params(which='major', length=14, labelsize=15)
    # peaks = np.arange(1, 20, 2) * period / 4
    # for pos in peaks:
    #     ax33.plot(pos * np.ones((10, )), np.linspace(-300, 300, 10), '--m')


    # Distribution of scattering states
    # N_lead, N_trans, N_well = int(Nx * (L_lead / L)), int(Nx * (L_trans / L)), int(Nx * (L_well / L))
    check_imaginary(scatt_density_up)
    density_plot = ax34.imshow(np.real(scatt_density_up) / np.max(np.real(scatt_density_up)),
                                origin='lower', cmap='plasma', vmin=0, vmax=1, aspect='auto')
    divider34 = make_axes_locatable(ax34)
    cax34 = divider34.append_axes("right", size="5%", pad=0.05)
    cbar34 = fig.colorbar(density_plot, cax=cax34, orientation='vertical')
    cbar34.set_label(label='$\\vert \psi \\vert ^2$', labelpad=10, fontsize=20)
    cbar34.ax.tick_params(which='major', length=14, labelsize=15)
    ax34.set_xlabel("$x$ [nm]", fontsize=20)
    ax34.set_ylabel("$r\\theta$ [nm]", fontsize=20)
    ax34.set(xticks=[0, int(Nx / 4), int(Nx / 2), int(3 * Nx / 4), Nx - 1], xticklabels=[0, int(L / 4), int(L / 2), int(3 * L / 4), L])
    ax34.set(yticks=[0, int(Ntheta_plot / 2), Ntheta_plot - 1], yticklabels=[0, int(pi * r), int(2 * pi * r)])
    ax34.tick_params(which='major', width=0.75, labelsize=15)
    ax34.tick_params(which='major', length=14, labelsize=15)
    ax34.set_xlim(0, Nx)
    # ax34.plot(N_lead * np.ones((10, )), np.linspace(0, 300, 10), '--m')
    # ax34.plot((N_lead + N_trans) * np.ones((10,)), np.linspace(0, 300, 10), '--m')
    # ax34.plot((N_lead + N_trans + N_well) * np.ones((10,)), np.linspace(0, 300, 10), '--m')
    # ax34.plot((N_lead + 2 * N_trans + N_well) * np.ones((10,)), np.linspace(0, 300, 10), '--m')
    # peaks = np.arange(1, 20, 2) * period / 4
    # for pos in peaks:
    #     Npos = int(Nx * (pos / L))
    #     ax34.plot(Npos * np.ones((10, )), np.linspace(0, 300, 10), '--c')



    # Transmission eigenvalues
    ax35.plot(np.arange(len(trans_eigenvalues)), np.sort(trans_eigenvalues), 'o', color=color_list[2])
    ax35.plot(len(trans_eigenvalues) - transmission_eigenval - 1,
              np.sort(trans_eigenvalues)[len(trans_eigenvalues) - transmission_eigenval - 1], 'o', color='red')
    ax35.set_yscale('log')
    ax35.yaxis.set_label_position("right")
    ax35.yaxis.tick_right()
    ax35.set_ylim(10e-16, 10)
    ax35.set_xlabel("Transmission eigenvalues", fontsize=20)
    ax35.set_ylabel("eig$(t^\dagger t)$", fontsize=20)
    ax35.tick_params(which='major', width=0.75, labelsize=20)
    ax35.tick_params(which='major', length=14, labelsize=20)
    ax35.set_title('Transmission eigenvalues for $E=$ {:.2f} meV'.format(fermi[E_resonance_index]), fontsize=20)

    # Distribution of scattering states in logscale
    check_imaginary(scatt_density_up)
    density_plot = ax36.imshow(np.log(np.real(scatt_density_up) / np.max(np.real(scatt_density_up))),
                                      origin='lower', cmap='plasma', aspect='auto', vmax=0, vmin=-8)
    divider36 = make_axes_locatable(ax36)
    cax36 = divider36.append_axes("right", size="5%", pad=0.05)
    cbar36 = fig.colorbar(density_plot, cax=cax36, orientation='vertical')
    cbar36.set_label(label='$log \\vert \psi \\vert ^2$', labelpad=10, fontsize=20)
    cbar36.ax.tick_params(which='major', length=14, labelsize=15)
    ax36.set_xlabel("$x$ [nm]", fontsize=20)
    # ax36.set_ylabel("$r\\theta$ [nm]", fontsize=20)
    ax36.set(xticks=[0, int(Nx / 4), int(Nx / 2), int(3 * Nx / 4), Nx - 1],
             xticklabels=[0, int(L / 4), int(L / 2), int(3 * L / 4), L])
    ax36.set(yticks=[])
    ax36.tick_params(which='major', width=0.75, labelsize=15)
    ax36.tick_params(which='major', length=14, labelsize=15)
    ax36.set_title(" Bound state density at energy $E=$ {:.2f} meV".format(fermi[E_resonance_index]))

    plt.show()


# Broken rotational symmetry
else:

    fig = plt.figure(figsize=(6, 8))
    gs = GridSpec(2, 6, figure=fig, wspace=0.5, hspace=0.5)
    ax31 = fig.add_subplot(gs[0, 0:2])
    ax32 = fig.add_subplot(gs[1, 0:2])
    ax33 = fig.add_subplot(gs[0, 2:4])
    ax34 = fig.add_subplot(gs[1, 2:4])
    ax35 = fig.add_subplot(gs[0, 4:])
    ax36 = fig.add_subplot(gs[1, 4:])

    # Conductance vs Fermi level
    ax31.plot(fermi, G, color=color_list[1], label='$r=$ {} nm'.format(r))
    ax31.set_xlim(-200, 200)
    ax31.set_ylim(0, 12.5)
    ax31.set_xlabel("$E_F$ [meV]", fontsize=20)
    ax31.set_ylabel("$G[2e^2/h]$", fontsize=20)
    ax31.tick_params(which='major', width=0.75, labelsize=20)
    ax31.tick_params(which='major', length=14, labelsize=20)
    ax31.arrow(fermi[E_resonance_index], 3.5, 0, -1, width=0.4, head_length=0.1)
    # ax31.legend(loc='upper right', ncol=1, fontsize=20)

    # Conductance vs Fermi level (no background region)
    ax32.plot(fermi, G, color=color_list[1], label='$r=$ {} nm'.format(r))
    ax32.set_xlim(-50, 50)
    ax32.set_ylim(0, 15)
    ax32.arrow(fermi[E_resonance_index], 3.5, 0, -1, width=0.4, head_length=0.1)
    # ax32.plot(np.repeat(2 * Vstd_th_2d, 10), np.linspace(0, 10, 10), '--', color='#A9A9A9', alpha=0.5)
    # ax32.plot(np.repeat(-2 * Vstd_th_2d, 10), np.linspace(0, 10, 10), '--', color='#A9A9A9', alpha=0.5)
    # ax32.text(2 * Vstd_th_2d - 10, 3, '$2\sigma$', fontsize=20, rotation='vertical', color='#A9A9A9', alpha=0.5)
    # ax32.text(- 2 * Vstd_th_2d + 3, 3, '$2\sigma$', fontsize=20, rotation='vertical', color='#A9A9A9', alpha=0.5)
    ax32.set_xlabel("$E_F$ [meV]", fontsize=20)
    ax32.set_ylabel("$G[2e^2/h]$", fontsize=20)
    ax32.tick_params(which='major', width=0.75, labelsize=20)
    ax32.tick_params(which='major', length=14, labelsize=20)
    ax32.set_title("Conductance in the resonant region ")
    ax32.legend(loc='upper right', ncol=1, fontsize=20)

    # Potential sample
    sample = 0
    auxV = V_real + np.abs(np.min(V_real))
    density_plot = ax33.imshow(np.log(auxV) / np.max(auxV), origin='lower', aspect='auto')
    divider33 = make_axes_locatable(ax33)
    cax33 = divider33.append_axes("right", size="5%", pad=0.05)
    cbar33 = fig.colorbar(density_plot, cax=cax33, orientation='vertical')
    cbar33.set_label(label='$V$ [meV]', labelpad=-5, fontsize=20)
    cbar33.ax.tick_params(which='major', length=14, labelsize=15)
    ax33.set_xlabel("$x$ [nm]", fontsize=20)
    ax33.set_ylabel("$r\\theta$ [nm]", fontsize=20)
    ax33.set(xticks=[0, int(Nx / 4), int(Nx / 2), int(3 * Nx / 4), Nx - 1],
             xticklabels=[0, int(L / 4), int(L / 2), int(3 * L / 4), L])
    ax33.set(yticks=[0, int(Ntheta_plot / 2), Ntheta_plot - 1], yticklabels=[0, int(pi * r), int(2 * pi * r)])
    ax33.tick_params(which='major', width=0.75, labelsize=15)
    ax33.tick_params(which='major', length=14, labelsize=15)
    ax33.set_title(
        " Gaussian correlated potential samples with $\\xi=$ {} nm, $K_V=$ {} and $N_x=$ {}".format(corr_length,
                                                                                                    dis_strength, Nx))

    gap = vf / (2 * r)
    # contours = ax33.contour(V_real_storage[sample, :, :], levels=[fermi[E_resonance_index] - 2 * gap,
    #                                      fermi[E_resonance_index] + 2 * gap], colors="black", linewidths=2)

    # Distribution of scattering states
    check_imaginary(scatt_density_tot)
    density_plot = ax34.imshow(np.real(scatt_density_tot) / np.max(np.real(scatt_density_tot)),
                               cmap='plasma', vmin=0, vmax=1, aspect='auto')
    divider34 = make_axes_locatable(ax34)
    cax34 = divider34.append_axes("right", size="5%", pad=0.05)
    cbar34 = fig.colorbar(density_plot, cax=cax34, orientation='vertical')
    cbar34.set_label(label='$\\vert \psi \\vert ^2$', labelpad=10, fontsize=20)
    cbar34.ax.tick_params(which='major', length=14, labelsize=15)
    ax34.set_xlabel("$x$ [nm]", fontsize=20)
    ax34.set_ylabel("$r\\theta$ [nm]", fontsize=20)
    ax34.set(xticks=[0, int(Nx / 4), int(Nx / 2), int(3 * Nx / 4), Nx - 1],
             xticklabels=[0, int(L / 4), int(L / 2), int(3 * L / 4), L])
    ax34.set(yticks=[0, int(Ntheta_plot / 2), Ntheta_plot - 1], yticklabels=[0, int(pi * r), int(2 * pi * r)])
    ax34.tick_params(which='major', width=0.75, labelsize=15)
    ax34.tick_params(which='major', length=14, labelsize=15)
    ax34.set_title(" Bound state density at energy $E=$ {:.2f} meV".format(fermi[E_resonance_index]))
    gap = vf / (2 * r)
    # contours = ax34.contour(V_real_storage[sample, :, :], levels=[fermi[E_resonance_index] - 2 * gap,
    #                                                               fermi[E_resonance_index] + 2 * gap], colors="black",
    #                         linewidths=2)

    # Transmission eigenvalues
    ax35.plot(np.arange(len(trans_eigenvalues)), np.sort(trans_eigenvalues), 'o', color=color_list[2])
    ax35.plot(len(trans_eigenvalues) - transmission_eigenval - 1,
              np.sort(trans_eigenvalues)[len(trans_eigenvalues) - transmission_eigenval - 1], 'o', color='red')
    ax35.set_yscale('log')
    ax35.set_ylim(10e-16, 10)
    ax35.yaxis.set_label_position("right")
    ax35.yaxis.tick_right()
    ax35.set_xlabel("Transmission eigenvalues", fontsize=20)
    ax35.set_ylabel("eig$(t^\dagger t)$", fontsize=20)
    ax35.tick_params(which='major', width=0.75, labelsize=20)
    ax35.tick_params(which='major', length=14, labelsize=20)
    ax35.set_title('Transmission eigenvalues for $E=$ {:.2f} meV'.format(fermi[E_resonance_index]), fontsize=20)


    # Distribution of scattering states in logscale
    check_imaginary(scatt_density_up)
    density_plot = ax36.imshow(np.log(np.real(scatt_density_up) / np.max(np.real(scatt_density_up))),
                                                                        cmap='plasma', aspect='auto', vmax=0, vmin=-8)
    divider36 = make_axes_locatable(ax36)
    cax36 = divider36.append_axes("right", size="5%", pad=0.05)
    cbar36 = fig.colorbar(density_plot, cax=cax36, orientation='vertical')
    cbar36.set_label(label='$log \\vert \psi \\vert ^2$', labelpad=10, fontsize=20)
    cbar36.ax.tick_params(which='major', length=14, labelsize=15)
    ax36.set_xlabel("$x$ [nm]", fontsize=20)
    # ax36.set_ylabel("$r\\theta$ [nm]", fontsize=20)
    ax36.set(xticks=[0, int(Nx / 4), int(Nx / 2), int(3 * Nx / 4), Nx - 1],
             xticklabels=[0, int(L / 4), int(L / 2), int(3 * L / 4), L])
    ax36.set(yticks=[])
    ax36.tick_params(which='major', width=0.75, labelsize=15)
    ax36.tick_params(which='major', length=14, labelsize=15)
    ax36.set_title(" Bound state density at energy $E=$ {:.2f} meV".format(fermi[E_resonance_index]))

    plt.show()




