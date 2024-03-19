import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec

# Managing system, data and config files
import h5py
import os

outdir = "ProductionData"
for file in os.listdir(outdir):
    if file == 'Experiment152.h5':
        file_path = os.path.join(outdir, file)
        with h5py.File(file_path, 'r') as f:
            datanode1     = f['Potential_xy']
            datanode2     = f['Conductance']
            V             = datanode1[()]
            G             = datanode2[()]
            fermi0        = datanode2.attrs['fermi0']
            fermiN        = datanode2.attrs['fermif']
            Nfermi        = datanode2.attrs['Nfermi']
            x0            = datanode2.attrs['x0']
            xN            = datanode2.attrs['xf']
            Nx            = datanode2.attrs['Nx']
            Ntheta        = datanode2.attrs['Ntheta_grid']
            r             = datanode2.attrs['r']

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
axcolour = ['#FF7D66', '#FF416D', '#00B5A1', '#3F6CFF']
color_list = ['#FF7256', '#00BFFF', '#00C957', '#9A32CD', '#FFC125']


# fig = plt.figure(figsize=(6, 8))
# gs = GridSpec(2, 6, figure=fig, wspace=0.5, hspace=0.5)
# ax31 = fig.add_subplot(gs[0, 0:2])
# ax32 = fig.add_subplot(gs[1, 0:2])
# ax33 = fig.add_subplot(gs[0, 2:4])
# ax34 = fig.add_subplot(gs[1, 2:4])
# ax35 = fig.add_subplot(gs[0, 4:])
# ax36 = fig.add_subplot(gs[1, 4:])
#
# # Conductance plot
# fig, ax1 = plt.subplots(figsize=(8, 6))
# ax1.plot(fermi, G_avg, color=color_list[1], label='$r=$ {} nm'.format(r))
# ax1.set_xlim(min(fermi), max(fermi))
# ax1.set_ylim(0, 10)
# ax1.plot(np.repeat(2 * Vstd_th_1d, 10), np.linspace(0, 10, 10), '--', color='#A9A9A9', alpha=0.5)
# ax1.plot(np.repeat(-2 * Vstd_th_1d, 10), np.linspace(0, 10, 10), '--', color='#A9A9A9', alpha=0.5)
# ax1.text(2 * Vstd_th_1d - 10, 9, '$2\sigma$', fontsize=20, rotation='vertical', color='#A9A9A9', alpha=0.5)
# ax1.text(- 2 * Vstd_th_1d + 3, 9, '$2\sigma$', fontsize=20, rotation='vertical', color='#A9A9A9', alpha=0.5)
# ax1.tick_params(which='major', width=0.75, labelsize=20)
# ax1.tick_params(which='major', length=14, labelsize=20)
# ax1.set_xlabel("$E_F$ [meV]", fontsize=20)
# ax1.set_ylabel("$G[2e^2/h]$",fontsize=20)
# # ax1.set_title(" Gaussian correlated: ExpID= {}, $\\xi=$ {} nm, $N_q=$ {}, $N_s=$ {}, $L=$ {} nm, $K_v=$ {}".format(expID, corr_length, Nx, N_samples, x[-1], dis_strength))
# ax1.legend(loc='upper right', ncol=1, fontsize=20)
# # Inset showing a potential sample
# sample = 0
# left, bottom, width, height = [0.35, 0.65, 0.3, 0.3]
# inset_ax1 = ax1.inset_axes([left, bottom, width, height])
# inset_ax1.plot(x, V_real_storage[sample, 0, :], color='#6495ED')
# inset_ax1.set_xlim(x[0], x[-1])
# inset_ax1.set_ylim(-4 * Vstd_th_1d, 4 * Vstd_th_1d)
# inset_ax1.plot(x, Vstd_th_1d * np.ones(x.shape), '--k')
# inset_ax1.plot(x, -Vstd_th_1d * np.ones(x.shape), '--k')
# inset_ax1.plot(x, 2 * Vstd_th_1d * np.ones(x.shape), '--k')
# inset_ax1.plot(x, -2 * Vstd_th_1d * np.ones(x.shape), '--k')
# inset_ax1.plot(x, 3 * Vstd_th_1d * np.ones(x.shape), '--k')
# inset_ax1.plot(x, -3 * Vstd_th_1d * np.ones(x.shape), '--k')
# inset_ax1.text(450, 1.1 * Vstd_th_1d, '$1\sigma$', fontsize=20)
# inset_ax1.text(450, 2.1 * Vstd_th_1d, '$2\sigma$', fontsize=20)
# inset_ax1.text(450, 3.1 * Vstd_th_1d, '$3\sigma$', fontsize=20)
# inset_ax1.text(450, -1.5 * Vstd_th_1d,'$1\sigma$', fontsize=20)
# inset_ax1.text(450, -2.5 * Vstd_th_1d, '$2\sigma$', fontsize=20)
# inset_ax1.text(450, -3.5 * Vstd_th_1d, '$3\sigma$', fontsize=20)
# ax1.tick_params(which='major', width=0.75, labelsize=20)
# ax1.tick_params(which='major', length=14, labelsize=20)
# inset_ax1.set_xlabel("$L$ [nm]", fontsize=20)
# inset_ax1.set_ylabel("$V$ [meV]", fontsize=20)
# # inset_ax1.set_title(" Gaussian correlated potential samples with $\\xi=$ {} nm, $K_V=$ {} and $N_x=$ {}".format(corr_length, dis_strength, Nx))
# inset_ax1.plot()
# plt.show()
