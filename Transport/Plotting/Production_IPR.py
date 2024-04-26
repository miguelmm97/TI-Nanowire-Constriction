#%% Modules setup

# Math and plotting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn

# Managing system, data and config files
from functions import load_my_data


#%% Loading data
file_list = ['IPR17.h5', 'IPR16.h5']
data_dict = load_my_data(file_list, '../Data')

res1           = data_dict[file_list[0]]['Parameters']['res_index']
r1             = data_dict[file_list[0]]['Parameters']['r']
fermi1         = data_dict[file_list[0]]['Simulation']['fermi']
x1             = data_dict[file_list[0]]['Simulation']['delta_x']
theta1         = data_dict[file_list[0]]['Simulation']['delta_theta']
IPR_up1        = data_dict[file_list[0]]['Simulation']['IPR_up']
fit1_params1   = data_dict[file_list[0]]['Simulation']['fit_params1']
fit2_params1   = data_dict[file_list[0]]['Simulation']['fit_params2']

res2           = data_dict[file_list[1]]['Parameters']['res_index']
r2             = data_dict[file_list[1]]['Parameters']['r']
fermi2         = data_dict[file_list[1]]['Simulation']['fermi']
x2             = data_dict[file_list[1]]['Simulation']['delta_x']
theta2         = data_dict[file_list[1]]['Simulation']['delta_theta']
IPR_up2        = data_dict[file_list[1]]['Simulation']['IPR_up']
fit1_params2   = data_dict[file_list[1]]['Simulation']['fit_params1']
fit2_params2   = data_dict[file_list[1]]['Simulation']['fit_params2']



#%% # Inverse participation ratio vs L
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
palette1 = seaborn.color_palette(palette='Blues', n_colors=len(res1))
palette2 = seaborn.color_palette(palette='dark:salmon_r', n_colors=len(res1))

# Set figure
fig1 = plt.figure(figsize=(5, 4))
gs = GridSpec(1, 1, figure=fig1)
ax2 = fig1.add_subplot(gs[0, 0])

# Plot
def funcfit(y, C1, C2): return C1 / y ** C2
for i in range(len(res1)):
    ax2.plot(x1 * theta1 * r1, IPR_up1[i, :], '.', color=palette1[i], label='$E_f=$ {:.2f} [meV], $\\alpha=$ {:.2f}'
             .format(fermi1[res1[i]], fit1_params1[i]))
    ax2.plot(np.linspace(0.1, 3000, 100), funcfit(np.linspace(0.1, 3000, 100), fit1_params1[i], fit2_params2[i]), '--',
             color=palette1[i], alpha=0.5)

for i in range(len(res2)):
    ax2.plot(x2 * theta2 * r2, IPR_up2[i, :], '.', color=palette2[i], label='$E_f=$ {:.2f} [meV], $\\alpha=$ {:.2f}'
             .format(fermi1[res2[i]], fit1_params2[i]))
    ax2.plot(np.linspace(0.1, 3000, 100), funcfit(np.linspace(0.1, 3000, 100), fit1_params2[i], fit2_params2[i]), '--',
             color=palette2[i], alpha=0.5)

# Ticks and labels
xticklabels, xticks = [], []
for i, value in enumerate(x1):
    if i % 2 != 0:
        xticklabels.append('{:.1f}'.format(value / data_dict[file_list[0]]['Parameters']['corr_length']))
        xticks.append(value * theta1[i] * r1)
ax2.set(xticks=xticks, xticklabels=xticklabels)
ax2.tick_params(which='major', width=0.75, labelsize=10)
ax2.tick_params(which='major', length=6, labelsize=10)

# Axes format and legend
ax2.set_yscale('log')
ax2.set_xlabel("$L / \\xi $ (linear size, scaled with $L^2$)", fontsize=10)
ax2.set_ylabel("Inverse participation ratio",fontsize=10)
ax2.legend(loc='best')


plt.show()
