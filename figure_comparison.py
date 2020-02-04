# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 16:20:08 2018

@author: quentinpeter
"""
import numpy as np
from diffusiophoresis_processing import get_profs, get_images, plot_and_save_diffusiophoresis, get_Conc_str
from matplotlib.pyplot import figure, plot, imshow
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib as mpl
import json
import matplotlib.gridspec as gridspec
import os
# %%
out_folder = '../Plots/comparison'
file_names = ['../Data/Experiments/20170301/small_deadend/i200o0p02_2_metadata.json',
              "../Data/Experiments/20170301/small_deadend/i200o200_metadata.json",
              "../Data/Experiments/20170301/small_deadend/i0p02o200_metadata.json"]
salt_conc = False
proteins_conc = False
name = 'figure_salt_a'

file_names = ["../Data/Experiments/20170517/small_deadend/i100_o0p01_2_metadata.json",
              "../Data/Experiments/20170517/small_deadend/i100_o100_metadata.json",
              '../Data/Experiments/20170517/small_deadend/i0p01_o100_2_metadata.json']
salt_conc = False
proteins_conc = False
name = 'figure_salt_b'

file_names = ['../Data/Experiments/20171109/small_channel/i2MLiCl_o10uMBSA_3_metadata.json',
              "../Data/Experiments/20171109/small_channel/i2MLiCl_o100uMBSA_1_metadata.json",
              "../Data/Experiments/20171109/small_channel/i2MLiCl_o1000uMBSA_1_metadata.json"]
salt_conc = False
proteins_conc = True
name = 'figure_prot_a'

# file_names = ["../Data/Experiments/20171122/small_channel/i200mMLiCl_o1uMThy_metadata.json",
#               "../Data/Experiments/20171122/small_channel/i200mMLiCl_o10uMThy_metadata.json"]
# salt_conc = False
# proteins_conc = True
# name = 'figure_prot_b'

# file_names = ["../Data/Experiments/20171116/small_channel/i200mMKIO3_o10uMLYS_metadata.json",
#               "../Data/Experiments/20171109/small_channel/i200mMKIO3_o100uMLYS_1_metadata.json",
#               ]
# salt_conc = False
# proteins_conc = True
# name = 'figure_prot_c'

file_names = ["../Data/Experiments/20171122/small_channel/i200mMLiCl_o1uMThy_metadata.json",
              "../Data/Experiments/20180427/oThy_0p5gpl_iNaOH_1M_1_metadata.json"
              ]
salt_conc = True
proteins_conc = True
name = 'figure_prot_d'

cmap = mpl.cm.get_cmap('plasma')
norm = LogNorm(vmin=.1, vmax=10)

ymax = 0
list_profiles = []
list_X = []
list_Metadata = []
list_times = []
for i, fnmd in enumerate(file_names):

    with open(fnmd) as f:
        Metadata = json.load(f)
    ims, channel_position_px, times = get_images(fnmd)
    X_pos, profiles, background_profiles = get_profs(
        ims, channel_position_px, Metadata)

    list_profiles.append(profiles)
    list_X.append(X_pos)
    list_times.append(times)
    list_Metadata.append(Metadata)

# %%
ymax = np.nanmax([np.nanmax(p) for p in list_profiles])
Nprofs = len(file_names)
plt.figure(figsize=(Nprofs*4, 3))
# set up subplot grid
gridspec.GridSpec(1, Nprofs*10+1)
for i, (X_pos, profiles, times, Metadata) in enumerate(zip(
        list_X, list_profiles, list_times, list_Metadata)):
    ax = plt.subplot2grid((1, Nprofs*10+1), (0, 10*i), colspan=10, rowspan=1)

    if salt_conc:
        Cpout = Metadata['Analyte Concentration In [M]']
        ptype = Metadata['Analyte Type']
        myTitle = r"{}: {:g}M".format(ptype, Cpout)
    elif proteins_conc:
        Cpout = Metadata['Proteins Concentration Out [M]']*1e6
        ptype = Metadata['Proteins Type']
        myTitle = r"{}: {:g}$\mu$M".format(ptype, Cpout)

    else:
        Cin = Metadata["Analyte Concentration In [M]"]
        Cout = Metadata["Analyte Concentration Out [M]"]

        myTitle = "In: {} / Out: {}".format(
            get_Conc_str(Cin), get_Conc_str(Cout))
    ax.set_title(myTitle)  # , fontsize=18)

    # Plot curves
    for j, Y in enumerate(profiles):
        ax.plot(X_pos, Y, c=cmap(norm(times[j])), label="%.1fs" % times[j])

    plt.xlim([-100, 600])
    plt.ylim([0, ymax*1.05])
    plt.xticks(np.arange(6)*100)
    if i > 0:
        plt.tick_params(
            axis='y',          # changes apply to the x-axis
            labelleft=False)  # labels along the bottom edge are off
    if i == 1:
        plt.xlabel('Position [$\mu$m]')
    if i == 0:
        plt.ylabel('Intensity [a.u.]')
    print(np.nanmax(profiles))

plt.subplots_adjust(wspace=0.001)
# %
ax = plt.subplot2grid((1, Nprofs*10+1), (0, Nprofs*10), colspan=1, rowspan=1)
cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm)
cb1.set_label('Time [min]')

plt.tight_layout(pad=0.001, h_pad=0.001, w_pad=0.001)
plt.savefig(os.path.join(out_folder, name+'.pdf'))
