# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 16:20:08 2018

@author: quentinpeter

This script is distributed under CC BY-NC-SA version 4.0. You should have
recieved a copy of the licence along with it. If not, see
https://creativecommons.org/licenses/by-nc-sa/4.0/.
"""
import numpy as np
from matplotlib.pyplot import figure, plot, imshow
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib as mpl
import json
import cv2
import matplotlib.gridspec as gridspec
import os


from diffusiophoresis_processing import get_profs, get_images, plot_and_save_diffusiophoresis, get_Conc_str, add_inset
# %%
outfolder = '../Plots'

file_names = [

    ["../Data/Experiments/20171122/small_channel/i200mMLiCl_o1uMThy_metadata.json",
     "../Data/Experiments/20171122/small_channel/i200mMKCl_o10uMThy_metadata.json",
     "../Data/Experiments/20171122/small_channel/i200mMKIO3_o1uMThy_metadata.json", ],

    ["../Data/Experiments/20171116/small_channel/i200mMLiCl_o10uMBSA_metadata.json",
     "../Data/Experiments/20171116/small_channel/i200mMKCl_o10uMBSA_metadata.json",
     "../Data/Experiments/20171116/small_channel/i200mMKIO3_o10uMBSA_metadata.json", ],

    ["../Data/Experiments/20171116/small_channel/i200mMLiCl_o10uMB-LAC_2_metadata.json",
     "../Data/Experiments/20171116/small_channel/i200mMKCl_o10uMB-LAC_metadata.json",
     "../Data/Experiments/20171116/small_channel/i200mMKIO3_o10uMB-LAC_metadata.json", ],

    ["../Data/Experiments/20171129/small_channel/i200mMLiCl_o0p5gplMYO_metadata.json",
     "../Data/Experiments/20171129/small_channel/i200mMKCl_o0p5gplMYO_metadata.json",
     "../Data/Experiments/20171129/small_channel/i200mMKIO3_o0p5gplMYO_metadata.json", ],

    ["../Data/Experiments/20171116/small_channel/i200mMLiCl_o10uMLYS_metadata.json",
     "../Data/Experiments/20171116/small_channel/i200mMKCl_o10uMLYS_metadata.json",
     "../Data/Experiments/20171116/small_channel/i200mMKIO3_o10uMLYS_metadata.json", ],
]
name = "figure_proteins"

cmap = mpl.cm.get_cmap('plasma')
norm = LogNorm(vmin=.1, vmax=10)
maskmargin = 20
ymax = 0
list_dict = []
for idx_prot, list_mdfn in enumerate(file_names):
    list_dict.append([])
    for idx_salt, fnmd in enumerate(list_mdfn):

        with open(fnmd) as f:
            Metadata = json.load(f)
        ims, channel_position_px, times = get_images(fnmd)
        X_pos, profiles, background_profiles = get_profs(
            ims, channel_position_px, Metadata)

        list_dict[idx_prot].append({
            'ims': ims,
            'channel_position_px': channel_position_px,
            'times': times,
            'X_pos': X_pos,
            'profiles': profiles,
            'background_profiles': background_profiles,
            'Metadata': Metadata,
            'mdfn': fnmd})

# %%
plt.figure(figsize=(6*2, 3*5))

colspan = 10
rowspan = 10
grid_size = (5*rowspan, 3*colspan+1)
# set up subplot grid
gridspec.GridSpec(*grid_size)

for idx_prot, col_dict in enumerate(list_dict):

    ymax = np.nanmax([np.nanmax(p['profiles']) for p in col_dict])

    for idx_salt, info_dict in enumerate(col_dict):
        ax = plt.subplot2grid(grid_size, (rowspan*idx_prot, colspan*idx_salt),
                              colspan=colspan, rowspan=rowspan)

        X_pos = info_dict['X_pos']
        profiles = info_dict['profiles']
        times = info_dict['times']
        Metadata = info_dict['Metadata']

        Cin = Metadata["Analyte Concentration In [M]"]
        Cout = Metadata["Analyte Concentration Out [M]"]

        myTitle = "In: {} / Out: {}".format(
            get_Conc_str(Cin), get_Conc_str(Cout))
#        ax.set_title(myTitle)  # , fontsize=18)

        # Plot curves
        for j, Y in enumerate(profiles):
            ax.plot(X_pos, Y, c=cmap(norm(times[j])), label="%.1fs" % times[j])

        plt.xlim([-100, 600])
        plt.ylim([0, ymax*1.05])
        plt.xticks(np.arange(6)*100)
        plt.yticks(np.arange(0, ymax//0.5 + 1)*0.5)
        if idx_salt > 0:
            plt.tick_params(
                axis='y',          # changes apply to the x-axis
                labelleft=False)  # labels along the bottom edge are off
        else:
            plt.ylabel(Metadata['Proteins Type'], fontweight='bold')

        if idx_prot == 0:
            ax.set_title(Metadata['Analyte Type'], fontweight='bold')

        if idx_prot < 5-1:
            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                labelbottom=False)  # labels along the bottom edge are off
        else:
            if idx_salt == 1:
                plt.xlabel('Position [$\mu$m]')
        info_dict['axis'] = ax


# %
for i in range(5):
    ax = plt.subplot2grid(grid_size, (i*rowspan, 3*colspan),
                          colspan=1, rowspan=rowspan)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm)
    cb1.set_label('Time [min]')


plt.tight_layout(pad=0.001, h_pad=0.001, w_pad=0.001)


for idx_prot, col_dict in enumerate(list_dict):
    for idx_salt, info_dict in enumerate(col_dict):

        add_inset(info_dict['ims'], info_dict['channel_position_px'],
                  info_dict['profiles'], info_dict['mdfn'], maskmargin, info_dict['axis'])
plt.savefig(os.path.join(outfolder, name+'.pdf'))
