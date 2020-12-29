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
from scipy.special import erf

from matplotlib.patches import Circle, Rectangle
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox, AnchoredOffsetbox)
from matplotlib.cbook import get_sample_data

from diffusiophoresis_processing import get_profs, get_images, plot_and_save_diffusiophoresis, get_Conc_str, add_inset
# %%
outfolder = '../Plots'

file_names = [
    ["../Data/Experiments/20171116/small_channel/i200mMLiCl_o10uMBSA_metadata.json",
     "../Data/Experiments/20171116/small_channel/i200mMKCl_o10uMBSA_metadata.json",
     "../Data/Experiments/20171116/small_channel/i200mMKIO3_o10uMBSA_metadata.json", ],

    ["../Data/Experiments/20171116/small_channel/i200mMLiCl_o10uMLYS_metadata.json",
     "../Data/Experiments/20171116/small_channel/i200mMKCl_o10uMLYS_metadata.json",
     "../Data/Experiments/20171116/small_channel/i200mMKIO3_o10uMLYS_metadata.json", ],

    ["../Data/Experiments/20171122/small_channel/i200mMLiCl_o1uMThy_metadata.json",
     "../Data/Experiments/20171122/small_channel/i200mMKCl_o10uMThy_metadata.json",
     "../Data/Experiments/20171122/small_channel/i200mMKIO3_o1uMThy_metadata.json", ],

    ["../Data/Experiments/20171116/small_channel/i200mMLiCl_o10uMB-LAC_2_metadata.json",
     "../Data/Experiments/20171116/small_channel/i200mMKCl_o10uMB-LAC_metadata.json",
     "../Data/Experiments/20171116/small_channel/i200mMKIO3_o10uMB-LAC_metadata.json", ],

    ["../Data/Experiments/20171129/small_channel/i200mMLiCl_o0p5gplMYO_metadata.json",
     "../Data/Experiments/20171129/small_channel/i200mMKCl_o0p5gplMYO_metadata.json",
     "../Data/Experiments/20171129/small_channel/i200mMKIO3_o0p5gplMYO_metadata.json", ],

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
        X_pos, profiles, background_profiles, profiles_std = get_profs(
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


def get_circle(color, positive, radius):
    # Annotate the 2nd position with a circle patch
    da = DrawingArea(2 * radius, 2 * radius, 0, 0)
    p = Circle((radius, radius), radius, color=color)
    da.add_artist(p)
    r = Rectangle((radius - 4, radius - 1), 8, 2, color='k')
    da.add_artist(r)
    if positive:
        r = Rectangle((radius - 1, radius - 4), 2, 8, color='k')
        da.add_artist(r)
    return da

sizes_circles = {
    "LiCl": [12, 8],
    "KCl": [10, 10],
    "KIO3": [8, 12]
    }

plt.figure(figsize=(12, 12))

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
            plt.ylabel(Metadata['Proteins Type'].upper(), fontweight='bold', fontsize=18)
            da = get_circle("C8", Metadata['Proteins Type'] == "LYS", 10)
            anchored_box = AnchoredOffsetbox(loc='center',
                                 child=da, pad=0.,
                                 frameon=False,
                                 bbox_to_anchor=(-0.28, 0.5),
                                 bbox_transform=ax.transAxes,
                                 borderpad=0.,
                                 )

            plt.gca().add_artist(anchored_box)

        if idx_prot == 0:

            ax.set_title(Metadata['Analyte Type'], fontweight='bold', fontsize=18)
            cs = sizes_circles[Metadata['Analyte Type']]
            pos_circle = get_circle("C3", True, cs[0])
            neg_circle = get_circle("C0", False, cs[1])
            anchored_box = AnchoredOffsetbox(loc='center',
                                 child=pos_circle, pad=0.,
                                 frameon=False,
                                 bbox_to_anchor=(0.45, 1.25),
                                 bbox_transform=ax.transAxes,
                                 borderpad=0.,
                                 )
            plt.gca().add_artist(anchored_box)
            anchored_box = AnchoredOffsetbox(loc='center',
                                 child=neg_circle, pad=0.,
                                 frameon=False,
                                 bbox_to_anchor=(0.55, 1.25),
                                 bbox_transform=ax.transAxes,
                                 borderpad=0.,
                                 )
            plt.gca().add_artist(anchored_box)


        if idx_prot < 5-1:
            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                labelbottom=False)  # labels along the bottom edge are off
        else:
            if idx_salt == 1:
                plt.xlabel('Position [$\mu$m]', fontsize=18)
        info_dict['axis'] = ax


for i in range(5):
    ax = plt.subplot2grid(grid_size, (i*rowspan, 3*colspan),
                          colspan=1, rowspan=rowspan)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm)
    cb1.set_label('Time [min]', fontsize=16)


plt.tight_layout(pad=0.001, h_pad=0.001, w_pad=0.001)


# for idx_prot, col_dict in enumerate(list_dict):
#     for idx_salt, info_dict in enumerate(col_dict):

#         add_inset(info_dict['ims'], info_dict['channel_position_px'],
#                   info_dict['profiles'], info_dict['mdfn'], maskmargin, info_dict['axis'])
plt.savefig(os.path.join(outfolder, name+'.pdf'), bbox_inches='tight')


# %%

proteins_infos = json.load(open('../Litterature Values/proteins_info.json'))
salts_infos = json.load(open('../Litterature Values/salts_info.json'))
proteins_infos = {key.lower():value for key, value in proteins_infos.items()}
salts_infos = {key.lower():value for key, value in salts_infos.items()}


# Physical parameters
temperature = 273.14 + 23  # Kelvins
boltzmann = 1.38e-23
viscosity = 1e-3  # Water viscosity
electronic_charge = 1.602e-19  # J / V
gamma_over_mu_beta = boltzmann * temperature / electronic_charge
DRh = temperature * boltzmann / (6 * np.pi * viscosity)


plt.figure(figsize=(12, 12))

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

        Dp = DRh / proteins_infos[Metadata['Proteins Type'].lower()]['Hydrodynamic radius']
        Ds = salts_infos[Metadata['Analyte Type'].lower()]['Diffusion']

        Cin = Metadata["Analyte Concentration In [M]"]
        Cout = Metadata["Analyte Concentration Out [M]"]

        myTitle = "In: {} / Out: {}".format(
            get_Conc_str(Cin), get_Conc_str(Cout))
#        ax.set_title(myTitle)  # , fontsize=18)

        # Plot curves
        for t, Y in zip(times, profiles):
            eta = X_pos * 1e-6 / np.sqrt(4 * Ds * t * 60)
            ax.plot(eta, Y, c=cmap(norm(t)),
                        label="%.1fs" % t)
        x = np.linspace(0, 2)
        ax.plot(x, 1 - erf(x/np.sqrt(Dp/Ds)), '--C2')


        plt.xlim([-0.1, .4])
        plt.xticks(np.arange(0, .4, 0.1))

        plt.ylim([0, ymax*1.05])
        plt.yticks(np.arange(0, ymax//0.5 + 1)*0.5)
        if idx_salt > 0:
            plt.tick_params(
                axis='y',          # changes apply to the x-axis
                labelleft=False)  # labels along the bottom edge are off
        else:
            plt.ylabel(Metadata['Proteins Type'].upper(), fontweight='bold', fontsize=18)
            da = get_circle("C8", Metadata['Proteins Type'] == "LYS", 10)
            anchored_box = AnchoredOffsetbox(loc='center',
                                 child=da, pad=0.,
                                 frameon=False,
                                 bbox_to_anchor=(-0.28, 0.5),
                                 bbox_transform=ax.transAxes,
                                 borderpad=0.,
                                 )

            plt.gca().add_artist(anchored_box)

        if idx_prot == 0:

            ax.set_title(Metadata['Analyte Type'], fontweight='bold', fontsize=18)
            cs = sizes_circles[Metadata['Analyte Type']]
            pos_circle = get_circle("C3", True, cs[0])
            neg_circle = get_circle("C0", False, cs[1])
            anchored_box = AnchoredOffsetbox(loc='center',
                                 child=pos_circle, pad=0.,
                                 frameon=False,
                                 bbox_to_anchor=(0.45, 1.25),
                                 bbox_transform=ax.transAxes,
                                 borderpad=0.,
                                 )
            plt.gca().add_artist(anchored_box)
            anchored_box = AnchoredOffsetbox(loc='center',
                                 child=neg_circle, pad=0.,
                                 frameon=False,
                                 bbox_to_anchor=(0.55, 1.25),
                                 bbox_transform=ax.transAxes,
                                 borderpad=0.,
                                 )
            plt.gca().add_artist(anchored_box)


        if idx_prot < 5-1:
            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                labelbottom=False)  # labels along the bottom edge are off
        else:
            if idx_salt == 1:
                plt.xlabel(r'$\eta$ [unitless]', fontsize=18)
        info_dict['axis'] = ax


for i in range(5):
    ax = plt.subplot2grid(grid_size, (i*rowspan, 3*colspan),
                          colspan=1, rowspan=rowspan)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm)
    cb1.set_label('Time [min]', fontsize=16)


plt.tight_layout(pad=0.001, h_pad=0.001, w_pad=0.001)


# for idx_prot, col_dict in enumerate(list_dict):
#     for idx_salt, info_dict in enumerate(col_dict):

#         add_inset(info_dict['ims'], info_dict['channel_position_px'],
#                   info_dict['profiles'], info_dict['mdfn'], maskmargin, info_dict['axis'])
plt.savefig(os.path.join(outfolder, name+'_eta.pdf'), bbox_inches='tight')


