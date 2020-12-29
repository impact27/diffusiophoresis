# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 13:58:50 2018

@author: quentinpeter

This script is distributed under CC BY-NC-SA version 4.0. You should have
recieved a copy of the licence along with it. If not, see
https://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

# %%


from glob import glob
import os
from matplotlib.pyplot import figure, plot
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import json
# Load local libraries
from matplotlib.colors import LogNorm

from diffusiophoresis_processing import (get_profs, get_images,
                            plot_and_save_diffusiophoresis)


# %%
fnmd = sorted(glob('../Data/Experiments/20171116/*mall*/*metadata.json'))[-1]
maskmargin = 20
flatten = True

# Get functions
cmap = matplotlib.cm.get_cmap('plasma')
norm = LogNorm(vmin=.1, vmax=10)


# %% Treat data
with open(fnmd) as f:
    Metadata = json.load(f)

ims, channel_position_px, times = get_images(fnmd, flatten=flatten)
X_pos, profiles, background_profiles, profiles_std = get_profs(
    ims, channel_position_px, Metadata, maskmargin=maskmargin)
plot_and_save_diffusiophoresis(ims, channel_position_px, times,
                               X_pos, profiles, background_profiles,
                               fnmd, maskmargin, profiles_std)


# %%
idx = 90
im = ims[-1]
[first, last, ySide] = channel_position_px
pixs = Metadata['Pixel Size [m]']*1e6

smallim = im[ySide+idx-50:ySide+idx+50, first-20:last+20]
width = last-first
widthprof = np.mean(smallim, 0)
xpos = (np.arange(len(widthprof)) - len(widthprof) / 2) * pixs

figure()
plot(xpos, widthprof)
# plot(filt)
plot(np.ones(2) * (-width / 2) * pixs, [widthprof.min(), widthprof.max()], 'r')
plot(np.ones(2) * (width / 2) * pixs, [widthprof.min(), widthprof.max()], 'r')

plt.xlabel(r"Position [$\mu$m]")
plt.ylabel("Fluorescent intensity")

origin, size = plt.gca().get_position().get_points()

ax2 = plt.axes([.7, .5, .2, .35])
ax2.imshow(smallim)
ax2.plot([20, width + 20], [50, 50], 'r', alpha=.7)
ax2.plot([20, 20], [40, 60], 'r', alpha=.7)
ax2.plot([width + 20, width + 20], [40, 60], 'r', alpha=.7)
ax2.set_xticks([])
ax2.set_yticks([])

# plt.savefig('channel_width.pdf')
