# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 14:35:39 2016

@author: quentinpeter
"""
# %% from matplotlib.pyplot import plot, imshow, show, colorbar
import os
from matplotlib.pyplot import figure, plot, imshow
import matplotlib.pyplot as plt
import cv2
import numpy as np
import json
# Load local libraries
from dph_processing import get_profs, get_images, plot_and_save_diffusiophoresis

# %%
outfolder = '../Plots/timelapse'
maskmargin = 20

flatten = True
fnmd = '../Data/Experiments/20171116/small_channel/i200mMKIO3_o10uMLYS_metadata.json'


outfolder = os.path.abspath(outfolder)
if not os.path.isdir(outfolder):
    os.mkdir(outfolder)


# %% Treat data

with open(fnmd) as f:
    Metadata = json.load(f)

ims, channel_position_px, times = get_images(fnmd, flatten=flatten)
X_pos, profiles, background_profiles = get_profs(
    ims, channel_position_px, Metadata, maskmargin=maskmargin)


# Plot curves
plot_and_save_diffusiophoresis(ims, channel_position_px, times,
                               X_pos, profiles, background_profiles,
                               fnmd, maskmargin, outfolder)

# %% Plot timeline
[first, last, ySide] = channel_position_px
pixs = Metadata['Pixel Size [m]']*1e6
plotidx = np.arange(6, len(ims))
imsdisplay = ims[:, ySide-20:ySide+int(500/pixs)+20,
                 first-maskmargin:last+1+maskmargin][plotidx]
imsdisplay = np.concatenate(imsdisplay, axis=1)
totalwidth = last-first+1+2*maskmargin


imsdisplay = cv2.GaussianBlur(imsdisplay, (5, 5), 0)


figure(figsize=(6, 3))
imshow(imsdisplay, vmin=0, vmax=0.95*np.max(imsdisplay))
plt.xticks(np.arange(len(plotidx))*totalwidth+totalwidth/2,
           ["{:.1f}".format(t) for t in times[plotidx]])


plt.yticks(20 + np.arange(6)*500/pixs/5, np.arange(6)*100)

plt.xlabel("Time [min]")
plt.ylabel("Position [$\mu$m]")
plt.tight_layout()
plt.savefig(os.path.join(outfolder, "timelapse.pdf"))


# %%
im = cv2.GaussianBlur(ims[-2], (3, 3), 0)

left = pixs*(first-40)
right = pixs*(last-40)
top = pixs*(ySide-80)
bot = 500 + top


figure(figsize=(4, 4))
imshow(im[80:480, 40:440], vmin=0, vmax=0.95 *
       np.nanmax(im), extent=(0, pixs*400, pixs*400, 0))
plot([left, left], [top, bot], 'r--')
plot([right, right], [top, bot], 'r--')
plot([left, right], [bot, bot], 'r--')
plt.xlabel('[$\mu$m]')
plt.ylabel('[$\mu$m]')
plt.tight_layout()
plt.savefig(os.path.join(outfolder, "timelapse_single.pdf"))
