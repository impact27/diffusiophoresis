# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 13:58:50 2018

@author: quentinpeter
"""

#%% from matplotlib.pyplot import plot, imshow, show, colorbar


from glob import glob
import os
from matplotlib.pyplot import figure, plot, imshow
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import json
#Load local libraries
from matplotlib.colors import LogNorm



from kCommun import get_profs, get_images, plot_and_save_diffusiophoresis
#%%

dateignore = [ '20160907', '20160908', '20170315']#'20170315',
plotanalyte = ['LiCl', 'KIO3', 'KCl', 'NaCl', 'LiCl', 'KIO3', 'KCl', 'Dextrose', 'CsCl']#'NaCl', 'LiCl', 'KIO3', 'KCl', 'Dextrose', 'CsCl'
threshold = .5
outfolder = 'output'#'best' + "".join(plotanalyte)
mdfns = sorted(glob('../Data/20171116/*mall*/*metadata.json'))
maskmargin = 20


#Get functions
cmap = matplotlib.cm.get_cmap('plasma')

#%% Treat filenames
#mdfns = glob(mdfns)
mdfns = [os.path.abspath(fn) for fn in mdfns]
outfolder = os.path.abspath(outfolder)
if not os.path.isdir(outfolder):
    os.mkdir(outfolder)
    

    
#%% Treat data

norm=LogNorm(vmin=.1, vmax=10)

skip = 0
flatten=True
fnmd = mdfns[-1]

    
with open(fnmd) as f:
    Metadata = json.load(f)

success = Metadata["Success [1-3]"]
date = Metadata['Date']

analyte = Metadata["Analyte Type"]

ims, channel_position_px, times = get_images(fnmd, flatten=flatten)
X_pos, profiles, background_profiles = get_profs(
        ims, channel_position_px, Metadata, maskmargin=maskmargin)
plot_and_save_diffusiophoresis(ims, channel_position_px, times,
                               X_pos, profiles, background_profiles,
                               fnmd, maskmargin, outfolder)
    

    
    


#%%
idx = 90
im = ims[-1]
[first, last, ySide] = channel_position_px
pixs = Metadata['Pixel Size [m]']*1e6
#Y=savgol_filter(im, 21, 5, axis=0)[ySide+idx,first-maskmargin:last+1+maskmargin]
#filt = savgol_filter(Y, 11, 5)
#filt = savgol_filter(filt, 11, 5)
#filt = savgol_filter(filt, 11, 5)

smallim = im[ySide+idx-50:ySide+idx+50, first-20:last+20]
width = last-first
widthprof = np.mean(smallim,0)
xpos = (np.arange(len(widthprof))-len(widthprof)/2)*pixs

figure()
plot(xpos, widthprof)
#plot(filt)
plot(np.ones(2)*(-width/2)*pixs,[widthprof.min(),widthprof.max()],'r')
plot(np.ones(2)*(width/2)*pixs,[widthprof.min(),widthprof.max()],'r')

plt.xlabel("Position [$\mu$m]")
plt.ylabel("Fluorescent intensity")

origin, size = plt.gca().get_position().get_points()



ax2 = plt.axes([.7, .5, .2, .35])
ax2.imshow(smallim)
ax2.plot([20, width+20], [50, 50],'r', alpha=.7)
ax2.plot([20, 20], [40, 60],'r', alpha=.7)
ax2.plot([width+20, width+20], [40, 60],'r', alpha=.7)
ax2.set_xticks([])
ax2.set_yticks([])


#plt.savefig('channel_width.pdf')