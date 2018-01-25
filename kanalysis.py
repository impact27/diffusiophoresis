
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 14:35:39 2016

@author: quentinpeter
"""
#%% from matplotlib.pyplot import plot, imshow, show, colorbar


from glob import glob
import os
from matplotlib.pyplot import figure, plot, imshow, colorbar, show
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import json
#Load local libraries




from kCommun import get_profs, get_images, plot_and_save_diffusiophoresis
#%%


threshold = .5
outfolder = 'output'
mdfns = sorted(glob('../Data/2017*/*mall*/*_metadata.json'))
#mdfns = glob("../Data/20171129/small_channel/i200mMLiCl_o0p5gplMYO_metadata.json")
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

skip = 0
flatten=True
for metadata_number, fnmd in enumerate(mdfns[skip:]):
    #Get info from metadata
    print(fnmd, metadata_number + skip)
    
    with open(fnmd) as f:
        Metadata = json.load(f)
    
    success = Metadata["Success [1-3]"]
    date = Metadata['Date']
#    print(success) 
    if success < 3:
        continue
    analyte = Metadata["Analyte Type"]
    ims, channel_position_px, times = get_images(fnmd, flatten=flatten)
    X_pos, profiles, background_profiles = get_profs(
            ims, channel_position_px, Metadata, maskmargin=maskmargin)
    plot_and_save_diffusiophoresis(ims, channel_position_px, times,
                                   X_pos, profiles, background_profiles,
                                   fnmd, maskmargin, outfolder)
    
    plt.show()
    
    


