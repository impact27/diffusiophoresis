# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 14:01:41 2018

@author: quentinpeter
"""


from glob import glob
import os
from matplotlib.pyplot import figure, plot, imshow
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import json
# Load local libraries
from matplotlib.colors import LogNorm


from dph_processing import get_profs, get_images, plot_and_save_diffusiophoresis
# %%

dateignore = ['20160907', '20160908', '20170315']  # '20170315',
plotanalyte = ['KIO3']  # 'NaCl', 'LiCl', 'KIO3', 'KCl', 'Dextrose', 'CsCl'
prot_list = ['BSA']
threshold = .5
outfolder = 'LiCl'  # 'best' + "".join(plotanalyte)

# for folder in sorted(glob('../Data/2017*/')):
#    folder = '../Data/2017*/'
folder = '../Data/Experiments/20170517/'
mdfns = sorted(glob(os.path.join(folder, '*mall*/*metadata.json')))
maskmargin = 20


# Get functions
cmap = matplotlib.cm.get_cmap('plasma')

# %% Treat filenames
#mdfns = glob(mdfns)
mdfns = [os.path.abspath(fn) for fn in mdfns]
outfolder = os.path.abspath(outfolder)
if not os.path.isdir(outfolder):
    os.mkdir(outfolder)

# %% Treat data
All_threshpos = []
All_times = []
All_integral = []
All_label = []
All_Cin = []
All_Cout = []
All_Cpout = []
norm = LogNorm(vmin=.1, vmax=10)

skip = 2
flatten = True
for metadata_number, fnmd in enumerate(mdfns):

    with open(fnmd) as f:
        Metadata = json.load(f)

    success = Metadata["Success [1-3]"]
    date = Metadata['Date']
    if success < 3:
        print(success)
        continue
    analyte = Metadata["Analyte Type"]
    if analyte not in plotanalyte:
        print(analyte)
        continue
    if date in dateignore:
        continue
    if Metadata['Proteins Type'] not in prot_list:
        print(Metadata['Proteins Type'])
        continue
    if np.abs(Metadata['Proteins Concentration Out [M]'] - 1e-5) > 1e-6:
        print(Metadata['Proteins Concentration Out [M]'])
        continue
    ims, channel_position_px, times = get_images(fnmd, flatten=flatten)
    X_pos, profiles, background_profiles = get_profs(
        ims, channel_position_px, Metadata, maskmargin=maskmargin)

    valid = np.logical_and(X_pos > 0, X_pos < 500)
    assert len(times) == len(profiles)
    All_times.append(times)
    All_threshpos.append(np.sum(
        X_pos[np.newaxis, valid]*profiles[..., valid], -1)/np.sum(profiles[..., valid], -1))
    All_integral.append(
        np.sum(profiles[..., valid] - background_profiles[..., valid], -1))
    All_Cin.append(Metadata['Analyte Concentration In [M]'])
    All_Cout.append(Metadata['Analyte Concentration Out [M]'])
    All_Cpout.append(Metadata['Proteins Concentration Out [M]'])

# %%
AS = np.argsort(1e6*np.array(All_Cin) /
                (np.array(All_Cout)+0.01)+np.array(All_Cin))[::-1]
Times = np.array(All_times)[AS]
integrals = np.array(All_integral)[AS]
labels = ["In: {:.3g}mM; Out: {:.3g}mM".format(
    Cin*1e3, Cout*1e3) for Cin, Cout, Cpout in zip(All_Cin, All_Cout, All_Cpout)]
labels = np.array(labels)[AS]
threshpos = np.array(All_threshpos)[AS]
unique_label, return_index = np.unique(labels, return_index=True)
unique_label = unique_label[np.argsort(return_index)]

if len(AS) > 0:
    figure()
    idx = 0
    lastlbl = None
    for times, i, lbl in zip(Times, integrals, labels):
        idx += 1
        x = times
        y = i/500
        color = 'C{}'.format(np.argwhere(lbl == unique_label)[0, 0] % 10)
        if lbl == lastlbl:
            lbl = None
        else:
            lastlbl = lbl
        plt.semilogx(x, y, '.--',  c=color, label=lbl)

    plt.xlabel('Time [min]')
    plt.ylabel('Integrated Intensity')
    # legend=plt.legend(('  Positive gradient','        Flat gradient', 'Negative gradient' )
    #    ,loc='best', fontsize=16)
    # legend=plt.legend(('Negative gradient', 'Positive gradient' )
    #    ,loc='best', fontsize=16)
    legend = plt.legend()
#    plt.title(folder)
    plt.savefig(os.path.join(outfolder, 'IntegratedKIO3.pdf'),
                bbox_inches='tight')
