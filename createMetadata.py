# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 11:53:18 2017

@author: quentinpeter
"""
from glob import glob
import json
from tifffile import TiffFile
from MMtiff import metaDataDict
import numpy as np
import os
import re

# %%
# ['NaCl', 'LiCl', 'KIO3', 'KCl']
Analyte_list = ['GLU', 'GLY', 'Urea', 'NaOH', 'HCl']
Protein_list = ['Lys', 'Thy']  # ['Thy', 'Myo', 'Ubi', 'MYO', 'INS']
FirstGoodFrame = 0
contactTime = 0  # s
Operator = 'Quentin Peter'
height_m = 50e-6
Q_ulph = 300
Success = 3  # 1-3
Zoom = 10
pixelsize_m = 1.65e-6  # Base is 8 for Q camera and 4.54 for Evolve /Zoom * Binning
channel_width = 50e-6
channel_height = 50e-6
channel_length = 500e-6

imagesfolderfn = "../Data/20180508/*/"
imagefn = "*.tif"
bglocfn = 'bg*/**/*.tif'

Buffer_Info = "No Buffer"
Solvent = 'H2O'
pump_infos = 'Nemesis'
# Devices list
devices_list = [
    '500umX50um',
    '2000umX200um',
    '2650umx50um_curved',
    '20mmx50um_curved',
    '500umX50um_reservoir',
    '2650umx50um_curved_reservoir',
]

device = devices_list[0]

Analyte_re = "(i|o)(" + '|'.join(Analyte_list)+')' + r"_([\dp]+)(uM|mM|M|gpl)"
Protein_re = "(i|o)(" + '|'.join(Protein_list)+')' + r"_([\dp]+)(uM|mM|M|gpl)"
regexp = "(?:" + Analyte_re + "|"\
    + Protein_re + "|"\
    + Analyte_re + Protein_re + "|"\
    + Protein_re + Analyte_re + ")"

for imagesfolder in [os.path.abspath(f) for f in glob(imagesfolderfn)]:

    mdfolder = os.path.dirname(imagesfolder)

    # Metadata file location
    mdfn = imagesfolder + '_metadata.json'

    # Images and corresponding BG (or None) relative to metadata
    fns = os.path.join(os.path.basename(mdfolder),
                       os.path.basename(imagesfolder))
    fns = os.path.join(fns, imagefn)

    bgfn = os.path.join(os.path.basename(mdfolder),
                        os.path.basename(imagesfolder),
                        bglocfn)

    regexp_results = re.findall(regexp, imagesfolder)

    Analyte = ''
    Protein = ''
    AnalyteIn_M = 0
    ProteinIn_M = 0

    AnalyteOut_M = 0
    ProteinOut_M = 0

    protein_unit = 'M'
    analyte_unit = 'M'

    for result in regexp_results:
        for i in range(0, len(result), 4):
            if result[i] != '':
                is_gpl = False
                number = float(re.sub('p', '.', result[i+2]))
                if result[i+3] == 'uM':
                    number *= 1e-6
                elif result[i+3] == 'mM':
                    number *= 1e-3
                elif result[i+3] == 'gpl':
                    is_gpl = True
                else:
                    assert result[i+3] == 'M'
                if result[i+1] in Protein_list:
                    Protein = result[i+1]
                    if result[i] == 'i':
                        ProteinIn_M = number
                    elif result[i] == 'o':
                        ProteinOut_M = number
                    if is_gpl:
                        protein_unit = 'g/l'
                elif result[i+1] in Analyte_list:
                    Analyte = result[i+1]
                    if result[i] == 'i':
                        AnalyteIn_M = number
                    elif result[i] == 'o':
                        AnalyteOut_M = number
                    if is_gpl:
                        analyte_unit = 'g/l'

    # Change WD
    fullfns = sorted(glob(os.path.join(os.path.dirname(mdfolder), fns)))

    if len(fullfns) > 0:
        # Get date
        date = re.findall(r'201\d\d\d\d\d', mdfolder)[-1]

        # Extract interesting infos from imagej metadata
        Time = np.zeros(len(fullfns), dtype=float)
        Exposure = np.zeros(len(fullfns), dtype=float)
        Binning = np.zeros(len(fullfns), dtype=int)
        for i, fn in enumerate(fullfns):
            with TiffFile(fn) as tif:
                header = metaDataDict(tif)

                if 'ElapsedTime-ms' in header:
                    Time[i] = (header['ElapsedTime-ms'])
                elif 'CustomIntervals_ms' in header:
                    Time = np.cumsum(header['CustomIntervals_ms'])

                if 'Exposure-ms' in header:
                    Exposure[i] = (header['Exposure-ms'])
                elif 'QCamera-Exposure' in header:
                    Exposure[i] = (header['QCamera-Exposure'])

                if 'Binning' in header:
                    Binning[i] = (header['Binning'])

        Time /= 1000
        Exposure /= 1000
        if np.all(Exposure == Exposure[0]) or np.all(np.isnan(Exposure)):
            Exposure = Exposure[0]
        else:
            Exposure = Exposure.tolist()

        if np.all(Binning == Binning[0]):
            Binning = int(Binning[0])
        else:
            Binning = Binning.tolist()

        Time = Time - Time[0]
        Time = Time.tolist()

        fullfns = [os.path.relpath(fn, os.path.dirname(mdfn))
                   for fn in fullfns]
        if len(fullfns) == 1:
            fullfns = fullfns[0]

        Metadata = {}
        Metadata['Operator'] = Operator
        Metadata['Date'] = date
        Metadata['Success [1-3]'] = Success
        Metadata['Analyte Type'] = Analyte
        Metadata[f'Analyte Concentration In [{analyte_unit}]'] = AnalyteIn_M
        Metadata[f'Analyte Concentration Out [{analyte_unit}]'] = AnalyteOut_M
        Metadata['Proteins Type'] = Protein
        Metadata[f'Proteins Concentration In [{protein_unit}]'] = ProteinIn_M
        Metadata[f'Proteins Concentration Out [{protein_unit}]'] = ProteinOut_M
        Metadata['Flow Rate [ulph]'] = Q_ulph
        Metadata['Buffer Infos'] = Buffer_Info
        Metadata['Device Type'] = device
        Metadata['Device Height [m]'] = height_m
        Metadata['Binning'] = Binning
        Metadata['Zoom'] = Zoom
        Metadata['Pixel Size [m]'] = pixelsize_m
        Metadata['Exposure Time [s]'] = Exposure
        Metadata['Solvent'] = Solvent
        Metadata['Pump Infos'] = pump_infos
        Metadata['First Good Frame'] = FirstGoodFrame
        Metadata['Contact Time [s]'] = contactTime
        Metadata['Images File Names'] = fullfns
        Metadata['Dead end width [m]'] = channel_width
        Metadata['Dead end height [m]'] = channel_height
        Metadata['Dead end length [m]'] = channel_length
        if bgfn is not None:
            bgfn = os.path.join(os.path.dirname(mdfolder), bgfn)
            bgfn = glob(bgfn, recursive=True)
            bgfn = os.path.relpath(bgfn[0], os.path.dirname(mdfn))
            Metadata['Background File Name'] = bgfn
        Metadata['Times [s]'] = Time

        with open(mdfn, 'w') as f:
            json.dump(Metadata, f, indent=4)
