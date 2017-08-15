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
#%%

#%%

for basefn in [os.path.abspath(f) for f in glob("Data/20170405/i*/i*")]:
    
    imagesfolder = basefn
    basefn = os.path.dirname(basefn)
    #Metadata file name. Whatever/YYYMMDD/NAME.json
    mdfn = basefn+'_metadata.json'
    
    #Images and corresponding BG (or None) relative to metadata
    fns = os.path.join(os.path.basename(basefn),
                       os.path.basename(imagesfolder),
                       'Pos0/img*.tif')
    bgfn = os.path.join(os.path.basename(basefn),
                        'bg/bg/Pos0/img_000000000_Default_000.tif')
    
    resultsout = re.findall('o([\dp]+)mMNaCl(?:([\dp]+)uMBSA)?', imagesfolder)
    resultsin = re.findall('i([\dp]+)mMNaCl(?:([\dp]+)uMBSA)?', imagesfolder)
    AnalyteIn_M = float(re.sub('p', '.', resultsin[0][0]))*1e-3
    AnalyteOut_M = float(re.sub('p', '.', resultsout[0][0]))*1e-3
    ProteinIn_M = 0
    ProteinOut_M = 0
    if resultsin[0][1]!='':
        ProteinIn_M = float(re.sub('p', '.', resultsin[0][1]))*1e-6
    if resultsout[0][1]!='':
        ProteinOut_M = float(re.sub('p', '.', resultsout[0][1]))*1e-6
    
    FirstGoodFrame = 0
    contactTime = 0 #s
    Operator = 'Raphael Jacquat'
    height_m = 50e-6
    Q_ulph = 0
    Success = 3#1-3
    Analyte = 'NaCl'
    Protein = 'BSA'
    Zoom = 10
    pixelsize_m = 1.6e-6 #Base is 8 for Q camera and 4.54 for Evolve /Zoom * Binning
    Buffer_Info = "No Buffer"
    Solvent = 'H2O'
    pump_infos = 'Hands'
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
    
    
    
    
    # =============================================================================
    # 
    # =============================================================================
    
    #Change WD
    path, mdfn = os.path.split(mdfn)
    os.chdir(path)
    
    fns = sorted(glob(fns))
    
    if len(fns)>0:
        #Get date
        date = path[-8:]
        
        #Extract interesting infos from imagej metadata
        Time = np.zeros(len(fns), dtype=float)
        Exposure = np.zeros(len(fns), dtype=float)
        Binning = np.zeros(len(fns), dtype=int)
        for i, fn in enumerate(fns):
            with TiffFile(fn) as tif:
                header = metaDataDict(tif)
                Time[i] = (header['ElapsedTime-ms'])
                if 'Exposure-ms' in header:
                    Exposure[i] = (header['Exposure-ms'])
                else:
                    Exposure[i] = (header['QCamera-Exposure'])
                
                Binning[i] = (header['Binning'])
                
        Time/=1000
        Exposure/=1000
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
        
                
        Metadata = {}
        Metadata['Operator'] = Operator
        Metadata['Date'] = date
        Metadata['Success [1-3]'] = Success
        Metadata['Analyte Type'] = Analyte
        Metadata['Analyte Concentration In [M]'] = AnalyteIn_M
        Metadata['Analyte Concentration Out [M]'] = AnalyteOut_M
        Metadata['Proteins Type'] = Protein
        Metadata['Proteins Concentration In [M]'] = ProteinIn_M
        Metadata['Proteins Concentration Out [M]'] = ProteinOut_M
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
        Metadata['Images File Names'] = fns
        if bgfn is not None:
            Metadata['Background File Name'] = bgfn
        Metadata['Times [s]'] = Time
        
        with open(mdfn, 'w') as f:
            json.dump(Metadata, f, indent=4)

