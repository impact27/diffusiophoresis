# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 09:53:27 2016

@author: quentinpeter
"""

import registrator.image as ir
import registrator.channel as cr
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import importlib
import numpy as np
from matplotlib import gridspec
import glob
from scipy.signal import gaussian, convolve
from PIL import Image
#%%
importlib.reload(ir)
importlib.reload(cr)
#%%
import matplotlib

cmap = matplotlib.cm.get_cmap('Spectral')

dirn='Data/20160908/k10_1/*.tif'

idxref=10

index=range(0,70,10)
exposure='Evolve-Exposure'
#'QCamera-Exposure'
#"""
#%%

def gaussFilt(y):
    window=gaussian(5,1)
    window/=window.sum()
    return convolve(y,window,'same')
    
def get_normalized_side_channel(im, a=None):
    if a is None: 
        #remove eventual angle
        a=ir.orientation_angle(cr.Scharr_edge(im))
    
    #rotate
    im=ir.rotate_scale(im,-a,1, borderValue=np.nan)
    
    #find the channel position
    ySide=np.nanargmin(gaussFilt(np.diff(np.nanmean(im,1))))+1
    xSlice=gaussFilt(np.diff(np.nanmean(im[ySide:,:],0))) 
    
    #extract channel pos
    first=np.nanargmax(xSlice)
    last=np.nanargmin(xSlice)+1
    mask=np.zeros(im.shape)
    mask[ySide:,first:last]=1  
    bgmask=1-mask
    bgmask[:ySide+20,:]=0
    mainmask=np.zeros(mask.shape)
    mainmask[:ySide]=1
    return a,mask, bgmask, mainmask, im

#%%


def metaDataDict(tag):
    import json
    #https://github.com/imagej/imagej1/blob/master/ij/io/TiffDecoder.java
    """
    1st: header: IJIJ + n*(type + count)
        type can be "info" (Info image property)
                    "labl" (slice labels)
                    "rang" (display ranges)
                    "luts" (channel LUTs)
                    "roi " (ROI)
                    "over" (overlay)
    sum count should correspond to # sizes
    if info: add odd and even and turn to string (one of them is 0x0)
    """
    #Get metadata count and data
    tag=dict(tag)
    mdsize=tag[50838]
    metadata=tag[50839]
    
    #check that the header is the only implemented header
    assert(metadata[:mdsize[0]]==b'IJIJ'+b'info'+(1).to_bytes(4,'big'))
    
    #retreve data
    header=metadata[mdsize[0]:]
    header=header[(header[0]==0)::2].decode('utf-8')
    return json.loads(header)
    

#%%
base=np.nan
bg=np.asarray([])
main=np.asarray([])
integral=np.asarray([])

files=glob.glob(dirn)
aref,*__ = get_normalized_side_channel(np.array(Image.open(files[idxref])))
plt.figure(0)
plt.clf()
plt.figure(1)
plt.clf()

#%%
for cidx,fn in enumerate([files[i] for i in index]):
    #get image and orientate it    
    img = Image.open(fn)
    header=metaDataDict(img.tag)
    im=np.array(img, dtype='float')
    
    #compute rotation and channel place for first image
    a, mask, bgmask, mainmask, im = get_normalized_side_channel(im,aref)
    
    imoe=2**14-2
    im[im>imoe]=np.inf
    im=im/float(header[exposure])
     
    plt.figure(0)
    
    #plot line
    plt.subplot(221)
    mask[mask==0]=np.nan
    channel=im*mask
    data=np.nanmean(channel,1)
    plt.hold(True)
    plt.plot(data , color=cmap(cidx/len(index)))    
        
    #plot diff
    plt.subplot(222)
    diff=(data-base)
    plt.plot(gaussFilt(diff), color=cmap(cidx/len(index)))
    integral=np.append(integral,diff[np.isfinite(diff)].mean())
    base=data
    
    #plot background
    plt.subplot(223)
    bg=np.append(bg,np.nanmean(im[bgmask>0]))
    plt.hold(False)
    plt.plot(bg)
    
    
    #plot figure
    plt.subplot(224)
    oeim=im.copy()
    oeim[im==np.inf]=oeim[np.isfinite(im)].max()
    main=np.append(main,np.nanmean(oeim[mainmask>0]))
    plt.hold(False)
    plt.plot(main)
    plt.hold(True)
    
    plt.pause(.1)
    
    plt.figure(1)
    
    plt.subplot(121)
    plt.hold(False)
    plt.plot(integral)
    plt.hold(True)
    
    plt.subplot(122)
    plt.hold(False)
    plt.imshow(im)
    plt.hold(True)
    plt.imshow(mask, alpha=.5)
    plt.pause(.1)
 
