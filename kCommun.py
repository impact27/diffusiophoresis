# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 13:45:05 2016

@author: quentinpeter
"""
import numpy as np
import registrator.image as ir
import registrator.channel as cr
import scipy
import scipy.ndimage.measurements as msr
from scipy.ndimage.morphology import binary_closing as close

gfilter=scipy.ndimage.filters.gaussian_filter1d
    
def get_normalized_side_channel(im, a=None, maxSide=None):
    if a is None: 
        #remove eventual angle
        eim=cr.Scharr_edge(im)
        nanmask=np.isnan(eim)
        eim[nanmask]=0
        a=ir.orientation_angle(eim,rotateAngle=np.pi/4)
        #rotate
        im=ir.rotate_scale(im,-a,1, borderValue=np.nan)
    
        #remove eventual angle
        if maxSide is None:
            half=int(np.mean(im.shape)//2)
            maxSide=[im[:half,:],im[:,-half:],im[-half:,:],im[:,:half]]
            maxSide=np.asarray([np.nansum(i) for i in maxSide])
            maxSide=maxSide.argmax()
        a+=np.pi/2*maxSide
        im=np.rot90(im,maxSide)
    else:
        #rotate
        im=ir.rotate_scale(im,-a,1, borderValue=np.nan)
    
    #find the channel position
    prof=np.diff(np.nanmean(im,1))
    ySide=np.nanargmin(gfilter(prof,21))
    valid=prof<.2*np.nanmin(prof)
    valid=close(valid,np.ones((3,)))
    valid=np.logical_and(valid,prof<0)
    lbl,__=msr.label(valid)
    valid=lbl==lbl[ySide]
    ySide=np.argmax(valid)
    border=int(np.ceil(np.abs(np.tan(ir.clamp_angle(a))*im.shape[0])))
    xSlice=gfilter(np.diff(np.nanmean(im[ySide:,:],0)),1)[border:-border]
   
    #extract channel pos
    first=np.nanargmax(xSlice)+border
    last=np.nanargmin(xSlice)+1+border
                 
    return a, int(first),int(last), int(ySide)

def getBase(im, first, last, channelMask):
    bases = np.nanmean(im[:, :first], 1)
    bases += np.nanmean(im[:, last:], 1)
    bases /=2
#    bases=np.empty((im.shape[0],))*np.nan
#    for i in range(0,im.shape[0]):
#
#        prof=im[i,:]
#        valid=np.logical_and(channelMask,np.isfinite(prof))
#        if np.isfinite(prof[(first+last)//2]) and np.any(valid):
#            
#            X=np.arange(len(prof))
#            Xp=X[valid]
#            Y=prof[valid]
#            F=np.poly1d(np.polyfit(Xp,Y,2))
#            bases[i]=F((first+last)//2)
            
    return bases
