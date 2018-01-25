# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 17:38:35 2016

@author: quentinpeter
"""

#%%
fns='Data/20161118/i0p02o0p02/i0p02mMNaCl_o0p02mMNaCl5uMBSA_1/Pos0/*.tif'
bgfn='Data/20161118/i0p02o0p02/bg/Pos0/img_000000000_Default_000.tif'
fns='Data/20161118/i200o200/i200mMNaCl_o200mMNaCl5uMBSA_2/Pos0/*.tif'
bgfn='Data/20161118/i200o200/background/Pos0/img_000000000_Default_000.tif'
zoom=10
#%%
from matplotlib.pyplot import figure, plot, imshow, show,close,semilogy, hold
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
import numpy as np
import importlib
#Load local libraries
import registrator.image as ir
import registrator.channel as cr
import background_rm as rmbg
from glob import glob
import scipy
import MMtiff
from matplotlib.colors import LogNorm
import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path']='/usr/local/bin/ffmpeg'
import matplotlib.animation as animation
import cv2
import warnings
warnings.filterwarnings('ignore', 'Mean of empty slice',RuntimeWarning)

cmap = matplotlib.cm.get_cmap('inferno')
gfilter=scipy.ndimage.filters.gaussian_filter1d

#%%Reload them in case they changed
importlib.reload(ir)
importlib.reload(cr)
importlib.reload(MMtiff)

#%%
def get_normalized_side_channel(im, a=None):
    if a is None: 
        #remove eventual angle
        eim=cr.Scharr_edge(im)
        nanmask=np.isnan(eim)
        eim[nanmask]=0
        a=ir.orientation_angle(eim)
        #rotate
        im=ir.rotate_scale(im,-a,1, borderValue=np.nan)
    
        #remove eventual angle
        maxSide=[im[:20,:],im[:,-20:],im[-20:,:],im[:,:20]]
        maxSide=np.asarray([np.nansum(i) for i in maxSide])
        a+=np.pi/2*maxSide.argmax()
        im=np.rot90(im,maxSide.argmax())
    else:
        #rotate
        im=ir.rotate_scale(im,-a,1, borderValue=np.nan)
    
    #find the channel position
    ySide=np.nanargmin(gfilter(np.diff(np.nanmean(im,1)),21))+1 
    border=int(np.ceil(np.abs(np.tan(ir.clamp_angle(a))*im.shape[0])))
    xSlice=gfilter(np.diff(np.nanmean(im[ySide:,:],0)),1)[border:-border]
    
    #extract channel pos
    first=np.nanargmax(xSlice)+border
    last=np.nanargmin(xSlice)+1+border
                 
    return a, int(first),int(last), int(ySide)



#%%
tiff=MMtiff.tiffGlob(fns)[15:]

pixs=MMtiff.pixSize(tiff[0]['header'],zoom)
headers=[im['header'] for im in tiff]
ims=[np.rot90(im['data'],0) for im in tiff]
     
bg=mpimg.imread(bgfn)
im0=ims[-1].copy()
im=rmbg.remove_curve_background(ims[-1],bg)
a, first,last,ySide=get_normalized_side_channel(im)

mask=np.ones(im.shape)
mask[:ySide+20,:]=0
mask[:,first-20:last+20]=0

mask=ir.rotate_scale(mask,a,1)
mask=mask>0

im2=im.copy()
im[mask]=np.nan
figure()
imshow(im)

for im in ims:
    infoDict={}
    im[:,:]=rmbg.remove_curve_background(im,bg,maskim=mask,infoDict=infoDict)



#%
times=np.asarray([im['time'] for im in tiff]) 
times-=-1
times=times/60
figure()
norm=LogNorm(vmin=times[0], vmax=times[-1])
colors=imshow(times[:,None], cmap=cmap, norm=norm)
plt.clf()
list_Y=[]
for i, im in enumerate(ims):
    im=ir.rotate_scale(im,-a,1, borderValue=np.nan)
    Y = np.nanmean(im[:,first:last],1)
#    Y-=Y[np.isfinite(Y)][-1]
#    Y/=np.mean(Y[np.logical_and(X<0,np.isfinite(Y))])
    X=(np.arange(im.shape[0])-ySide)*pixs
    plot(X,Y,c=cmap(norm(times[i])),label="%.1fs"%times[i])
    list_Y.append(Y)
plt.xlabel(r'distance [$\mu$m]', fontsize=14) 
plt.ylabel(r'Intensity', fontsize=14)    
#plt.legend(loc=1)
plt.title('in= 2mM NaCl, out=.02mM NaCl')
plt.colorbar(colors,label='time [min]')
plot([0, 0], [np.nanmin(Y), np.nanmax(Y)],'black')
#endCH=pixs*10*(last-first)+startCH
#plot([endCH, endCH], [np.nanmin(Y), np.nanmax(Y)],'black')
#plt.savefig("prot_i2o002.pdf",bbox_inches='tight')

#%%
#dY=np.diff(list_Y,axis=0)
#figure()
#for i in range(dY.shape[0]):
#    plot(gfilter(dY[i,:],3),c=cmap(norm(times[i])))
##%%
#fig=figure()
#res=[]
#for im in ims:
#    im=ir.rotate_scale(im,-a,1, borderValue=np.nan)
#    res.append((imshow(im,vmin=200,vmax=900),))
#
#im_ani =  animation.ArtistAnimation(fig,res,interval=500,repeat_delay=3000)
#im_ani.save('im.mp4', metadata={'artist':'Guido'})
#%%
import scipy.ndimage.measurements as msr
from scipy.ndimage.morphology import binary_closing as close
prof=np.nanmean(im[ySide+10:,:],0)
prof-=np.nanmin(prof)
X=np.arange(len(prof))
valid=prof>.2*np.nanmax(prof)
valid=close(valid,np.ones((3,)))
lbl,n=msr.label(valid)
valid=lbl==lbl[np.nanargmax(prof)]
Xf=X[valid]
Y=np.log(prof[valid])
fit=np.polyfit(Xf,Y,2)
F=np.poly1d(fit)
fit0=np.exp(F(X))
figure()
plot(prof)
plot(fit0)
#%%
fit=fit0
res=np.zeros((im.shape[0],))
bases=np.zeros((im.shape[0],))
s=np.sum(fit)

for i in range(0,im.shape[0]):

    prof=im[i,:]
    
    if np.isfinite(prof[(first+last)//2]):
        valid=np.logical_and(fit<.01*fit.max(),np.isfinite(prof))
        X=np.arange(len(prof))
        Xp=X[valid]
        Y=prof[valid]
        F=np.poly1d(np.polyfit(Xp,Y,2))
        base=F(X)
        
        valid=fit>.01*fit.max()
        C=np.nansum(fit[valid]*(prof-base)[valid])/np.nansum((fit[valid])**2)
        res[i]=(C*s)*pixs/50
           
      
        bases[i]=F((first+last)//2)
    else:
        res[i]=np.nan

  

bases=np.asarray(bases)
X=(np.arange(len(res))-ySide)*pixs
zero=np.poly1d(np.polyfit(X[X>100],bases[X>100],1))(X)
cen=(first+last)//2
#%
init=np.nanmean(im[:,cen-10:cen+10],1)
figure()
plot(X,init-zero)
plot(X,bases+res-zero)
plot(X,res)
plot(X,bases-zero)
plot([0,0],[0,np.nanmax(res)],'k')
#%%
i=200
prof=im[i,:]

valid=np.logical_and(fit<.01*fit.max(),np.isfinite(prof))
X=np.arange(len(prof))
Xp=X[valid]
Y=prof[valid]
F=np.poly1d(np.polyfit(Xp,Y,2))
base=F(X)

valid=fit>.01*fit.max()
C=np.nansum(fit[valid]*(prof-base)[valid])/np.nansum((fit[valid])**2)
resss=(C*s)*pixs/50
   
  
basesss=F((first+last)//2)
X=np.arange(len(prof))*pixs
figure()
plot(X,prof)
plot(X,C*fit)
lo=X[fit.argmax()]-25
up=lo+50
plot([lo,lo],[0,1],'k')
plot([up,up],[0,1],'k')
#%%
#
#res=np.zeros((im.shape[0],))*np.nan
#bases=np.zeros((im.shape[0],))*np.nan
#centers=np.zeros((im.shape[0],))*np.nan
#stdevs=np.zeros((im.shape[0],))*np.nan
#figure()
#for i in range(im.shape[0]):
#    prof=im[i,:]
#    valid=np.logical_and(fit0<.01*fit0.max(),np.isfinite(prof))
#    if np.sum(np.isfinite(prof))>100:
#        X=np.arange(len(prof))[valid]
#        Y=prof[valid]
#        F=np.poly1d(np.polyfit(X,Y,2))
#        base=F((first+last)//2)
#        bases[i]=base
#         
#    valid=fit0>.01*fit0.max()
#    if np.isfinite(prof[(first+last)//2]):
#        prof=prof-base
#        X=np.arange(len(prof))
#        valid=prof>.2*np.nanmax(prof)
#        valid=close(valid,np.ones((3,)))
#        valid=np.logical_and(valid,prof>0)
#        lbl,n=msr.label(valid)
#        valid=lbl==lbl[np.nanargmax(prof)]
#        Xf=X[valid]
#        Y=np.log(prof[valid])
#        if len(Y)>10:
#            fit=np.polyfit(Xf,Y,2)
#            a=fit[0]
#            centers[i]=fit[1]/(-2*a)
#            stdevs[i]=np.sqrt(1/(-2*a))
#            F=np.poly1d(fit)
#            fit=np.exp(F(X)) 
#            res[i]=np.sum(fit)*pixs/50
#            
#
#           
#
#
#
#bases=np.asarray(bases)
#X=(np.arange(len(res))-ySide)*pixs
#zero=np.nanmean(bases[X>100])
#cen=(first+last)//2
#figure()
#plot(X,np.nanmean(im[:,cen-10:cen+10],1)-zero)
#plot(X,bases+res-zero)
#plot(X,res)
#plot(X,bases-zero)
#plot([0,0],[0,np.nanmax(res)],'k')
#plt.ylim(-100,3000)

