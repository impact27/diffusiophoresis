
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 14:35:39 2016

@author: quentinpeter
"""
#%%


from glob import glob
import os
from matplotlib.pyplot import figure, plot, imshow
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
import numpy as np
import json
#Load local libraries
import registrator.image as ir
import background_rm as rmbg
import scipy
from matplotlib.colors import LogNorm
from kCommun import get_normalized_side_channel, getBase
from tifffile import imread




threshold = 500
outfolder = 'output'
mdfns = 'Data/20161201/*metadata.json'
maskmargin = 20


#Get functions
cmap = matplotlib.cm.get_cmap('viridis')
gfilter=scipy.ndimage.filters.gaussian_filter1d

#%% Treat filenames
mdfns = glob(mdfns)
mdfns = [os.path.abspath(fn) for fn in mdfns]
outfolder = os.path.abspath(outfolder)
if not os.path.isdir(outfolder):
    os.mkdir(outfolder)

#%% Treat data
All_threshpos=[]
All_times=[]
All_integral=[]
All_label=[]
for i, fnmd in enumerate(mdfns):
    #Get info from metadata
    with open(fnmd) as f:
        Metadata = json.load(f)
    fns = Metadata["Images File Names"]
    pixs = Metadata["Pixel Size [m]"]*1e6
    times = np.asarray(Metadata["Times [s]"])
    inittime = Metadata['Contact Time [s]']
    times -= inittime-1
    times = times/60
    Cin = Metadata["Analyte Concentration In [M]"]*1e3
    Cout = Metadata["Analyte Concentration Out [M]"]*1e3
    firstframe =  Metadata['First Good Frame']
    
    #Load fns    
    fns = [os.path.join(os.path.dirname(fnmd), fn) for fn in fns]
    ims = np.asarray([imread(fn) for fn in fns], dtype = float)
    ims = ims[firstframe:]
    times = times[firstframe:]
    
    
    if "Background File Name" in Metadata:
        #Load bg image
        bgfn = Metadata["Background File Name"]
        bgfn = os.path.join(os.path.dirname(fnmd), bgfn)
        bg = mpimg.imread(bgfn)
        
        #Get best guess for mask
        mask = np.ones(ims[0].shape)
        sideprof = np.nanmean(ims[-1], 1)
        thres = 2*np.median(sideprof)-np.min(sideprof)
        mask[sideprof>thres]=0
        mask=mask>0
        
        #Flatten last image
        im=rmbg.remove_curve_background(ims[-1], bg, maskim=mask)
        
#        figure()
#        imshow(im)
#        imshow(mask, alpha=.5, cmap='Reds')
        
        #Get infos from last image
        a, first, last, ySide = get_normalized_side_channel(im)

        mask = np.ones(im.shape)
        mask[:ySide+maskmargin,:] = 0
        mask[:,first-maskmargin:last+maskmargin] = 0
        mask=ir.rotate_scale(mask,a,1)
        mask=mask>0
        
        for im in ims:
            im[:,:]=rmbg.remove_curve_background(im,bg,maskim=mask)
            
#        m2=np.zeros(im.shape,dtype=float)
#        m2[mask]=np.nan
#        figure()
#        imshow(ir.rotate_scale(im, -a, 1))
#        imshow(ir.rotate_scale(m2, -a, 1), alpha=.5, cmap='Reds')

    #Get channel position from last image
    a, first, last, ySide = get_normalized_side_channel(ims[-1])
    
    X=np.arange(ims[0].shape[0])*pixs
    X-=ySide*pixs
    
    threshpos=[]
    integral=[]
    list_Y=[]
    mask=np.ones((ims[0].shape[1],))
    mask[first-30:last+30]=0
    mask=mask>0
    
    for i, im in enumerate(ims):
        im[:]=ir.rotate_scale(im,-a,1, borderValue=np.nan)
#        figure()
#        imshow(im)
        backprof=getBase(im,first,last, mask)
        Y = np.nanmean(im[:,first:last],1)
        Y /= np.nanmean(Y[X<0])
        backprof /= np.nanmean(backprof[X<0])
        list_Y.append(Y)
        pos=np.where((Y-threshold)<0)
        if len(pos[0])>0:
            threshpos.append((pos[0][0]-ySide)*pixs)
        else:
            threshpos.append(0)
        integral.append(np.nansum(Y[X>0]-backprof[X>0])*(X[1]-X[0]))
    
    
    figure()
    #Create dummy colormap for times
    norm=LogNorm(vmin=times.min(), vmax=times.max())
    colors=imshow(times[:,None], cmap=cmap, norm=norm)
    plt.clf()
    #Plot curves
    for i, Y in enumerate(list_Y):
        plot(X, gfilter(Y, 2),c=cmap(norm(times[i])),label="%.1fs"%times[i])
    #Plot background profiles
    plot(X,  gfilter(backprof, 2), 'r')
#    plt.xlim((-200,600))
#    plt.ylim((-.2,1.2))
    plt.xlabel(r'distance [$\mu$m]', fontsize=18) 
    plt.ylabel(r'Intensity', fontsize=18)    
    #plt.legend(loc=1)

    myTitle=('IN ' + str(Cin) + 'mM NaCl, OUT ' + str(Cout) +'mM NaCl')
    plt.title(myTitle, fontsize=18)
    plt.colorbar(colors).set_label(label='time [min]',size=18) 
    
    plot([0, 0], [np.nanmin(Y), np.nanmax(Y)],'black')

    plt.yticks(plt.yticks()[0][:-1])
    

    figfn = ('IN' + str(Cin) + 'mMNaCL_OUT' + str(Cout) + 'mMNaCl.pdf')
    plt.savefig(os.path.join(outfolder, figfn), bbox_inches='tight')
    All_threshpos.append(threshpos)
    All_integral.append(integral)
    All_times.append(times)
    All_label.append('Out ' + str(Cout) +'mM, in ' + str(Cin) +'mM')


#%%
Y=np.nanmean(im[ySide:,first-30:last+30],0)
X=pixs*np.arange(len(Y))
figure()
plot(X,Y)
plot(X,Y*0+np.nanmean(im[ySide:,first:last]))
plot(pixs*np.array([30,30]),[Y.min(),Y.max()],'k')
plot(pixs*np.array([30+last-first,30+last-first]),[Y.min(),Y.max()],'k')
plot(pixs*np.array([30,30])+50,[Y.min(),Y.max()],'y')

#%%
#figure()
#for times,tp,e in zip(All_times, All_threshpos,End):
#    x=times
#    y=np.array(tp)
#    plot(x,y,'x')
#
#plt.xlabel('time [min]', fontsize=14)
#plt.ylabel(r'Position[$\mu$m]', fontsize=14)
#plt.legend(labl,loc='best')
#plt.savefig('Comp'+str(threshold)+'thresh.pdf',bbox_inches='tight')

#%%
AS=np.argsort(All_label)
All_times=np.array(All_times)
All_integral=np.array(All_integral)
All_label=np.array(All_label)
figure()
idx=0
for times,i in zip(All_times[AS], All_integral[AS]):
    idx+=1
    x=times
    y=np.array(i)
    plt.plot(x,y-y[-7],'x')

plt.xlabel('Time [min]', fontsize=18)
plt.ylabel('Integrated Intensity', fontsize=18)
#legend=plt.legend(('  Positive gradient','        Flat gradient', 'Negative gradient' )
#    ,loc='best', fontsize=16)
#legend=plt.legend(('Negative gradient', 'Positive gradient' )
#    ,loc='best', fontsize=16)
legend=plt.legend(All_label[AS]
    ,loc='best', fontsize=12)
plt.savefig(os.path.join(outfolder, 'IntegratedI.pdf'),bbox_inches='tight')
