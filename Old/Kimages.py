# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 14:35:14 2017

@author: quentinpeter
"""
import numpy as np
import matplotlib.image as mpimg
import background_rm as rmbg
from glob import glob
from matplotlib.pyplot import figure, imshow, plot
from kCommun import get_normalized_side_channel, getBase

fns='Data/20161201/i200o0p02/i200mMNaCl_o0p02mMNaCl7uMBSA_1/Pos0/*.tif'
bgfn='Data/20161201/i200o0p02/bg/bg/Pos0/img_000000000_Default_000.tif'

bg=np.asarray(mpimg.imread(bgfn))
ims=np.asarray([mpimg.imread(fn)for fn in glob(fns)])
mask=np.zeros(bg.shape)
mask[150:,:]=1
mask[:,200:300]=0
mask=mask>0
for im in ims:
    figure()

    data=rmbg.remove_curve_background(im,bg,maskim=mask)
    imshow(data)
    
    #%%
a, first,last,ySide=get_normalized_side_channel(data)
figure()
imshow(data)
plot([0,500],[ySide,ySide],'r--')
plot([first,first],[0,500],'r--')
plot([last,last],[0,500],'r--')
#%%
figure()
imshow(mask)