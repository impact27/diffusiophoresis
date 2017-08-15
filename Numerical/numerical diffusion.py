# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 15:09:43 2017

@author: quentinpeter
"""
import numpy as np

import diffusiosim as ds
import matplotlib.pyplot as plt
import matplotlib
cmap = matplotlib.cm.get_cmap('viridis')
#assert(np.max(np.abs(eigvals(F)))<=1.)

#%

#settings

tmax=600 #Time in s



L=500e-6 #Length in m
nx=300   #number of division of the x axis
CsaltIN= 200  #Initial salt concentration in the channel
CsaltOUT=.02 #Initial salt concentration out of the channel

#Diffusion and diffusiophoresis coefficients
Dprot=5.9e-11
Dsalt=2*1e-9
Ddiffph=-3e-11


nt=10    #Number of times at which to plot
name='res_%.1fin_%.1foutNaCl'%(CsaltIN,CsaltOUT)
reservoir=False  # Dead end or reservoir






dx = L/nx
Cx = ds.getCx(nx, dx, reservoir)
x=(np.arange(nx)*dx)[:,None]
#get t for plotting
t=np.exp(np.linspace(0,np.log(tmax),nt))

simulation = ds.diffusioSim(L, nx, CsaltIN, CsaltOUT, Dprot, 
                            Dsalt, Ddiffph, reservoir)
             
  
#Start loop                          
for i, dt in enumerate(np.diff(t)):
    simulation.advance(dt)
    c=cmap(i/(nt-1))
    plt.figure(1)
    plt.semilogx(x*1e6, np.log(simulation.Csalt), c=c)
    plt.figure(2)
    plt.plot(x[1:]*1e6, (Cx@simulation.Csalt)[1:], c=c)
    plt.figure(3)
    plt.plot(x[1:]*1e6, (Cx@simulation.Csalt/simulation.Csalt)[1:], c=c)
    plt.figure(0)
    plt.semilogx(x*1e6, simulation.Cprot, c=c)
    
        
plt.figure(0)
plt.xlabel(r'Position [$\mu$m]') 
plt.xlim([0,500]) 
plt.savefig('prot{:d}.pdf'.format(nx))
plt.figure(1)
plt.xlim([0,500])
plt.savefig('salt{:d}.pdf'.format(nx))
plt.figure(2)
plt.xlim([0,500])
plt.savefig('dsalt{:d}.pdf'.format(nx))
plt.figure(3)
plt.xlim([0,500])
plt.savefig('dlnsalt{:d}.pdf'.format(nx))


# plt.figure(1)
# plt.xlabel(r'Position [$\mu$m]')  
# plt.savefig(name + 'Cs.pdf')
# plt.figure(2)
# plt.xlabel(r'Position [$\mu$m]')  
# plt.savefig(name + 'GCs.pdf')
# plt.figure(3)
# plt.xlabel(r'Position [$\mu$m]')   
# plt.savefig(name + 'GlnCs.pdf') 
