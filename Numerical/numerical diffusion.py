# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 15:09:43 2017

@author: quentinpeter
"""
import numpy as np
from matplotlib.pyplot import figure, plot
import diffusiosim_diff as ds
# import diffusiosim as ds
import matplotlib.pyplot as plt
import matplotlib
import time
cmap = matplotlib.cm.get_cmap('viridis')
# assert(np.max(np.abs(eigvals(F)))<=1.)

#%%

# settings

tmax = 600  # Time in s
L = 500e-6  # Length in m

CsaltIN = 100 # Initial salt concentration in the channel
CsaltOUT = 1  # Initial salt concentration out of the channel

# Diffusion and diffusiophoresis coefficients
Dprot = 5.9e-11#2e-11#
Dsalt = 1e-9
Ddiffph = -1.5e-10


nt = 10  # Number of times at which to plot
name = 'res_%.1fin_%.1foutNaCl' % (CsaltIN, CsaltOUT)
reservoir = False  # Dead end or reservoir


# get t for plotting
times_plot = np.exp(np.linspace(0, np.log(tmax), nt))
NX = [100]#np.array([50, 100, 200])
posmax = np.zeros((len(NX), len(times_plot)))
maxval = np.zeros((len(NX), len(times_plot)))
timeit = np.zeros(len(NX))



for j, nx in enumerate(NX):
    nx = int(nx)

    simulation = ds.diffusiophoresisIntegrator(
                        L, nx, CsaltIN, CsaltOUT, Dprot,
                        Dsalt, Ddiffph, reservoir, 1)
    figure()
    for i, t in enumerate(times_plot): 
        plt.plot(np.linspace(0, L, nx)*1e6, simulation.dxlnq(t),
                 c=cmap(i / len(times_plot)))
    plt.xlabel('X [$\mu$m]')
    plt.ylabel('Intensity')
    plt.savefig('sgmout/dlnq_{}.pdf'.format(nx))
        #%%
    X = simulation.X*1e6
    t0 = time.clock()
    plt.figure()
    # Start loop
    results = np.zeros((nt, len(simulation.Cprot)))
    for i, dt in enumerate([times_plot[0], *np.diff(times_plot)]):
        simulation.advance(dt)
        results[i] = simulation.Cprot
        maxval[j, i] = np.max(simulation.Cprot)
        posmax[j, i] = X[np.argmax(simulation.Cprot)]
        
    for i, prof in enumerate(results):
        plt.plot(X[X < 500], prof[X < 500], c=cmap(i / len(times_plot)))
    plt.xlabel('X [$\mu$m]')
    plt.ylabel('Intensity')
    plt.savefig('sgmout/sim_{}.pdf'.format(nx))
    timeit[j] = time.clock() - t0
#%%
#figure()
#plot(np.diff(results, axis=0).T)
#figure()
#plot(np.diff(results, axis=1).T)
#%%
#plt.figure()
#for i, ay in enumerate(maxval):
#    plt.plot(t, ay, 'x-', c=cmap(i / (len(NX)-1)))
#    
#plt.xlabel('Time [s]')
#plt.ylabel('Maximum')
#plt.legend(NX)
#plt.savefig('sgmout/Maximum.pdf')

#plt.figure()
#for i, ay in enumerate(posmax):
#    plt.plot(t, ay, 'x-', c=cmap(i / (len(NX)-1)))
#plt.xlabel('Time [s]')
#plt.ylabel('Maximum Position [$\mu$m]')
#plt.legend(NX)
#plt.savefig('sgmout/MaximumPos.pdf')
#
#c = np.polyfit(np.log(NX), np.log(timeit), 1)
#plt.figure()
#plt.semilogy(L*1e6/NX, timeit, 'x')
##plt.plot(NX, np.exp(np.poly1d(c)(np.log(NX))))
##plt.plot(L*1e6/NX, np.exp(c[1]-2)*NX**4)
#plt.xlabel('Spatial step [$\mu$m]')
#plt.ylabel('Computation Time [s]')
#plt.legend(['Data', 'X^4'])
#plt.savefig('sgmout/time.pdf')