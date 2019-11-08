# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 10:19:07 2017

@author: rapha
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
cmap = matplotlib.cm.get_cmap('inferno')
# 3128
L = 3128e-6
L = 500e-6
nx = 100
nN = 1000
tmax = 600
nt = 15
x = (np.arange(nx) * L / nx)[:, None]
D = 1.99 * 1e-9
D = 1e-9
# D=5.9e-11
tlog = np.linspace(0, np.log(tmax), nt)
t = np.exp(tlog)
# t=times*60
t = t[None, :]

test = np.empty((x.size, nN), dtype=float)


def p(N):
    return (np.pi / 2 + N * np.pi) / L


def Lambda(N):
    return np.sqrt(D) * p(N)


def C(x, t):
    ret = np.zeros((x.size, t.size), dtype=float)
    for N in range(nN):
        ret += (2 / np.pi) * (1 / (N + 0.5)) * \
            np.exp(-Lambda(N)**2 * t) * np.sin(p(N) * x)
        test[:, N] = ret[:, 0]
    return ret


Conc = 1 - C(x, t)

plt.figure()
norm = LogNorm(vmin=t.min(), vmax=t.max())
colors = plt.imshow(t, cmap=cmap, norm=norm)
plt.clf()
Tcol = np.log(t)
Colors = cmap((Tcol - Tcol.min()) / (Tcol.max() - Tcol.min()))[0]
for Y, col in zip(1 - np.transpose(Conc), Colors):
    plt.plot(x * 1e6, Y, c=col)
plt.colorbar(colors).set_label(label='time [min]', size=18)
plt.xlabel(r'distance [$\mu$m]', fontsize=18)
plt.ylabel(r'Intensity', fontsize=18)
# plt.xlim(-200,600)
plt.savefig('Theorical.pdf', bbox_inches='tight')

#%%
from matplotlib.pyplot import figure, plot, semilogx
# A=np.sum(C(x,t),0)
figure()
# semilogx(np.ravel(t),A)
semilogx(np.ravel(t), np.sum(1 - C(x, t), 0))
plt.xlabel('time [s]', fontsize=18)
plt.ylabel(r'Salt in Channel [%]', fontsize=18)
plt.legend((r' 500$\mu$m', r'2500$\mu$m'))
plt.savefig('IntNaCl.pdf', bbox_inches='tight')

#%%
RMS = np.sqrt(np.mean((test - test[:, -1:])**2, axis=0))
plt.figure()
plt.semilogy(RMS, 'x-')
