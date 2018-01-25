# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 10:12:16 2017

@author: quentinpeter
"""
import numpy as np


L = 500e-6
D = 2 * 1e-9
nN = 100

X = np.linspace(0, L, 1000)


def p(N):
    return (np.pi / 2 + N * np.pi) / L


def Lambda(N):
    return np.sqrt(D) * p(N)


def C(x, t):
    ret = np.zeros((x.size), dtype=float)
    for N in range(nN):
        ret += (2 / np.pi) * (1 / (N + 0.5)) * \
            np.exp(-Lambda(N)**2 * t) * np.sin(p(N) * x)
    return ret


def dC(x, t):
    ret = np.zeros((x.size), dtype=float)
    for N in range(nN):
        ret += (2 / L) * np.exp(-Lambda(N)**2 * t) * np.cos(p(N) * x)
    return ret


#%%
import diffusiosim as ds
X = np.linspace(0, L, 100)
dx = X[1] - X[0]
q = ds.getQs(len(X))

Cx = 1 / np.max(dx) / 12 * (-q[2] + 8 * q[1] - 8 * q[-1] + q[-2])
#            Cx = 1/np.max(dx)/4*(q[1]+3*q[0]-5*q[-1]+q[-2])
Cx = 1 / np.max(dx) / 2 * (q[1] - q[-1])
Cx = 1 / np.max(dx) * (q[0] - q[-1])
Cx = 1 / np.max(dx) * (q[1] - q[0])

import matplotlib.pyplot as plt
plt.figure()
plt.plot(X, dC(X, 1) / (CsaltOUT / (CsaltIN - CsaltOUT) + C(X, 1)), 'x')
X = np.linspace(0, L, 1000)
dx = X[1] - X[0]
plt.plot(X, dC(X, 1) / (CsaltOUT / (CsaltIN - CsaltOUT) + C(X, 1)))
X = np.linspace(0, L, 500)
dx = X[1] - X[0]
plt.plot(X[1:] - dx / 2, np.diff(np.log(CsaltOUT +
                                        (CsaltIN - CsaltOUT) * C(X, 1))) / dx, '.')
