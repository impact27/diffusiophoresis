#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 16:02:17 2019

@author: quentinpeter

This script is distributed under CC BY-NC-SA version 4.0. You should have
recieved a copy of the licence along with it. If not, see
https://creativecommons.org/licenses/by-nc-sa/4.0/.
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from matplotlib.pyplot import figure, imshow
import os.path
import json
from scipy.interpolate import interp1d

from diffusiophoresis_fitting import (color,
                                      get_similarity)


Dp = 5.9e-11
Ds = 1e-9
Gamma = 1.5e-10

Pout = 0.01

Cout = 1
Cin = 200

beta = Cout / Cin
init = [beta, Gamma/Ds, Dp/Ds]

eta = 10 ** np.linspace(-2, 1, 1000)
eta[0] = 0


res = get_similarity(eta, *init)


similarity = interp1d(res.x, res.y[0] * Pout)

cmap = mpl.cm.get_cmap('plasma')
norm = LogNorm(vmin=.1, vmax=10)

sets_file = '../Data/Simulations_Processed/sets_list.json'
with open(sets_file, 'r') as f:
    sets = json.load(f)

set_name = 'Dim'
axes = [np.load(os.path.join(
        os.path.dirname(sets_file), fn)) for fn in sets[set_name]['axes']]
times = axes[0]['t'] * 60

C_p = [np.load(os.path.join(
       os.path.dirname(sets_file), fn)) for fn in sets[set_name]['C_p']]

idx = 20
figure(figsize=(6, 1.8))
ax = plt.subplot2grid((3, 1), (0, 0))
ax.tick_params(axis='x', labelbottom=False)
ax.tick_params(axis='y', labelleft=False)
plt.ylabel('1D')
imshow(C_p[0][idx][np.newaxis] * np.ones(51)[:, np.newaxis])
ax = plt.subplot2grid((3, 1), (1, 0))
ax.tick_params(axis='x', labelbottom=False)
ax.tick_params(axis='y', labelleft=False)
imshow(C_p[1][idx].T)
plt.ylabel('2D')
ax = plt.subplot2grid((3, 1), (2, 0))
ax.tick_params(axis='y', labelleft=False)
imshow(C_p[2][idx].T)
plt.ylabel('3D')
plt.xlabel('Position [um]')

plt.savefig('../Plots/spatial_distribution.pdf')


#%%
for idx in range(len(C_p)):
    if C_p[idx].ndim == 3:
        C_p[idx] = np.mean(C_p[idx], axis=-1)
C_p = np.array(C_p)

plt.figure(figsize=(12, 12))

colspan = 12
rowspan = 2
grid_size = (3, colspan * 3 + 1)
axes = []
for idx in range(3):
    ax = plt.subplot2grid(grid_size, (0, colspan * idx),
                          colspan=colspan, rowspan=rowspan)
    if idx > 0:
        ax.tick_params(axis='y', labelleft=False)
    ax.tick_params(axis='x', labelbottom=False)
    axes.append(ax)

for idx in range(3):
    ax = plt.subplot2grid(grid_size, (2, colspan * idx),
                          colspan=colspan)
    if idx > 0:
        ax.tick_params(axis='y', labelleft=False)
    axes.append(ax)

log_indices = np.asarray([0, *np.unique(np.ceil(np.logspace(0, 2, 6)))], int)
for idx in log_indices:
    time = times[idx]
    for i in range(3):
        axes[i].plot(C_p[i, idx], c=color(time))
    ylim = axes[0].get_ylim()
    axes[0].set_title('1D')
    axes[0].set_ylabel('Protein concentration [M]')
    axes[1].set_ylim(ylim)
    axes[1].set_title('2D')
    axes[2].set_ylim(ylim)
    axes[2].set_title('3D')

    for i in range(3, 6):
        axes[i].plot(C_p[i - 3, idx] / np.max(C_p[i - 3, idx]), c=color(time))
        axes[i].set_ylim([-0.1, 1.1])
    axes[3].set_ylabel('Normalised Protein concentration')

idx_mid = log_indices[len(log_indices) // 2 + 1]
time = times[idx_mid]
positions = np.arange(501) * 1e-6
eta = positions / np.sqrt(4 * Ds * time)
expected = similarity(eta)
norm_expected = expected / np.max(expected)

for i in range(6):
    ax = axes[i]
    ax.plot(0, 0, '-', c='gray', label='Simulation')
    if i < 3:
        y = expected
    else:
        y = norm_expected
    ax.plot(y, 'k-.', label=f'Solution at {time/60:.1f}min')

    ax.legend()

for i in range(3, 6):
    axes[i].set_xlabel('Position [um]')




ax = plt.subplot2grid(grid_size, (1, colspan * 3))
cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm)
cb1.set_label('Time [min]')
plt.tight_layout(pad=0.001, h_pad=0.001, w_pad=0.001)

ax = plt.subplot2grid(grid_size, (rowspan, colspan * 3))
cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm)
cb1.set_label('Time [min]')
plt.tight_layout(pad=0.001, h_pad=0.001, w_pad=0.001)

plt.savefig('../Plots/intensity_dimentions.pdf')