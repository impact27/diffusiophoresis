#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 17:15:50 2019

@author: quentinpeter
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from matplotlib.pyplot import figure
import os.path
import json
from scipy.interpolate import interp1d

parameter_dict = {
    'CsOut': 'salt_concentartion_out',
    'Gamma': 'diffusiophoresis_coefficient',
    'Ds': 'salt_diffusion',
    'Dp': 'protein_diffusion'}

cmap = mpl.cm.get_cmap('plasma')
norm = LogNorm(vmin=.1, vmax=10)
parameters_file = '../Data/Simulations_Processed/simulation_parameters.json'

sets_file = '../Data/Simulations_Processed/sets_list.json'
with open(sets_file, 'r') as f:
    sets = json.load(f)

set_names = ['CsOut', 'Gamma', 'Dp', 'Ds']
xlabel = {
    'Dim': 'Dimension',
    'CsOut': 'Salt ratio',
    'Gamma': 'Diffusiophoresis coefficient [$m^2/s$]',
    'Ds': 'Salt diffusion coefficient [$m^2/s$]',
    'Dp': 'Protein Diffusion coefficient [$m^2/s$]'}
with open(parameters_file, 'r') as f:
    parameters = json.load(f)
max_intensities = {}

for set_name in set_names:
    axes = [np.load(os.path.join(
            os.path.dirname(sets_file), fn)) for fn in sets[set_name]['axes']]
    times = axes[0]['t'] * 60
    C_p = [np.load(os.path.join(
           os.path.dirname(sets_file), fn)) for fn in sets[set_name]['C_p']]
    for idx in range(len(C_p)):
        if C_p[idx].ndim == 3:
            C_p[idx] = np.mean(C_p[idx], axis=-1)
    C_p = np.array(C_p)
    max_intensities[set_name] = np.max(C_p, axis=(1, 2))

default_value = max_intensities['CsOut'][0]

#%%
figure(figsize=(12, 8))
for idx, key in enumerate(set_names):
    plt.subplot2grid((2, 2), (idx%2, idx // 2))
    x = np.asarray(sets[key]['numbers'])
    if key == 'CsOut':
        x /= parameters['salt_concentartion_in']
    plt.loglog(x, max_intensities[key] * 1e2, 'x--', label='Simulation')
    plt.xlabel(xlabel[key])
    plt.ylabel('Intensity')
    if key in parameter_dict:
        regular_x = parameters[parameter_dict[key]]
        if key == 'CsOut':
            regular_x /= parameters['salt_concentartion_in']
        plt.plot(np.abs(regular_x), default_value * 1e2, 'x', label='Reference')

    else:
        plt.plot(2, default_value * 1e2, 'x')

    plt.ylim([1, 2e3])
    plt.legend()
plt.tight_layout()
plt.savefig('../Plots/intensity_parameters.pdf')