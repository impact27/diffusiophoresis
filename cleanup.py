#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 18:23:43 2019

@author: quentinpeter
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, imshow
import pandas as pd
from matplotlib.pyplot import figure, plot, imshow, show

from diffusiophoresis_fitting import (get_similarity,
                                      get_Gamma,
                                      normalise_profile,
                                      fit_and_plot)

from matplotlib.cm import plasma

# %%
sets = []
names = ['1d', '2d', '3d']
sets.append(dict(
    set_name = 'Dim',
    numbers = [1, 2, 3],
    names = names,
    axes = [np.load(f'k-device_gamma-1p5e-10_{nm}_axes.npz') for nm in names],
    C_p = [np.load(f'k-device_gamma-1p5e-10_{nm}_C_p{nm[0]}.npy') for nm in names],
    xlog = False))

names = ['0p1uM', '1uM', '10uM', '100uM', '1mM', '10mM']
sets.append(dict(
    set_name = 'CsOut',
    names = names,
    numbers = [0.1e-6, 1e-6, 10e-6, 100e-6, 1e-3, 10e-3],
    axes = [np.load(f'k-device_gamma-1p5e-10_CsOut_{nm}_axes.npz') for nm in names],
    C_p = [np.load(f'k-device_gamma-1p5e-10_CsOut_{nm}_C_p2.npy') for nm in names],
    xlog = True))

names = np.fromstring('-2e-11 -5e-11 -1e-10 -2e-10 -5e-10', sep=' ')
sets.append(dict(
    set_name = 'Gamma',
    numbers = [2e-11, 5e-11, 1e-10, 2e-10, 5e-10],
    names = names,
    axes = [np.load(f'k-device_gamma{nm}_axes.npz') for nm in names],
    C_p = [np.load(f'k-device_gamma{nm}_C_p2.npy') for nm in names],
    xlog = True))

names = ['1e-10', '2e-10', '5e-10', '1e-09', '2e-09', '5e-09', '1e-08']
sets.append(dict(
    set_name = 'Ds',
    names = names,
    numbers = [1.e-10, 2.e-10, 5.e-10, 1.e-09, 2.e-09, 5.e-09, 1.e-08],
    axes = [np.load(f'k-device_Ds_{nm}_axes.npz') for nm in names],
    C_p = [np.load(f'k-device_Ds_{nm}_C_p2.npy') for nm in names],
    xlog = True))

names = '5e-12 1e-11 2e-11 5e-11 1e-10 2e-10 5e-10'.split(' ')
sets.append(dict(
    set_name = 'Dp',
    names = names,
    numbers = np.asarray([5.e-12, 1.e-11, 2.e-11, 5.e-11, 1.e-10, 2.e-10, 5.e-10]),
    axes = [np.load(f'k-device_Dp_{nm}_axes.npz') for nm in names],
    C_p = [np.load(f'k-device_Dp_{nm}_C_p2.npy') for nm in names],
    xlog = True))

# names = ['C', 'logC']
# sets.append(dict(
#     set_name = 'LogC',
#     numbers = [1, 2],
#     names = names,
#     axes = [np.load(f'k-device_gamma-1.5e-10_{nm}_axes.npz') for nm in names],
#     C_p = [np.load(f'k-device_gamma-1.5e-10_{nm}_C_p2.npy') for nm in names],
#     xlog = False))

# names = ['10um', '20um', '50um', '100um']
# sets.append(dict(
#     set_name = 'Ramp',
#     names = names,
#     numbers = [10, 20, 50, 100],
#     axes = [np.load(f'k-device_gamma-1.5e-10_ramp_{nm}_axes.npz') for nm in names],
#     C_p = [np.load(f'k-device_gamma-1.5e-10_ramp_{nm}_C_p2.npy') for nm in names],
#     xlog = False))
# %%
for aset in sets:

    ch_width = 50e-6
    Dp = 5.9e-11
    Ds = 1e-9
    Gamma = 1.5e-10
    Cs_in = 2e-1
    Cs_out = 1e-7

    set_name, names, numbers, axes, C_p, xlog = [
        aset[key] for key in ('set_name', 'names', 'numbers',
                              'axes', 'C_p', 'xlog')]
    print(set_name)
    numbers = np.asarray(numbers)
    if set_name == 'Gamma':
        Gamma = numbers
        numbers = numbers / Ds
    elif set_name == 'Dp':
        Dp = numbers
        numbers = numbers / Ds
    elif set_name == 'Ds':
        Ds = numbers
    elif set_name == 'CsOut':
        Cs_out = numbers

    for i in range(len(C_p)):
        if C_p[i].ndim == 3:
            C_p[i] = np.mean(C_p[i], axis=-1)
    C_p = np.array(C_p)

    results = pd.DataFrame(index=numbers)
    for i in range(len(names)):

        Dsi = Ds
        if set_name == 'Ds':
            Dsi = Ds[i]
        Gammai = Gamma
        if set_name == 'Gamma':
            Gammai = Gamma[i]
        Dpi = Dp
        if set_name == 'Dp':
            Dpi = Dp[i]
        Cs_outi = Cs_out
        if set_name == 'CsOut':
            Cs_outi = Cs_out[i]

        # Get time and position
        profiles = C_p[i]
        times = axes[i]['t'] * 60
        positions = np.arange(np.shape(profiles)[1]) * 1e-6

        # fit D
        norm_profiles, mask_times, idx_max = normalise_profile(profiles)
        end_intensity = np.mean(profiles[:, 400:], axis=1) / np.mean(profiles[:,0])
        mask_times = np.logical_and(mask_times, end_intensity < 0.2)
        mask_valid = times > 0

        beta_salt = Cs_outi / Cs_in

        fit_Dp, fit_Gp = fit_and_plot(
            norm_profiles[mask_valid], times[mask_valid], positions,
            idx_max[mask_valid], Cs_outi / Cs_in, Dsi, mask_times[mask_valid],
            plot_freq=10,
            expected_Dp=Dpi,
            expected_Gp=Gammai)

        plt.legend()
        plt.title(names[i])
        plt.savefig(f'figs/{set_name}_{names[i]}.pdf')

        results.at[numbers[i], "Simulation diffusiophoresis"] = Gammai
        results.at[numbers[i], "Fit diffusiophoresis"] = fit_Gp
        results.at[numbers[i], "Simulation diffusion"] = Dpi
        results.at[numbers[i], "Fit diffusion"] = fit_Dp


    #%%
    figure()
    plt.loglog(results.loc[:, "Simulation diffusiophoresis"], '.', label='Expected')
    plt.loglog(results.loc[:, "Fit diffusiophoresis"], '.', label='Fit')
    plt.xlabel(set_name)
    plt.legend()
    plt.title(f'Diffusiophoresis')
    plt.savefig(f'figs/{set_name}_diffusiophoresis.pdf')

    figure()
    plt.loglog(results.loc[:, "Simulation diffusion"], '.', label='Expected')
    plt.loglog(results.loc[:, "Fit diffusion"], '.', label='Fit')
    plt.xlabel(set_name)
    plt.legend()
    plt.title(f'Diffusion')
    plt.savefig(f'figs/{set_name}_diffusion.pdf')

    figure()
    plt.loglog(results.loc[:, "Fit diffusiophoresis"] /
               results.loc[:, "Simulation diffusiophoresis"],
               '.', label='Diffusiophoresis')
    plt.loglog(results.loc[:, "Fit diffusion"] /
               results.loc[:, "Simulation diffusion"],
               '.', label='Diffusion')
    plt.xlabel(set_name)
    plt.legend()
    plt.title(f'ratios')
    plt.savefig(f'figs/{set_name}_error_ratios.pdf')

    diffusion_error = np.sqrt(np.mean(np.square(
        results.loc[:, "Fit diffusion"] /
        results.loc[:, "Simulation diffusion"] - 1)))

    diffusiophoresis_error = np.sqrt(np.mean(np.square(
        results.loc[:, "Fit diffusiophoresis"] /
        results.loc[:, "Simulation diffusiophoresis"] - 1)))
    print(f"{set_name}: diffusion {diffusion_error * 100:.1f}%,"
          f" diffusiophoresis {diffusiophoresis_error * 100:.1f}%")

    plt.show()
