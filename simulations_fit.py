#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 18:23:43 2019

@author: quentinpeter

Fits the comsol simulations

This script is distributed under CC BY-NC-SA version 4.0. You should have
recieved a copy of the licence along with it. If not, see
https://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from matplotlib.pyplot import figure
import pandas as pd
import os.path
import json

from diffusiophoresis_fitting import (normalise_profile,
                                      fit_diffusiophoresis,
                                      plot_diffusiophoresis,
                                      color)

# %% Define sets of data
figures_folder = '../Plots/simulations_fits'
sets_file = '../Data/Simulations_Processed/sets_list.json'
parameters_file = '../Data/Simulations_Processed/simulation_parameters.json'
plot_freq = 10

# Set names to fit
set_names = ['Dim', 'CsOut', 'Gamma', 'Dp', 'Ds']
set_names = ['CsOut', 'Gamma', 'Dp', 'Ds']

xlabel = {
    'Dim': 'Dimension',
    'CsOut': 'Salt ratio',
    'Gamma': 'Diffusiophoresis coefficient [$m^2/s$]',
    'Ds': 'Salt diffusion coefficient [$m^2/s$]',
    'Dp': 'Protein Diffusion coefficient [$m^2/s$]'}

# %% Loop over all the sets
cmap = mpl.cm.get_cmap('plasma')
norm = LogNorm(vmin=.1, vmax=10)


def get_31_axes():
    """Get axes for 3_1 plot."""

    fig = plt.figure(figsize=(16*.8, 10*.8))
    yield fig

    rowspan = 8
    colspan = 12
    grid_size = ((5 * rowspan) // 2, colspan * 3 + 1)

    for idx in range(3):
        ax = plt.subplot2grid(grid_size, (0, colspan * idx),
                              colspan=colspan, rowspan=rowspan-1, fig=fig)
        yield ax
        if idx > 0:
            ax.tick_params(axis='y', labelleft=False)
            ax.set_ylabel('')
        if idx < 2:
            ax.set_xticks(ax.get_xticks()[:-1])

    ax = plt.subplot2grid(grid_size, (0, 3*colspan),
                          colspan=1, rowspan=rowspan-1, fig=fig)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm)
    cb1.set_label('Time [min]')

    ax = plt.subplot2grid(grid_size,
                          (rowspan + 1, colspan//2),
                          colspan=2 * colspan,
                          rowspan=(3 * rowspan) // 2 + 2,
                          fig=fig)
    yield ax

# %% Loop over all the sets
with open(sets_file, 'r') as f:
    sets = json.load(f)
Gmin = Gmax = Dmin = Dmax = 1
all_results = {}

fig_ex = figure(figsize=(12, 8))
fig = figure(figsize=(12, 8))
for set_idx, set_name in enumerate(set_names):
    print(set_name)
#    axes_generator = get_31_axes()
#    fig_composite = next(axes_generator)
    with open(parameters_file, 'r') as f:
        parameters = json.load(f)

    # Load the data
    Dp = parameters['protein_diffusion']
    Ds = parameters['salt_diffusion']
    Gamma = np.abs(parameters['diffusiophoresis_coefficient'])
    Cs_in = parameters['salt_concentartion_in']
    Cs_out = parameters['salt_concentartion_out']

    names = sets[set_name]['names']
    numbers = sets[set_name]['numbers']
    xlog = sets[set_name]['xlog']
    axes = [np.load(os.path.join(
        os.path.dirname(sets_file), fn)) for fn in sets[set_name]['axes']]
    C_p = [np.load(os.path.join(
        os.path.dirname(sets_file), fn)) for fn in sets[set_name]['C_p']]

    # Replace parameters by whatever we are sweeping over
    numbers = np.asarray(numbers)
    if set_name == 'Gamma':
        Gamma = numbers
    elif set_name == 'Dp':
        Dp = numbers
    elif set_name == 'Ds':
        Ds = numbers
    elif set_name == 'CsOut':
        Cs_out = numbers
        numbers = numbers / Cs_in

    # Take the mean over the width if we are not in 1D
    for idx in range(len(C_p)):
        if C_p[idx].ndim == 3:
            C_p[idx] = np.mean(C_p[idx], axis=-1)
    C_p = np.array(C_p)

    # Fit each iteration on the sweep
    results = pd.DataFrame(index=numbers)
    for idx in range(len(names)):
        # Get the corect parameter for index idx
        Dsi = Ds
        if set_name == 'Ds':
            Dsi = Ds[idx]
        Gammai = Gamma
        if set_name == 'Gamma':
            Gammai = Gamma[idx]
        Dpi = Dp
        if set_name == 'Dp':
            Dpi = Dp[idx]
        Cs_outi = Cs_out
        if set_name == 'CsOut':
            Cs_outi = Cs_out[idx]

        # Get time and position with correct units (seconds and meters)
        profiles = C_p[idx]
        times = axes[idx]['t'] * 60
        positions = np.arange(np.shape(profiles)[1]) * 1e-6
        results.at[numbers[idx], "Max intensity"] = np.max(profiles)

        # Get normalised profiles and select regions to fit
        norm_profiles, mask_times, idx_max = normalise_profile(profiles)
        end_intensity = (np.mean(profiles[:, 400:], axis=1)
                         / np.mean(profiles[:, 0]))
        mask_times = np.logical_and(mask_times, end_intensity < 0.2)
        mask_valid = times > 0
        beta_salt = Cs_outi / Cs_in

        # Fit :)
        fit_Dp, fit_Gp = fit_diffusiophoresis(
            norm_profiles[mask_valid], times[mask_valid], positions,
            idx_max[mask_valid], Cs_outi / Cs_in, Dsi, mask_times[mask_valid])

        # Plot
        plot_diffusiophoresis(
            norm_profiles[mask_valid], times[mask_valid], positions,
            idx_max[mask_valid], Cs_outi / Cs_in, Dsi, mask_times[mask_valid],
            fit_Dp, fit_Gp,
            plot_freq=plot_freq,
            expected_Dp=Dpi,
            expected_Gp=Gammai)

        # Set fit title and saves
        plt.title(names[idx])
        plt.savefig(os.path.join(figures_folder,
                                 f'{set_name}_{names[idx]}.pdf'))

        title_dict = {
            'CsOut': f'Ratio {numbers[idx]:.2f}',
            'Gamma': f'Diffusiophoresis = {numbers[idx]} $m^2/s$',
            'Ds': f'Salt diffusion = {numbers[idx]} $m^2/s$',
            'Dp': f'Protein diffusion = {numbers[idx]} $m^2/s$'}

        if (set_name == 'CsOut' and idx == len(names) - 1) or idx == 0:
            ax = plt.subplot2grid((2, 2), (set_idx%2, set_idx // 2),
                                  fig=fig_ex)
            # Plot
            plot_diffusiophoresis(
                norm_profiles[mask_valid], times[mask_valid], positions,
                idx_max[mask_valid], Cs_outi / Cs_in,
                Dsi, mask_times[mask_valid],
                fit_Dp, fit_Gp,
                plot_freq=plot_freq,
                expected_Dp=Dpi,
                expected_Gp=Gammai,
                ax=ax)
            ax.set_title(title_dict[set_name])

        # Save results
        results.at[numbers[idx], "Simulation diffusiophoresis"] = Gammai
        results.at[numbers[idx], "Fit diffusiophoresis"] = fit_Gp
        results.at[numbers[idx], "Simulation diffusion"] = Dpi
        results.at[numbers[idx], "Fit diffusion"] = fit_Dp

    diffusiophoresis_ratio = (
        results.loc[:, "Fit diffusiophoresis"] /
        results.loc[:, "Simulation diffusiophoresis"])
    diffusion_ratio = (
        results.loc[:, "Fit diffusion"] /
        results.loc[:, "Simulation diffusion"])
    #%%
#    ax = next(axes_generator)
    ax = plt.subplot2grid((2, 2), (set_idx%2, set_idx // 2), fig=fig)
    ax.plot(diffusion_ratio.index, np.ones(len(diffusion_ratio)) - 1, 'k-.', label='Expected')
    ax.plot(100 * (diffusiophoresis_ratio - 1), 'x--', label='Diffusiophoresis')
    ax.plot(100 * (diffusion_ratio - 1), 'x--', label='Diffusion')

    ax.set_xlabel(xlabel[set_name])
    # plt.title(set_name)
    # plt.yscale('log')
    ylim = [-41, 41]
    ax.set_ylabel('Error [%]')
    ax.set_ylim(ylim)
    if set_name == 'Dim':
        ax.set_xticks([1, 2, 3])
    if xlog:
        ax.set_xscale('log')

    parameter_dict = {
        'CsOut': 'salt_concentartion_out',
        'Gamma': 'diffusiophoresis_coefficient',
        'Ds': 'salt_diffusion',
        'Dp': 'protein_diffusion'}


    if set_name in parameter_dict:
        regular_x = parameters[parameter_dict[set_name]]
        if set_name == 'CsOut':
            regular_x /= parameters['salt_concentartion_in']
        ax.plot(np.ones(2) * np.abs(regular_x), ylim, '--', label='Reference')
    ax.legend()
#    plt.show()

    diffusion_error = np.sqrt(np.mean(np.square(diffusion_ratio - 1)))
    diffusiophoresis_error = np.sqrt(np.mean(np.square(
        diffusiophoresis_ratio - 1)))

    print(f"{set_name}: diffusion {diffusion_error * 100:.1f}%,"
          f" diffusiophoresis {diffusiophoresis_error * 100:.1f}%")

    Gmin = Gmin if Gmin < np.min(diffusiophoresis_ratio) else np.min(diffusiophoresis_ratio)
    Gmax = Gmax if Gmax > np.max(diffusiophoresis_ratio) else np.max(diffusiophoresis_ratio)
    Dmin = Dmin if Dmin < np.min(diffusion_ratio) else np.min(diffusion_ratio)
    Dmax = Dmax if Dmax > np.max(diffusion_ratio) else np.max(diffusion_ratio)
    all_results[set_name] = results

plt.figure(fig.number)
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, f'error_ratios.pdf'))

plt.figure(fig_ex.number)
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, f'extreme_fit.pdf'))
