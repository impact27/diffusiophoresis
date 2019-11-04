#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 18:23:43 2019

@author: quentinpeter

Fits the comsol simulations
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
import os.path
import json

from diffusiophoresis_fitting import (normalise_profile,
                                      fit_and_plot)

# %% Define sets of data
figures_folder = '../Plots/simulations_fits'
sets_file = '../Data/Simulations_Processed/sets_list.json'
parameters_file = '../Data/Simulations_Processed/simulation_parameters.json'

# Set names to fit
set_names = ['Dim', 'CsOut', 'Gamma', 'Ds', 'Dp']
set_names = ['Dp']

# %% Loop over all the sets
with open(sets_file, 'r') as f:
    sets = json.load(f)

for set_name in set_names:
    print(set_name)
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
        numbers = numbers / Ds
    elif set_name == 'Dp':
        Dp = numbers
        numbers = numbers / Ds
    elif set_name == 'Ds':
        Ds = numbers
    elif set_name == 'CsOut':
        Cs_out = numbers

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

        # Get normalised profiles and select regions to fit
        norm_profiles, mask_times, idx_max = normalise_profile(profiles)
        end_intensity = (np.mean(profiles[:, 400:], axis=1)
                         / np.mean(profiles[:, 0]))
        mask_times = np.logical_and(mask_times, end_intensity < 0.2)
        mask_valid = times > 0
        beta_salt = Cs_outi / Cs_in

        # Fit :)
        fit_Dp, fit_Gp = fit_and_plot(
            norm_profiles[mask_valid], times[mask_valid], positions,
            idx_max[mask_valid], Cs_outi / Cs_in, Dsi, mask_times[mask_valid],
            plot_freq=10,
            expected_Dp=Dpi,
            expected_Gp=Gammai)

        # Set fit title and saves
        plt.title(names[idx])
        plt.savefig(os.path.join(figures_folder,
                                 f'{set_name}_{names[idx]}.pdf'))

        # Save results
        results.at[numbers[idx], "Simulation diffusiophoresis"] = Gammai
        results.at[numbers[idx], "Fit diffusiophoresis"] = fit_Gp
        results.at[numbers[idx], "Simulation diffusion"] = Dpi
        results.at[numbers[idx], "Fit diffusion"] = fit_Dp

    # Print results for sweep
    figure()
    plt.loglog(results.loc[:, "Simulation diffusiophoresis"], '.',
               label='Expected')
    plt.loglog(results.loc[:, "Fit diffusiophoresis"], '.', label='Fit')
    plt.xlabel(set_name)
    plt.legend()
    plt.title(f'Diffusiophoresis')
    plt.savefig(os.path.join(figures_folder,
                             f'{set_name}_diffusiophoresis.pdf'))

    figure()
    plt.loglog(results.loc[:, "Simulation diffusion"], '.', label='Expected')
    plt.loglog(results.loc[:, "Fit diffusion"], '.', label='Fit')
    plt.xlabel(set_name)
    plt.legend()
    plt.title(f'Diffusion')
    plt.savefig(os.path.join(figures_folder, f'{set_name}_diffusion.pdf'))

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
    plt.savefig(os.path.join(figures_folder, f'{set_name}_error_ratios.pdf'))

    diffusion_error = np.sqrt(np.mean(np.square(
        results.loc[:, "Fit diffusion"] /
        results.loc[:, "Simulation diffusion"] - 1)))

    diffusiophoresis_error = np.sqrt(np.mean(np.square(
        results.loc[:, "Fit diffusiophoresis"] /
        results.loc[:, "Simulation diffusiophoresis"] - 1)))
    print(f"{set_name}: diffusion {diffusion_error * 100:.1f}%,"
          f" diffusiophoresis {diffusiophoresis_error * 100:.1f}%")

    plt.show()
