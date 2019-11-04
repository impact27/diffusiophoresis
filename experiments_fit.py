#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 13:50:39 2018

@author: quentinpeter
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, figure
import matplotlib
import pandas as pd
import re
from scipy.ndimage.filters import maximum_filter
from scipy.interpolate import interp1d
import json

from diffusiophoresis_fitting import fit_and_plot, color


# %% Data
cmap = matplotlib.cm.get_cmap('plasma')
save_dir = '../Plots/experiments_fit'
fits_filename = '../Data/Experiments_processed/fit_sets.json'
salts_infos = json.load(open('../Litterature Values/salts_info.json'))
proteins_infos = json.load(open('../Litterature Values/proteins_info.json'))

# Physical parameters
temperature = 273.14 + 23  # Kelvins
boltzmann = 1.38e-23
viscosity = 1e-3  # Water viscosity
electronic_charge = 1.602e-19  # J / V
gamma_over_mu_beta = boltzmann * temperature / electronic_charge
DRh = temperature * boltzmann / (6 * np.pi * viscosity)


# %% Compute results
sets = json.load(open(fits_filename))
results_list = {}
for set_name in sets:
    fns = sets[set_name]
    results = pd.DataFrame(index=fns)
    results['name'] = [re.findall(r'\/([^\/]+)\.npz', fn)[0] for fn in fns]
    # Extract data from file name
    for idx in results.index:
        pconc, punits, prot, sconc, sunits, salt = re.findall(
            r'o([\d\.]+)(mM|gpl)([^_]+)_i(\d+)(m?M)(.+)',
            results.loc[idx, 'name'])[0]
        pconc = float(pconc)
        sconc = float(sconc)
        if punits == 'M':
            pconc *= 1e3
        if sunits == 'M':
            sconc *= 1e3
        results.loc[idx, 'prot'] = prot.upper()
        results.loc[idx, 'salt'] = salt.upper()
        results.loc[idx, 'prot_conc'] = pconc
        results.loc[idx, 'salt_conc'] = sconc
        date = re.findall(
            r'(\d{8})/', idx)[0]
        results.loc[idx, 'date'] = date

    # Process each file
    for fn in fns:
        print(fn)
        # Load data
        name = results.at[fn, 'name']
        data = np.load(os.path.join(os.path.dirname(fits_filename), fn))
        profiles = data['profiles']
        positions = data['X_pos'] * 1e-6
        times = data['times'][:len(profiles)] * 60
        Ds = salts_infos[results.at[fn, 'salt']]['Diffusion']
        # The salt concentration out needs to be small.
        # The exact value doesn't affect the salt much
        # as long as it is much smaller than the "in" value.
        Cs_out = 0.2  # 200uM
        Cs_in = results.at[fn, 'salt_conc']
        beta_salt = Cs_out / Cs_in

        # Print profiles
        figure()
        for idx, prof in enumerate(profiles):
            plot(positions * 1e6, prof, c=color(times[idx]))
        plt.title(name)
        plt.savefig(os.path.join(save_dir, name + '_profiles.pdf'))

        # Find maximas
        max_idx = np.zeros(len(times), int)
        for idx, prof in enumerate(profiles):
            # We want local maximas with intensity of at least 0.5
            # and position > 0
            local_max_mask = np.logical_and(
                prof == maximum_filter(prof, 51),
                prof > 0.5)
            local_max_mask = np.logical_and(
                local_max_mask,
                positions > 0)
            # Get first local maxima
            max_idx[idx] = np.argmax(local_max_mask)

        # Select profiles to plot
        fit_times = times[max_idx > 0]
        fit_profiles = profiles[max_idx > 0]
        max_idx = max_idx[max_idx > 0]
        # Normalize by the max value
        fit_profiles /= fit_profiles[
            np.arange(len(fit_profiles)), max_idx][:, np.newaxis]

        # Check if profiles has a large intensity between 400 and 500
        value_end = np.zeros_like(fit_times)
        for idx, prof in enumerate(fit_profiles):
            mask = np.logical_and(positions < 500e-6, positions > 400e-6)
            value_end[idx] = np.mean(prof[mask])
        fit_time_mask = value_end < 0.2

        # Remove negative positions
        idx_start = np.argmax(positions > 0)
        fit_positions = positions[idx_start:]
        fit_profiles = fit_profiles[:, idx_start:]
        max_idx -= idx_start

        # Find time offset
        def lse_offset(t0):
            """Look at the least square error for a given time offset."""
            res = 0
            last_eta = None
            last_profile = None
            for idx, prof in enumerate(fit_profiles[fit_time_mask]):
                _eta = fit_positions / np.sqrt(
                    4 * Ds * (fit_times[fit_time_mask][idx] - t0))
                if idx > 0:
                    mask = np.logical_and(
                        _eta > np.min(last_eta),
                        _eta < np.max(last_eta))
                    res += np.nanmean(np.square(
                        prof[mask] - last_profile(_eta[mask])))

                last_eta = _eta[max_idx[idx]:]
                last_profile = interp1d(last_eta, prof[max_idx[idx]:])
            return res

        times_offset = np.arange(-20, 40)
        lse = np.vectorize(lse_offset)(times_offset)
        fit_times = fit_times - times_offset[np.nanargmin(lse)]

        # Plot curve
        figure()
        plt.semilogy(times_offset, lse)
        plt.plot(times_offset[np.nanargmin(lse)], lse[np.nanargmin(lse)], 'x')

        # Get explected litterature values
        protein = results.at[fn, 'prot']
        salt = results.at[fn, 'salt']
        expected_mobility = proteins_infos[protein]['Mobility']
        beta = salts_infos[salt]['beta']
        expected_Gp = expected_mobility * beta * gamma_over_mu_beta
        expected_radius = proteins_infos[protein]['Hydrodynamic radius']
        expected_Dp = DRh / expected_radius

        # Plot and fit
        Dp, Gp = fit_and_plot(fit_profiles, fit_times, fit_positions, max_idx,
                              beta_salt, Ds, fit_time_mask,
                              expected_Dp=expected_Dp,
                              expected_Gp=expected_Gp)

        # Set title and save plot
        plt.title(name)
        plt.savefig(os.path.join(save_dir, name + '_eta.pdf'))
        plt.show()

        # Save results
        results.loc[fn, 'Diffusiophoresis fit'] = Gp
        results.loc[fn, 'Diffusion fit'] = Dp
        results.loc[fn, 'protein'] = protein
        results.loc[fn, 'salt'] = salt

    results_list[set_name] = results
# %% Plot summary
figure()
for set_name in results_list:
    results = results_list[set_name]
    Rh = np.asarray([proteins_infos[protein]['Hydrodynamic radius']
                     for protein in results.loc[:, 'protein']])[0]
    plt.errorbar(Rh * 1e9,
                 np.abs(np.mean(results.loc[:, 'Diffusion fit'])),
                 np.std(results.loc[:, 'Diffusion fit']),
                 fmt='x',
                 label=set_name)

plt.loglog([2, 6, 9], DRh / np.asarray([2e-9, 6e-9, 9e-9]), 'k-.',
           label='Expected')
plt.ylim([1e-11, 3e-10])
plt.legend()
plt.xlabel('Hydrodynamic radius [$nm$]')
plt.ylabel('Fitted Diffusion [$m^2/s$]')
plt.savefig(os.path.join(save_dir, 'Results_diffusion.pdf'))


figure()
mus = []
for set_name in results_list:
    results = results_list[set_name]
    mobility = np.asarray([proteins_infos[protein]['Mobility']
                           for protein in results.loc[:, 'protein']])
    mobility = mobility[0]
    beta = np.asarray([salts_infos[salt]['beta']
                       for salt in results.loc[:, 'salt']])
    fitted_mobility = (results.loc[:, 'Diffusiophoresis fit']
                       / beta / gamma_over_mu_beta)

    plt.errorbar(np.abs(mobility),
                 np.abs(np.mean(fitted_mobility.to_numpy())),
                 np.std(fitted_mobility.to_numpy()),
                 fmt='x',
                 label=set_name)
    mus.append(mobility)

plt.plot(mus, mus, 'k-.', label='Expected')

plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.xlabel('Litterature Mobility [$m^2 / (V s)$]')
plt.ylabel('Fitted Mobility [$m^2 / (V s)$]')
plt.savefig(os.path.join(save_dir, 'Results_diffusiophoresis.pdf'))
