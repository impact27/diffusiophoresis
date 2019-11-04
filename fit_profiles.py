#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 13:50:39 2018

@author: quentinpeter
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, figure, show, imshow
import matplotlib
from scipy.signal import savgol_filter
from glob import glob
import pandas as pd
import re
import sys
from scipy.ndimage.filters import maximum_filter
sys.path.append('../COMSOL')
from diffusiophoresis_fitting import (
    fit_and_plot, get_similarity, color)
cmap = matplotlib.cm.get_cmap('plasma')
from scipy.interpolate import interp1d
from scipy.optimize import minimize
# %% Create out directory
save_dir = 'plots_fit'
try:
    os.mkdir(save_dir)
except:
    pass
# %% Physical parameters

Ds_dic = {
    'KIO3': 1.51e-9,
    'LICL': 1.53e-9,
    'KCL': 1.995e-9,
    'GLU': 1e-9,
    'HCL': 1e-9,
    'UREA': 1e-9,
    'NAOH': 1e-9,
    'H2O': 1e-9,
    'GLY': 1e-9,
    }

beta_dic = {
    'KIO3': 0.298,
    'LICL': -0.326,
    'KCl': -0.019,
    'GLU': 0,
    'HCL': 0.642,
    'UREA': 0,
    'NAOH': -0.597,
    'H2O': 0,
    'GLY': 0,
    }

RHp_dic = {
    'MYO': 2.12e-9,
    'B-LAC': 2.68e-9,
    'BSA': 3.48e-9,
    'LYS': 2.05e-9,
    'THY': 8.58e-9,
    }

isoelectric_point_dic = {
    'MYO': 6.97,
    'B-LAC': 5.1,
    'BSA': 4.7,
    'LYS': 9.1,
    'THY': 4.5,
    }

mobility_dic = {
    'MYO': -1.4e-8,
    'B-LAC': -1.9e-8,
    'BSA': -1e-8,
    'LYS': 1.8e-8,
    'THY': -0.5e-8,
    }

temperature = 273.14 + 23
boltzmann = 1.38e-23
viscosity = 1e-3
DRh = temperature * boltzmann / (6 * np.pi * viscosity)

# %%Files to fit

# sets = [
#     ('LYS HCl', [
#     'output_2018/20180502/small_channel/o0.5gplLys_i20mMHCl_2/o0.5gplLys_i20mMHCl.npz',
#     'output_2018/20180502/small_channel/o0.1gplLys_i20mMHCl_2/o0.1gplLys_i20mMHCl.npz',
#     ]),
#     ('LYS KIO3', [
#     'output_2017/20171109/small_channel/o0.1mMLYS_i200mMKIO3_1/o0.1mMLYS_i200mMKIO3.npz',
#     'output_2017/20171116/small_channel/o0.01mMLYS_i200mMKIO3/o0.01mMLYS_i200mMKIO3.npz',
#     ]),
#     ('THY NAOH', [
#     'output_2018/20180427/small_channel/o0.5gplThy_i1MNaOH_1/o0.5gplThy_i1MNaOH.npz',
#     ]),
#     ('THY LiCl', [
#     'output_2017/20171122/small_channel/o0.001mMThy_i200mMLiCl/o0.001mMThy_i200mMLiCl.npz',
#     ]),
#     ('B-LAC LiCl', [
#     'output_2017/20171116/small_channel/o0.01mMB-LAC_i200mMLiCl/o0.01mMB-LAC_i200mMLiCl.npz',
#     'output_2017/20171116/small_channel/o0.01mMB-LAC_i200mMLiCl_2/o0.01mMB-LAC_i200mMLiCl.npz',
#     ]),
#     ('MYO LiCl', [
#     'output_2017/20171129/small_channel/o0.5gplMYO_i200mMLiCl/o0.5gplMYO_i200mMLiCl.npz',
#     ]),
#     ('BSA LiCl', [
#     'output_2017/20171116/small_channel/o0.01mMBSA_i200mMLiCl/o0.01mMBSA_i200mMLiCl.npz',
#     'output_2017/20171031/small_channel/o1mMBSA_i200mMLiCl_4/o1mMBSA_i200mMLiCl.npz',
#     'output_2017/20171109/small_channel/o0.01mMBSA_i2MLiCl_2/o0.01mMBSA_i2MLiCl.npz',
#     'output_2017/20171109/small_channel/o0.01mMBSA_i2MLiCl_3/o0.01mMBSA_i2MLiCl.npz',
#     'output_2017/20171109/small_channel/o0.1mMBSA_i2MLiCl_1/o0.1mMBSA_i2MLiCl.npz',
#     'output_2017/20171109/small_channel/o1mMBSA_i2MLiCl_1/o1mMBSA_i2MLiCl.npz',
#     'output_2017/20171031/small_channel/o1mMBSA_i15MLiCl_3/o1mMBSA_i15MLiCl.npz',
#     ]),
#     ]

sets = [
    ('Lysosyme', [
    'output_2018/20180502/small_channel/o0.5gplLys_i20mMHCl_2/o0.5gplLys_i20mMHCl.npz',
    'output_2018/20180502/small_channel/o0.1gplLys_i20mMHCl_2/o0.1gplLys_i20mMHCl.npz',
    'output_2017/20171109/small_channel/o0.1mMLYS_i200mMKIO3_1/o0.1mMLYS_i200mMKIO3.npz',
    'output_2017/20171116/small_channel/o0.01mMLYS_i200mMKIO3/o0.01mMLYS_i200mMKIO3.npz',
    ]),
    ('Thyroglobulin', [
    'output_2018/20180427/small_channel/o0.5gplThy_i1MNaOH_1/o0.5gplThy_i1MNaOH.npz',
    'output_2017/20171122/small_channel/o0.001mMThy_i200mMLiCl/o0.001mMThy_i200mMLiCl.npz',
    ]),
    ('Beta-Lactoglobulin', [
    'output_2017/20171116/small_channel/o0.01mMB-LAC_i200mMLiCl/o0.01mMB-LAC_i200mMLiCl.npz',
    'output_2017/20171116/small_channel/o0.01mMB-LAC_i200mMLiCl_2/o0.01mMB-LAC_i200mMLiCl.npz',
    ]),
    ('Myoglobin', [
    'output_2017/20171129/small_channel/o0.5gplMYO_i200mMLiCl/o0.5gplMYO_i200mMLiCl.npz',
    ]),
    ('Bovine Serum Albumin', [
    'output_2017/20171116/small_channel/o0.01mMBSA_i200mMLiCl/o0.01mMBSA_i200mMLiCl.npz',
    'output_2017/20171031/small_channel/o1mMBSA_i200mMLiCl_4/o1mMBSA_i200mMLiCl.npz',
    'output_2017/20171109/small_channel/o0.01mMBSA_i2MLiCl_2/o0.01mMBSA_i2MLiCl.npz',
    'output_2017/20171109/small_channel/o0.01mMBSA_i2MLiCl_3/o0.01mMBSA_i2MLiCl.npz',
    'output_2017/20171109/small_channel/o0.1mMBSA_i2MLiCl_1/o0.1mMBSA_i2MLiCl.npz',
    'output_2017/20171109/small_channel/o1mMBSA_i2MLiCl_1/o1mMBSA_i2MLiCl.npz',
    'output_2017/20171031/small_channel/o1mMBSA_i15MLiCl_3/o1mMBSA_i15MLiCl.npz',
    ]),
    ]

# %% Compute results
results_list = []
for set_name, fns in sets:
    results = pd.DataFrame(index=fns)
    results['name'] = [re.findall(r'\/([^\/]+)\.npz', fn)[0] for fn in fns]

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
                r'/(\d{8})/', idx)[0]
        results.loc[idx, 'date'] = date

    for fn in fns:
        print(fn)
        # Load data
        name = results.at[fn, 'name']
        data = np.load(fn)
        profiles = data['profiles']
        positions = data['X_pos'] * 1e-6
        times = data['times'][:len(profiles)] * 60
        Ds = Ds_dic[results.at[fn, 'salt']]
        Dp = DRh / RHp_dic[results.at[fn, 'prot']]
        Cs_out = 0.2
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
            max_idx[idx] = np.argmax(local_max_mask)

        # Select profiles to plot
        fit_times = times[max_idx > 0]
        fit_profiles = profiles[max_idx > 0]
        max_idx = max_idx[max_idx > 0]
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

                last_eta = _eta
                last_profile = interp1d(_eta, prof)
            return res

        times_offset = np.arange(-20, 40)
        lse = np.vectorize(lse_offset)(times_offset)
        figure()
        plt.semilogy(times_offset, lse)
        plt.plot(times_offset[np.nanargmin(lse)], lse[np.nanargmin(lse)], 'x')
        fit_times = fit_times - times_offset[np.nanargmin(lse)]

        protein = results.at[fn, 'prot']
        salt = results.at[fn, 'salt']
        expected_mobility = mobility_dic[protein]
        salt_differential_diffusion = beta_dic[salt]
        expected_Gp = expected_mobility * salt_differential_diffusion * 0.03
        expected_radius = RHp_dic[protein]
        expected_Dp = DRh / expected_radius

        Dp, Gp = fit_and_plot(fit_profiles, fit_times, fit_positions, max_idx,
                              beta_salt, Ds, fit_time_mask,
                              expected_Dp=expected_Dp,
                              expected_Gp=expected_Gp)

        # Save results
        results.loc[fn, 'Diffusiophoresis fit'] = Gp
        results.loc[fn, 'Diffusion fit'] = Dp
        results.loc[fn, 'protein'] = protein
        results.loc[fn, 'salt'] = salt

        plt.title(name)
        plt.savefig(os.path.join(save_dir, name + '_eta.pdf'))

        #%% Plots the maximum position to make sure it follows the correct law
        fit_idx_max = (np.mean(fit_times * np.square(fit_positions[max_idx])) /
                       np.mean(np.square(np.square(fit_positions[max_idx]))))

        figure()
        plt.loglog(fit_times, fit_positions[max_idx], 'x')
        plt.loglog(fit_times, np.sqrt(fit_times / fit_idx_max), '-')
        plt.title(fr'$\Gamma_p$ = {Gp:.2g}')
        # plt.savefig(os.path.join(save_dir, name + '_inflection_eta.pdf'))

        plt.show()

    # %%
    mask = np.logical_and(results['prot'] == 'BSA', results['salt'] == 'LiCl')

    figure()
    for date in np.unique(results[mask]['date']):
        mask2 = np.logical_and(mask, results['date'] == date)
        plt.semilogx(results[mask2]['prot_conc'], results[mask2]['eta'], 'x')
    plt.xlabel('Proteins conc [mM]')
    plt.tight_layout()

    figure()
    for date in np.unique(results[mask]['date']):
        mask2 = np.logical_and(mask, results['date'] == date)
        plt.semilogx(results[mask2]['salt_conc'], results[mask2]['eta'], 'x')
    plt.xlabel('Salt conc [mM]')

    figure()
    for date in np.unique(results[mask]['date']):
        mask2 = np.logical_and(mask, results['date'] == date)
        plt.semilogx(results[mask2]['salt_conc'], results[mask2]['gamma'], 'x')
    plt.xlabel('Salt conc [mM]')
    plt.ylabel('Gamma [m^2 / s]')

    beta = results[mask]['prot_conc'] / results[mask]['salt_conc']

    results_list.append((set_name, results))
#%%
figure()
for set_name, results in results_list:
    Rh = np.asarray([RHp_dic[prot] for prot in results.loc[:, 'protein']])[0]
    plt.errorbar(Rh * 1e9,
               np.abs(np.mean(results.loc[:, 'Diffusion fit'])),
               np.std(results.loc[:, 'Diffusion fit']),
               fmt='x',
               label=set_name)

plt.loglog([2, 6, 9], DRh / np.asarray([2e-9, 6e-9, 9e-9]), 'k-.', label='Expected')
plt.ylim([1e-11, 3e-10])
plt.legend()
plt.xlabel('Hydrodynamic radius [$nm$]')
plt.ylabel('Fitted Diffusion [$m^2/s$]')
plt.savefig(os.path.join(save_dir, 'Results_diffusion.pdf'))


figure()
mus = []
for set_name, results in results_list:
    mobility = np.asarray([mobility_dic[prot] for prot in results.loc[:, 'protein']])
    mobility = mobility[0]
    beta = np.asarray([beta_dic[salt] for salt in results.loc[:, 'salt']])
    fitted_mobility = (results.loc[:, 'Diffusiophoresis fit']
                        / beta / 0.03)

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