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
sys.path.append('../COMSOL')
from diffusiophoresis_fitting import fit_diffusiophoresis_1d, get_similarity

cmap = matplotlib.cm.get_cmap('plasma')

fn = 'output_2017/20171116/small_channel/o0.01mMLYS_i200mMKIO3/o0.01mMLYS_i200mMKIO3.npz'

fns = glob('output_*/*/small_channel/*BSA*LiCl*/*.npz')

# fns = ['output_2017/20171122/small_channel/o0.01mMThy_i200mMLiCl/o0.01mMThy_i200mMLiCl.npz']
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
    results.loc[idx, 'prot'] = prot
    results.loc[idx, 'salt'] = salt
    results.loc[idx, 'prot_conc'] = pconc
    results.loc[idx, 'salt_conc'] = sconc
    date = re.findall(
            r'/(\d{8})/', idx)[0]
    results.loc[idx, 'date'] = date

Ds_dic = {
        'KIO3': 1.51e-9,
        'LiCl': 1.53e-9,
        'KCl': 1.995e-9,
        'GLU': 1e-9,
        'HCl': 1e-9,
        'Urea': 1e-9,
        'NaOH': 1e-9,
        'H2O': 1e-9,
        'GLY': 1e-9,
        }

RHp_dic = {
        'MYO': 2.12e-9,
        'B-LAC': 2.68e-9,
        'BSA': 3.48e-9,
        'LYS': 2.05e-9,
        'Lys': 2.05e-9,
        'Thy': 8.58e-9,
        }

temperature = 273.14 + 23
boltzmann = 1.38e-23
viscosity = 1e-3
DRh = temperature * boltzmann / (6 * np.pi * viscosity)


class MyClass():
    pass


ignore = ['output_2017/20171031/small_channel/o1mMBSA_i200mMLiCl_3/o1mMBSA_i200mMLiCl.npz',
          'output_2017/20171031/small_channel/o1mMBSA_i15MLiCl/o1mMBSA_i15MLiCl.npz']

best_fns = [
    # 'output_2018/20180502/small_channel/o0.5gplLys_i20mMHCl_2/o0.5gplLys_i20mMHCl.npz',
    'output_2018/20180502/small_channel/o0.1gplLys_i20mMHCl_2/o0.1gplLys_i20mMHCl.npz',
    'output_2018/20180427/small_channel/o0.5gplThy_i1MNaOH_1/o0.5gplThy_i1MNaOH.npz',
    'output_2017/20171122/small_channel/o0.001mMThy_i200mMLiCl/o0.001mMThy_i200mMLiCl.npz',
    'output_2017/20171109/small_channel/o0.1mMBSA_i2MLiCl_1/o0.1mMBSA_i2MLiCl.npz',
    'output_2017/20171109/small_channel/o1mMBSA_i2MLiCl_1/o1mMBSA_i2MLiCl.npz',
    'output_2017/20171109/small_channel/o0.01mMBSA_i2MLiCl_2/o0.01mMBSA_i2MLiCl.npz',
    'output_2017/20171109/small_channel/o0.1mMLYS_i200mMKIO3_1/o0.1mMLYS_i200mMKIO3.npz',
    'output_2017/20171109/small_channel/o0.01mMBSA_i2MLiCl_3/o0.01mMBSA_i2MLiCl.npz',


    ]
for fn in fns:
    name = os.path.splitext(os.path.basename(fn))[0]
    save_dir = 'plots_fit'
    try:
        os.mkdir(save_dir)
    except:
        pass
    # if results.at[fn, 'prot'] != 'BSA':
    #     continue
    # if results.at[fn, 'salt'] != 'LiCl':
    #     continue
    # if fn in ignore:
    #     continue
    print(fn)
    data = np.load(fn)
    profs = data['profiles']
    X = data['X_pos']
    T = data['times'][:len(profs)]

    figure()
    for idx, prof in enumerate(profs):
        # figure()
        plot(X, prof, c=cmap(T[idx]/10))
    plt.title(name)
    plt.savefig(os.path.join(save_dir, name + '_profiles.pdf'))

    # %%
    # Get functions
    Xdiff = (X[1:] + X[:-1]) / 2

    diff_prof = np.diff(profs, axis=-1)
    filtered = savgol_filter(diff_prof, 51, 2, axis=-1)

    # [:, Xdiff < 400]
    inflx_pos = np.nanargmin(filtered, axis=-1)

    pos = Xdiff[inflx_pos]
    valid = pos > 0
    valid[pos > 400] = False
    argfirst = np.argmax(np.nanmax(profs, 1) > 0.9) + 3
    valid[:argfirst] = False

    # figure()
    # for t, dp, fp, idx_inflexion, val in zip(
    #         T, diff_prof, filtered, inflx_pos, valid):
    #     # figure()
    #     plot(Xdiff, dp, c=cmap(t/10))
    #     plot(Xdiff, fp, c=cmap(t/10))
    #     if val:
    #         plot(Xdiff[idx_inflexion], fp[idx_inflexion], 'ro')
    #     else:
    #         plot(Xdiff[idx_inflexion], fp[idx_inflexion], 'rx')

    # fit = np.polyfit(np.log(T[valid]), np.log(pos[valid]), 1)
    fit = np.exp(np.mean(2 * np.log(pos[valid]) - np.log(T[valid])))
    Ds = Ds_dic[results.at[fn, 'salt']]
    Dp = DRh / RHp_dic[results.at[fn, 'prot']]
    eta = np.sqrt(fit * 1e-12 / 60 / 4 / Ds)

    Cs_out = 0.2
    Cs_in = results.at[fn, 'salt_conc']
    try:
        fitted = fit_diffusiophoresis_1d(Dp, Ds, Cs_in, Cs_out, eta, is_inflexion=True)
    except:
        fitted = np.nan

    if fitted is not np.nan:
        figure()
        peak_mask = np.logical_and(
            X[np.nanargmax(profs, axis=1)] > 0,
            np.nanmax(profs, axis=1) > 0.5)
        idx_start = np.argmax(peak_mask)
        for idx, (prof, idx_inflexion) in enumerate(zip(profs, inflx_pos)):
            if idx < idx_start:
                continue
            _eta = X * 1e-6 / np.sqrt(4 * Ds * T[idx] * 60)
            plt.semilogx(_eta, prof/np.nanmax(prof), c=cmap(T[idx]/10))

        Gamma = fitted
        beta_salt = Cs_out / Cs_in
        diffusion_ratio = Dp / Ds
        phoresis_ratio = Gamma / Ds
        eta_tmp = 10 ** np.linspace(-4, 1, 1000)
        eta_tmp[0] = 0
        mfit = get_similarity(eta_tmp, beta_salt, phoresis_ratio, diffusion_ratio)
        inf_idx = np.argmin(np.diff(mfit.y[0]))
        plot(mfit.x, mfit.y[0]/np.max(mfit.y[0]), 'r--')
        plt.ylim([-0.1, 1.1])
        plt.title(name)
        plt.xlim([1e-2, 1])
        plt.savefig(os.path.join(save_dir, name + '_eta.pdf'))

    figure()
    plt.loglog(T[valid], pos[valid], 'x')
    plt.loglog(T[valid], np.sqrt(fit * T[valid]))
    plt.title(fr'$\Gamma_p$ = {fitted:.2g} ($\eta$ = {eta:.2g})')
    plt.savefig(os.path.join(save_dir, name + '_inflection_eta.pdf'))

    plt.show()
    results.loc[fn, 'fit'] = fit
    results.loc[fn, 'eta'] = eta
    results.loc[fn, 'gamma'] = fitted

    input_type = 'l'  # input('type: ')
    results.loc[fn, 'type'] = input_type
    obj = MyClass()
    obj.plot = (T[valid], pos[valid])
    results.loc[fn, 'plot'] = obj
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
figure()
plt.semilogx(beta, results[mask]['eta'], 'x')

# %%
fn = 'output_2017/20171031/small_channel/o1mMBSA_i200mMLiCl_4/o1mMBSA_i200mMLiCl.npz'
time, position = results.loc[fn, 'plot'].plot
fit = results.loc[fn, 'fit']
figure()
plt.loglog(time, position, 'x')
plt.loglog(time, np.sqrt(fit * time))
fit2 = 15000
plt.loglog(time, np.sqrt(fit2 * time))
