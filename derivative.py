#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 13:50:39 2018

@author: quentinpeter
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, figure, show, imshow
import matplotlib
from scipy.signal import savgol_filter
from glob import glob
import pandas as pd
import re

cmap = matplotlib.cm.get_cmap('plasma')

fn = 'output_2017/20171116/small_channel/o0.01mMLYS_i200mMKIO3/o0.01mMLYS_i200mMKIO3.npz'

fns = glob('output_*/*/small_channel/*/*.npz')

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
        'KCl': 1.995e-9}


class MyClass():
    pass


ignore = ['output_2017/20171031/small_channel/o1mMBSA_i200mMLiCl_3/o1mMBSA_i200mMLiCl.npz']
for fn in fns:
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
    T = data['times']

    figure()
    for idx, prof in enumerate(profs):
        # figure()
        plot(X, prof, c=cmap(T[idx]/10))

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

    figure()
    for t, dp, fp, idx_inflexion, val in zip(
            T, diff_prof, filtered, inflx_pos, valid):
        # figure()
        plot(Xdiff, dp, c=cmap(t/10))
        plot(Xdiff, fp, c=cmap(t/10))
        if val:
            plot(Xdiff[idx_inflexion], fp[idx_inflexion], 'ro')
        else:
            plot(Xdiff[idx_inflexion], fp[idx_inflexion], 'rx')

    # fit = np.polyfit(np.log(T[valid]), np.log(pos[valid]), 1)
    fit = np.exp(np.mean(2 * np.log(pos[valid]) - np.log(T[valid])))
    Ds = Ds_dic[results.at[fn, 'salt']]
    eta = np.sqrt(fit * 1e-12 / 60 / 4 / Ds)
    figure()
    plt.loglog(T[valid], pos[valid], 'x')
    plt.loglog(T[valid], np.sqrt(fit * T[valid]))
    plt.title(eta)

    plt.show()
    results.loc[fn, 'fit'] = fit
    results.loc[fn, 'eta'] = eta

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

figure()
for date in np.unique(results[mask]['date']):
    mask2 = np.logical_and(mask, results['date'] == date)
    plt.semilogx(results[mask2]['salt_conc'], results[mask2]['eta'], 'x')
plt.xlabel('Salt conc [mM]')

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
