# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 14:52:31 2018

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, imshow
from scipy.ndimage.filters import maximum_filter
import sys
from scipy.optimize import curve_fit
import pandas as pd

from diffusiophoresis_fitting import (fit_diffusiophoresis_1d,
                                      get_similarity,
                                      get_max_infl)

# %%
sets = []
# names = ['1d', '2d', '3d']
# sets.append(dict(
#     set_name = 'Dim',
#     numbers = [1, 2, 3],
#     names = names,
#     axes = [np.load(f'k-device_gamma-1p5e-10_{nm}_axes.npz') for nm in names],
#     C_p = [np.load(f'k-device_gamma-1p5e-10_{nm}_C_p{nm[0]}.npy') for nm in names],
#     xlog = False))

# names = ['0p1uM', '1uM', '10uM', '100uM', '1mM', '10mM']
# sets.append(dict(
#     set_name = 'CsOut',
#     names = names,
#     numbers = [0.1e-6, 1e-6, 10e-6, 100e-6, 1e-3, 10e-3],
#     axes = [np.load(f'k-device_gamma-1p5e-10_CsOut_{nm}_axes.npz') for nm in names],
#     C_p = [np.load(f'k-device_gamma-1p5e-10_CsOut_{nm}_C_p2.npy') for nm in names],
#     xlog = True))

names = np.fromstring('-2e-11 -5e-11 -1e-10 -2e-10 -5e-10', sep=' ')
sets.append(dict(
    set_name = 'Gamma',
    numbers = [2e-11, 5e-11, 1e-10, 2e-10, 5e-10],
    names = names,
    axes = [np.load(f'k-device_gamma{nm}_axes.npz') for nm in names],
    C_p = [np.load(f'k-device_gamma{nm}_C_p2.npy') for nm in names],
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

# names = ['1e-10', '2e-10', '5e-10', '1e-09', '2e-09', '5e-09', '1e-08']
# sets.append(dict(
#     set_name = 'Ds',
#     names = names,
#     numbers = [1.e-10, 2.e-10, 5.e-10, 1.e-09, 2.e-09, 5.e-09, 1.e-08],
#     axes = [np.load(f'k-device_Ds_{nm}_axes.npz') for nm in names],
#     C_p = [np.load(f'k-device_Ds_{nm}_C_p2.npy') for nm in names],
#     xlog = True))

# names = '5e-12 1e-11 2e-11 5e-11 1e-10 2e-10 5e-10'.split(' ')
# sets.append(dict(
#     set_name = 'Dp',
#     names = names,
#     numbers = np.asarray([5.e-12, 1.e-11, 2.e-11, 5.e-11, 1.e-10, 2.e-10, 5.e-10]),
#     axes = [np.load(f'k-device_Dp_{nm}_axes.npz') for nm in names],
#     C_p = [np.load(f'k-device_Dp_{nm}_C_p2.npy') for nm in names],
#     xlog = True))
# %%
for aset in sets:
    ch_width = 50e-6
    real_X0 = ch_width / 2
    real_X0 = 0
    Dp = 5.9e-11
    Ds = 1e-9
    Gamma = 1.5e-10
    Cs_in = 2e-1
    Cs_out = 1e-7

    set_name, names, numbers, axes, C_p, xlog = [
        aset[key] for key in ('set_name', 'names', 'numbers',
                              'axes', 'C_p', 'xlog')]

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


    # %% plot profiles
    # for i in range(len(names)):
    #     T = axes[i]['t'][5::10]
    #     figure()
    #     for j, t in enumerate(T):
    #         X = np.arange(np.shape(C_p)[2])
    #         plot(X, 1e3 * C_p[i, j * 10], label=f"{T[j]} min")  # , c=f'C{i}')
    #         # plot(X[:-1], np.diff(1e3 * C_p[i, j * 10]))#, c=f'C{i}')
    #     plt.legend()
    #     # plt.xscale('log')
    #     plt.xlabel('position [um]')
    #     plt.ylabel('C [uM]')
    #     plt.title(f'{set_name}_{names[i]}')
    #     plt.savefig(f'figs/profile_{set_name}_{names[i]}.pdf')

    #%%
    #for idx_t in range(10, 100, 10):
    #    figure()
    #    for i in range(len(names)):
    #
    #        plot(1e3 * (C_p[i, idx_t]/np.max(C_p[i, idx_t, 30:])),
    #             c=f'C{i}')
    #    plt.legend(names)
    #    plt.xlabel('Position [uM]')
    #    plt.ylabel('C [uM]')
    #    plt.title(f'{set_name}_{names[i]}')
    #    plt.savefig(f'figs/profile_{set_name}_{names[i]}.pdf')

    #%% plot max intensity

    # figure()
    # for i in range(len(names)):
    #     plot(axes[i]['t'], np.max(C_p[i], axis=1)*1000, label=names[i])
    # plt.legend()
    # plt.xlabel('Time [min]')
    # plt.ylabel('Max C [uM]')
    # plt.title(f'{set_name} Intensity')
    # plt.savefig(f'figs/MaxC_{set_name}.pdf')

    #%%

    def time_fit(x, Dm1):
        return Dm1 * np.square(x - real_X0)

    def square_fit(T, X):
        mask = np.isfinite(X)
        T, X = T[mask], X[mask]
        N = len(T)
        start = int(np.sqrt(N))
        threshold = 0
        def fun(x, Dm1):
            return time_fit(x, Dm1 * 1e11)
        for i in range(start, N):
            try:
                params = curve_fit(fun, X[:i], T[:i])
            except RuntimeError:
                pass
            else:
                fit = params[0]
                LSE = np.sqrt(np.mean(np.square(T[:i] - time_fit(X[:i], *fit))))
                if threshold == 0:
                    threshold = 2 * LSE
                elif LSE > threshold:
                    break
                D = 1e-11/fit[0]
        return D

    def plot_point(callback, xrange, ylabel, max_infl_idx):
        # fig_position = figure()
        fig_time = figure()
        # fig_norm = figure()
        results = pd.DataFrame(index=numbers)
        for i in range(len(names)):

            position = callback(i)
            time = axes[i]['t'] * 60

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

            figure()
            for j in range(len(position) // 10):
                try:
                    X = (np.arange(np.shape(C_p)[2] - 1) * 1e-6) / np.sqrt(4 * Dsi * time[10*j])
                    Y = 1e3 * C_p[i, j * 10]
                    idx = int(position[10*j]*1e6)
                    plot(X, np.diff(Y))
                    plt.semilogx(idx * 1e-6 / np.sqrt(4 * Dsi * time[10*j]) , np.diff(Y)[idx], 'rx')
                    plt.xlim([0.1, 1])
                except:
                    pass

            eta = (position - real_X0) / np.sqrt(4 * Dsi * time)

            mask = np.isfinite(eta)
            mask[0] = False
            mask[position < xrange[0]] = False
            mask[position > xrange[1]] = False

            D = square_fit(time[mask], position[mask])
            fit_eta = np.sqrt(D/(4 * Dsi))

            beta_salt = Cs_outi / Cs_in
            diffusion_ratio = Dpi / Dsi
            phoresis_ratio = Gammai / Dsi

            eta_tmp = 10 ** np.linspace(-4, 1, 1000)
            eta_tmp[0] = 0

            fit = get_similarity(eta_tmp, beta_salt, phoresis_ratio, diffusion_ratio)
            eta_sim = get_max_infl(fit)[max_infl_idx]

            # eta_step = 0.02
            # mask = np.isfinite(eta)
            # bins = np.arange(np.min(eta[mask]), np.max(eta[mask]) + eta_step, eta_step)
            # hist, bins = np.histogram(eta[mask], bins)
            # eta_min = bins[np.argmax(hist)] - eta_step
            # eta_max = eta_min + 3 * eta_step
            # mask[eta < eta_min] = False
            # mask[eta > eta_max] = False
            # fit_eta = np.mean(eta[mask])

            results.at[numbers[i], "eta"] = fit_eta
            results.at[numbers[i], "eta_sim"] = eta_sim

            # figure(fig_position.number)
            # plt.loglog(position * 1e6, eta, '-', label=names[i], c=f'C{i}')
            # plt.plot(position * 1e6, time * 0 + fit_eta, '--', c=f'C{i}')
            # # plt.plot(position * 1e6, time * 0 + eta_sim, '-.', c=f'C{i}')
            # plt.xlabel('position [um]')

            # figure(fig_norm.number)
            # plt.semilogx(position * 1e6, eta / fit_eta, '-', label=names[i], c=f'C{i}')
            # plt.plot(position * 1e6, np.ones_like(time), '-.', c=f'C{i}')
            # plt.xlabel('position [um]')

            figure(fig_time.number)
            plt.loglog(time / 60, eta, '-', label=names[i], c=f'C{i}')
            plt.plot(time / 60, time * 0 + fit_eta, '--', c=f'C{i}')
            # plt.plot(time / 60, time * 0 + eta_sim, '-.', c=f'C{i}')
            plt.xlabel('Time [min]')

        # for fig, title in zip([fig_position, fig_time], ['position', 'time']):
        for fig, title in zip([fig_time], ['time']):
            figure(fig.number)
            plt.legend()
            plt.ylabel(f'{ylabel} eta')
            plt.title(f'{set_name} {ylabel}')
            plt.ylim([results.loc[:, "eta"].min() * 0.8,
                      results.loc[:, "eta"].max() * 1.2])
            plt.savefig(f'figs/{ylabel}_{set_name}_{title}.pdf')

        # figure(fig_norm.number)
        # plt.legend()
        # plt.ylabel(f'{ylabel} ratio')
        # plt.title(f'{set_name} {ylabel}')
        # plt.savefig(f'figs/{ylabel}_{set_name}_norm_position.pdf')

        figure()
        plt.plot(numbers, results["eta"], 'o')
        # plt.plot(numbers, fit_function(numbers))
        plt.xlabel(set_name)
        plt.xscale('log')
        # plt.yscale('log')
        # fit = np.polyfit(numbers, results["eta"], 1)
        # plt.plot(numbers, np.poly1d(fit)(numbers), '--')
        # fit_log = np.polyfit(np.log(numbers), np.log(results["eta"]), 1)
        # plt.plot(numbers, np.exp(np.poly1d(fit_log)(np.log(numbers))), '-.')
        # plt.title(f"{fit[0]:.2e};{fit[1]:.2e};{fit_log[0]:.2e};{fit_log[1]:.2e}")
        return results["eta"], results["eta_sim"]

    def get_max(i):
        argmax = np.asarray(np.argmax(C_p[i, :, 20:], axis=1)+20, float) * 1e-6
        argmax[argmax == 20e-6] = np.nan
        argmax[argmax == 500e-6] = np.nan
        return argmax

    xrange = [60e-6, 400e-6]

    # max_eta, max_eta_sim = plot_point(get_max, xrange, "Max", 0)
    #%
    def get_infl(i):
        Y = np.diff(C_p[i], axis=1)
        X = (np.arange(np.shape(Y)[1]) + 0.5) * 1e-6
        res = np.zeros(np.shape(Y)[0])
        for idx in range(np.shape(Y)[0]):
            amin = np.argmin(Y[idx, 30:]) + 30
            if amin > np.shape(Y)[0] - 50:
                aslice = slice(amin-3, amin+4)
                fit = np.polyfit(X[aslice], Y[idx, aslice], 2)
                res[idx] = - fit[1] / (2 * fit[0])
            else:
                res[idx] = np.nan
        return res

    infl_eta, infl_eta_sim = plot_point(get_infl, xrange, "Inflection", 1)
    #%%
    figure()
    plot(numbers, infl_eta / infl_eta_sim, 'x')
    plt.xlabel(set_name)
    plt.ylabel('eta ratio')
    if xlog:
        plt.xscale('log')

    # #%%
    # figure()
    # plt.plot(numbers, max_eta, 'o')
    # plt.plot(numbers, infl_eta, 'o')


    # plot(numbers, numbers*0 + xmax, 'C0')
    # plot(numbers, numbers*0 + xinfl, 'C1')

    # figure()
    # plt.plot(numbers, xmax / max_eta, 'o')
    # plt.plot(numbers, xinfl / infl_eta, 'o')


    # %% fit with new script
    fitted = fit_diffusiophoresis_1d(Dp, Ds, Cs_in, Cs_out, infl_eta, is_inflexion=True)
    var = np.ones(len(fitted)) * Gamma
    figure()
    plt.semilogy(numbers, fitted, 'x')
    plot(numbers, var, 'x')
    if xlog:
        plt.xscale('log')
    plt.xlabel(set_name)
    figure()
    plt.plot(numbers, var / fitted, 'x')
    if xlog:
        plt.xscale('log')
    plt.xlabel(set_name)
    # plt.ylim([0.5, 1.5])

    # %% plot profiles
    for i in range(1):#len(names)):
        Dsi = Ds
        if set_name == 'Ds':
            Dsi = Ds[i]
        T = axes[i]['t'][5::10]
        figure()
        for j, t in enumerate(T):
            X = (np.arange(np.shape(C_p)[2]) * 1e-6 - real_X0) / np.sqrt(4 * Dsi * t * 60)
            X2 = np.arange(np.shape(C_p)[2])
            Y = 1e3 * C_p[i, j * 10]
            plot(X[1:], np.diff(Y/np.max(Y)), label=f"{T[j]} min")#, c=f'C{i}')
            # plot(X[:-1], np.diff(1e3 * C_p[i, j * 10]))#, c=f'C{i}')

        Gammai = Gamma
        if set_name == 'Gamma':
            Gammai = Gamma[i]
        Dpi = Dp
        if set_name == 'Dp':
            Dpi = Dp[i]
        Cs_outi = Cs_out
        if set_name == 'CsOut':
            Cs_outi = Cs_out[i]
        beta_salt = Cs_outi / Cs_in
        diffusion_ratio = Dpi / Dsi

        gammai_fitted = fitted[i]
        phoresis_ratio = Gammai / Dsi
        phoresis_ratio_fitted = gammai_fitted / Dsi
        eta_tmp = 10 ** np.linspace(-4, 1, 1000)
        eta_tmp[0] = 0
        fit = get_similarity(eta_tmp, beta_salt, phoresis_ratio, diffusion_ratio)
        plot(fit.x[1:], np.diff(fit.y[0]/np.max(fit.y[0])), 'r--')

        fit = get_similarity(eta_tmp, beta_salt, phoresis_ratio_fitted, diffusion_ratio)
        plot(fit.x[1:], np.diff(fit.y[0]/np.max(fit.y[0])), 'g--')

        plt.legend()
        plt.xscale('log')
        plt.xlabel('eta')
        plt.ylabel('C [uM]')
        plt.title(f'{set_name}_{names[i]}')
        plt.savefig(f'figs/profile_{set_name}_{names[i]}_eta.pdf')
        plt.ylim([-0.05, 0.05])
        plt.xlim([0.1, 1])

