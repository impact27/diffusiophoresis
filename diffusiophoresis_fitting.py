#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 07:15:24 2019

@author: quentinpeter
"""
import numpy as np
from scipy.special import erf
from scipy.integrate import solve_bvp
from scipy.interpolate import interp1d
from scipy.ndimage.filters import maximum_filter, gaussian_filter
from scipy.optimize import minimize

from matplotlib.colors import LogNorm
import matplotlib
import matplotlib.pyplot as plt


def get_similarity(eta, beta, gamma_p, D_p):
    """
    Get a similarity solution

    eta is the similarity variable (x / sqrt(4 D_s t))
    beta is the salt ratio
    gamma_p is the diffusiophoresis coefficient / Ds
    D_p is the diffusion coefficient of the proteins / Ds
    """
    def c(eta):
        """Salt concentration as a function of eta."""
        return beta + (1 - beta) * erf(eta)

    def dc(eta):
        """Derivative of the salt concentration profile."""
        return (1 - beta) * 2 / np.sqrt(np.pi) * np.exp(-eta**2)

    def d2c(eta):
        """Second derivative of the salt concentration profile."""
        return (beta - 1) * 4 / np.sqrt(np.pi) * eta * np.exp(-eta**2)

    def dlnc(eta):
        """
        Derivative of the salt concentration profile over the concentration.
        """
        return dc(eta) / c(eta)

    def d2lnc(eta):
        """Derivative of the above."""
        return (d2c(eta) * c(eta) - dc(eta)**2) / c(eta)**2

    def second(eta, y):
        """
        Second derivative of the proteins concentartion profile.

        y[0] is the proteins concentration and y[1] the derivative
        """
        return ((gamma_p * dlnc(eta) - 2 * eta) * y[1]
                + gamma_p * d2lnc(eta) * y[0]) / D_p

    def fun(eta, y):
        """
        derivative of y, where
        y[0] is the proteins concentration and y[1] the derivative.
        """
        return [y[1], second(eta, y)]

    def bc(ya, yb):
        """
        Boundary conditions:
            Concentration on the left is 1, concentration on the right is flat
        """
        return [ya[0] - 1, yb[1]]

    y = np.zeros((2, len(eta)))
    y[0, 0] = 1

    return solve_bvp(fun, bc, eta, y)


def get_max_infl(res):
    """Get maximum and inflexion eta."""
    arg_max = np.argmax(res.y[0])
    if arg_max == 0:
        eta_max = np.nan
    else:
        max_slice = slice(arg_max-1, arg_max+2)
        max_fit = np.polyfit(res.x[max_slice], res.y[0, max_slice], 2)
        eta_max = - max_fit[1] / (2 * max_fit[0])

    arg_infl = np.argmin(res.y[1])
    infl_slice = slice(arg_infl-1, arg_infl+2)
    infl_fit = np.polyfit(res.x[infl_slice], res.y[1, infl_slice], 2)
    eta_infl = - infl_fit[1] / (2 * infl_fit[0])

    return eta_max, eta_infl


def fit_diffusiophoresis_1d(
        diffusion_protein, diffusion_salt,
        salt_in_M, salt_out_M, measured_eta, is_inflexion=True):

    if len(np.shape(diffusion_protein)) > 0:
        return np.asarray([
                fit_diffusiophoresis_1d(
                    Dp, diffusion_salt,
                    salt_in_M, salt_out_M, eta, is_inflexion)
                    for Dp, eta in zip(diffusion_protein, measured_eta)])

    if len(np.shape(diffusion_salt)) > 0:
        return np.asarray([
                fit_diffusiophoresis_1d(
                    diffusion_protein, Ds,
                    salt_in_M, salt_out_M, eta, is_inflexion)
                    for Ds, eta in zip(diffusion_salt, measured_eta)])

    if len(np.shape(salt_in_M)) > 0:
        return np.asarray([
                fit_diffusiophoresis_1d(
                    diffusion_protein, diffusion_salt,
                    Cs_in, salt_out_M, eta, is_inflexion)
                    for Cs_in, eta in zip(salt_in_M, measured_eta)])

    if len(np.shape(salt_out_M)) > 0:
        return np.asarray([
                fit_diffusiophoresis_1d(
                    diffusion_protein, diffusion_salt,
                    salt_in_M, Cs_out, eta, is_inflexion)
                    for Cs_out, eta in zip(salt_out_M, measured_eta)])

    beta_salt = salt_out_M / salt_in_M
    diffusion_ratio = diffusion_protein / diffusion_salt

    # assert np.all(1e-3 < measured_eta < 1)
    eta = 10 ** np.linspace(-4, 1, 1000)
    eta[0] = 0

    phoresis_ratios = 10**np.linspace(-3, 0.5, 100)

    eta_fitted = np.zeros(len(phoresis_ratios))

    if is_inflexion:
        eta_idx = 1
    else:
        eta_idx = 0

    for idx, phoresis_ratio in enumerate(phoresis_ratios):
        fit = get_similarity(eta, beta_salt, phoresis_ratio, diffusion_ratio)
        eta_fitted[idx] = get_max_infl(fit)[eta_idx]

    interp_curve = interp1d(eta_fitted, phoresis_ratios)

    diffusiophoresis_coefficient = interp_curve(measured_eta) * diffusion_salt

    return diffusiophoresis_coefficient


def similarity_LSE(fit_eta, fit_norm, norm_profiles, times, positions,
                   diffusion_salt, idx_start=None):
    """Compute lse between fit and data"""

    fit_curve = interp1d(fit_eta, fit_norm)
    res = 0
    for idx, time in enumerate(times):
        prof = norm_profiles[idx]
        mask_valid = np.isfinite(prof)
        prof = prof[mask_valid]
        factor = 1 - np.mean(prof[positions[mask_valid] > 400e-6])
        eta = positions[mask_valid] / np.sqrt(4 * diffusion_salt * time)
        mask = eta < np.max(fit_eta)
        if idx_start is not None:
            mask[:idx_start[idx]] = False
        res += (np.sum(np.square(
            prof[mask]
            - fit_curve(eta[mask])))
            * factor)
    return np.sqrt(res)


def normalise_profile(profiles):
    # renormalise positons
    maxfilt = maximum_filter(profiles, footprint=np.ones((1, 31)))
    max_mask = profiles == maxfilt
    max_mask = np.logical_and(max_mask, profiles > 0.1 * np.max(
        profiles, axis=1)[:, np.newaxis])
    idx_max = np.argmax(max_mask[:, ::-1], axis=1)
    # Ignore maxs close to end
    mask_time = idx_max > 50
    idx_max = np.shape(profiles)[1] - 1 - idx_max
    max_int = profiles[
        np.arange(np.shape(profiles)[0]), idx_max]
    norm_profiles = profiles / max_int[:, np.newaxis]
    return norm_profiles, mask_time, idx_max


def get_Gamma(profiles, times, positions,
              salt_out_M, salt_in_M,
              diffusion_protein, diffusion_salt):

    beta_salt = salt_out_M / salt_in_M
    diffusion_ratio = diffusion_protein / diffusion_salt

    # assert np.all(1e-3 < measured_eta < 1)
    eta = 10 ** np.linspace(-4, 1, 1000)
    eta[0] = 0

    phoresis_ratios = 10**np.linspace(-3, 0.5, 100)

    LSE = np.zeros(len(phoresis_ratios)) * np.nan
    for idx, phoresis_ratio in enumerate(phoresis_ratios):
        fit = get_similarity(eta, beta_salt, phoresis_ratio, diffusion_ratio)

        LSE[idx] = similarity_LSE(fit.x, fit.y[0] / np.max(fit.y[0]),
                                  profiles, times, positions, diffusion_salt)

    return phoresis_ratios[np.argmin(LSE)] * diffusion_salt


def fit_norm_profile(profiles, times, positions, idx_start,
                     ratio_salt, diffusion_salt):

    def lse_diffusiophoresis(x):
        Dp, Gp = x / 100
        # assert np.all(1e-3 < measured_eta < 1)
        eta = 10 ** np.linspace(-4, 1, 1000)
        eta[0] = 0
        fit = get_similarity(eta, ratio_salt, Gp, Dp)
        return similarity_LSE(fit.x, fit.y[0] / np.max(fit.y[0]),
                              profiles, times, positions, diffusion_salt,
                              idx_start)

    return minimize(lse_diffusiophoresis, [1, 5]).x/100


def color(time):
    # Plot
    cmap = matplotlib.cm.get_cmap('plasma')
    norm = LogNorm(vmin=.1, vmax=10)
    return cmap(norm(time/60))


def fit_and_plot(profiles, times, positions, idx_start,
                 ratio_salt, diffusion_salt, time_mask, *, plot_freq=1,
                 expected=None, expected_Dp=None, expected_Gp=None):
    """Fit the profiles and plot the result."""

    if not np.any(time_mask):
        raise RuntimeError

    # Fit
    fit_diffusiophoresis = fit_norm_profile(
        profiles[time_mask], times[time_mask], positions,
        idx_start[time_mask], ratio_salt, diffusion_salt)

    Gp = fit_diffusiophoresis[1] * diffusion_salt
    Dp = fit_diffusiophoresis[0] * diffusion_salt

    # Plot
    cmap = matplotlib.cm.get_cmap('plasma')
    norm = LogNorm(vmin=.1, vmax=10)

    plt.figure()
    # Create dummy colormap for times
    colors = plt.imshow([[.1, 10], [.1, 10]], cmap=cmap, norm=norm)
    plt.clf()

    fig, ax = plt.subplots()
    fig.colorbar(colors, ax=ax).set_label(label='Time [min]')

    # Get similarity solution
    eta_tmp = 10 ** np.linspace(-4, 1, 1000)
    eta_tmp[0] = 0
    similarity_solution = get_similarity(
        eta_tmp, ratio_salt,
        fit_diffusiophoresis[1],
        fit_diffusiophoresis[0])

    expected_similarity = None
    if expected_Dp and expected_Gp:
        expected_similarity = get_similarity(
            eta_tmp, ratio_salt,
            expected_Gp / diffusion_salt,
            expected_Dp / diffusion_salt)

    # Plot selected parts
    plt.plot(0, 0, '--', c='gray', label='Data')
    for idx, prof in enumerate(profiles):
        if (idx + plot_freq//2) % plot_freq != 0:
            continue
        _eta = positions / np.sqrt(4 * diffusion_salt * times[idx])
        ax.plot(_eta, prof, '--', c=color(times[idx]))

    plt.plot(0, 0, '-', c='gray', label='Selected Data')
    for idx, prof in enumerate(profiles[time_mask]):
        if (idx + plot_freq//2) % plot_freq != 0:
            continue
        if not time_mask[idx]:
            continue
        time = times[idx]
        _eta = positions / np.sqrt(4 * diffusion_salt * time)
        pos_idx = idx_start[idx]
        ax.plot(_eta[pos_idx:], prof[pos_idx:], c=color(time))

    if expected_similarity is not None:
        ax.plot(expected_similarity.x,
                expected_similarity.y[0] / np.max(expected_similarity.y[0]),
                '-.', color='black', label='Expected')

    ax.plot(similarity_solution.x,
        similarity_solution.y[0]/np.max(similarity_solution.y[0]),
        '-.', color='tab:cyan', label='Fit')
    ax.set_ylim([-0.1, 1.1])
    ax.set_xlim([0, 1])
    ax.legend()
    ax.set_xlabel('$\eta$')
    ax.set_ylabel('Normalised profile')

    return Dp, Gp
