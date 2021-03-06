#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 07:15:24 2019

@author: quentinpeter

Fit diffusiophoresis experiemnts.

This script is distributed under CC BY-NC-SA version 4.0. You should have
recieved a copy of the licence along with it. If not, see
https://creativecommons.org/licenses/by-nc-sa/4.0/.
"""
import numpy as np
from scipy.special import erf
from scipy.integrate import solve_bvp
from scipy.interpolate import interp1d
from scipy.ndimage.filters import maximum_filter
from scipy.optimize import curve_fit

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
    """Normalise the profiles before fitting."""
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


def fit_data(profiles, times, positions,
             diffusion_salt, eta_max, idx_start, profiles_std):
    """Get data"""
    # Less than 20% of the fluorescence must be in the last fifth of the channel
    start_mask = (0 < positions) & (positions < 400e-6)
    end_mask = (400e-6 < positions) & (positions < 500e-6)
    time_mask = (
        np.nanmean(profiles[:, start_mask], axis=1)
        > 5 * np.nanmean(profiles[:, end_mask], axis=1))
    mask_valid = np.isfinite(profiles)
    mask_valid = np.logical_and(mask_valid, time_mask[:, np.newaxis])
    if idx_start is not None:
        for idx in range(len(mask_valid)):
            mask_valid[idx, :idx_start[idx]] = False

    m_positions = (positions[np.newaxis] * np.ones_like(times[:, np.newaxis]))
    m_positions = m_positions[mask_valid]

    m_times = (np.ones_like(positions[np.newaxis]) * times[:, np.newaxis])
    m_times = m_times[mask_valid]

    m_profiles = profiles[mask_valid]
    if profiles_std:
        m_profiles_std = profiles_std[mask_valid]
    else:
        m_profiles_std = None

    eta = m_positions / np.sqrt(4 * diffusion_salt * m_times)

    mask_range = eta_max > eta

    eta = eta[mask_range]
    
    m_profiles = m_profiles[mask_range]
    if m_profiles_std:
        m_profiles_std = m_profiles_std[mask_range]
    return eta, m_profiles, m_profiles_std


def fit_diffusiophoresis(profiles, times, positions, idx_start, ratio_salt,
                         diffusion_salt, time_mask, profiles_std=None):
    """Fit diffusiophoresis"""
    if not np.any(time_mask):
        raise RuntimeError

    profiles = profiles[time_mask]
    if profiles_std:
        profiles_std = profiles_std[time_mask]
    times = times[time_mask]
    idx_start = idx_start[time_mask]

    # Estimate initial values from eta_max
    eta_max = np.min(
        positions[idx_start] / np.sqrt(4 * diffusion_salt * times))
    if eta_max > 0.4:
        init = [-2, 0]
    elif eta_max > 0.3:
        init = [-3, -1.5]
    else:
        init = [-3, -3]

    def fit_func(X, log_Dp, log_Gp):
        """Fit function."""
        Dp, Gp = np.exp([log_Dp, log_Gp])
        eta = 10 ** np.linspace(-4, 1, 1000)
        eta[0] = 0
        fit = get_similarity(eta, ratio_salt, Gp, Dp)
        fit_curve = interp1d(fit.x, fit.y[0] / np.max(fit.y[0]))
        return fit_curve(X)

    eta, m_profiles, m_profiles_std = fit_data(
        profiles, times, positions, diffusion_salt, 10, idx_start, profiles_std)
    fit = curve_fit(fit_func , eta, m_profiles, init, absolute_sigma=True)

    Dp, Gp = np.exp(fit[0])
    # Error propagation
    Dp_std, Gp_std = np.exp(fit[0]) * np.sqrt(np.diag(fit[1]))
    return (
        Dp * diffusion_salt, Gp * diffusion_salt,
        Dp_std * diffusion_salt, Gp_std * diffusion_salt
        )


def color(time):
    """Get the color for a given time."""
    # Plot
    cmap = matplotlib.cm.get_cmap('plasma')
    norm = LogNorm(vmin=.1, vmax=10)
    return cmap(norm(time/60))


def plot_diffusiophoresis(profiles, times, positions, idx_start,
                          ratio_salt, diffusion_salt, time_mask,
                          fit_Dp, fit_Gp, *,
                          plot_freq=1, expected_Dp=None, expected_Gp=None,
                          ax=None):
    # Plot
    cmap = matplotlib.cm.get_cmap('plasma')
    norm = LogNorm(vmin=.1, vmax=10)

    if ax is None:
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
        fit_Gp / diffusion_salt,
        fit_Dp / diffusion_salt)

    expected_similarity = None
    if expected_Dp and expected_Gp:
        expected_similarity = get_similarity(
            eta_tmp, ratio_salt,
            expected_Gp / diffusion_salt,
            expected_Dp / diffusion_salt)

    # Plot selected parts
    ax.plot(0, 0, '--', c='gray', label='Data')
    for idx, prof in enumerate(profiles):
        if (idx + plot_freq//2) % plot_freq != 0:
            continue
        _eta = positions / np.sqrt(4 * diffusion_salt * times[idx])
        ax.plot(_eta, prof, '--', c=color(times[idx]))

    ax.plot(0, 0, '-', c='gray', label='Selected Data')
    for idx, prof in enumerate(profiles):
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
    ax.set_xlabel(r'$\eta$')
    ax.set_ylabel('Normalised profile')
