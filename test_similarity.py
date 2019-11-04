#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 16:32:09 2019

@author: quentinpeter
"""
import numpy as np
from matplotlib.pyplot import figure, plot
import matplotlib.pyplot as plt

from diffusiophoresis_fitting import get_similarity


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




salt_in_M = 2e-1
salt_out_M = 1e-7
diffusion_protein = 5.9e-11
diffusion_salt = 1e-9
gamma = 1.5e-10

beta_salt = salt_out_M / salt_in_M
beta_salt = 1
diffusion_ratio = diffusion_protein / diffusion_salt
phoresis_ratio = gamma / diffusion_salt

# assert np.all(1e-3 < measured_eta < 1)
eta = 10 ** np.linspace(-4, 1, 1000)
eta[0] = 0


scale = 10**np.linspace(-6, -0.1, 100)
eta_fitted = np.zeros((2, len(scale)))
max_intensity = np.zeros(len(scale))


figure(1)
figure(2)
for idx, s in enumerate(scale):
    try:
        fit = get_similarity(eta, s * beta_salt,
                             phoresis_ratio, diffusion_ratio)
        for i in [0, 1]:
            eta_fitted[i, idx] = get_max_infl(fit)[i]
        max_intensity[idx] = np.max(fit.y[0])
        if idx % 10 == 0:
            figure(0)
            plt.plot(fit.x[fit.x < 1], fit.y[0][fit.x < 1])
            figure(1)
            plt.plot(fit.x[fit.x < 1], fit.y[0][fit.x < 1] / np.max(fit.y[0]))
            figure(2)
            plt.plot(fit.x[fit.x < 1], - fit.y[1]
                     [fit.x < 1] / np.min(fit.y[1]))
    except:
        pass

figure(3)
plt.semilogx(scale, eta_fitted[0], 'x')
plt.semilogx(scale, eta_fitted[1], 'x')

figure(4)
plt.loglog(scale, max_intensity, 'x')
