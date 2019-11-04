#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 06:30:36 2019

@author: quentinpeter
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, imshow, show
import pandas as pd

from diffusiophoresis_fitting import get_similarity, get_max_infl


Dp = 5.9e-11
Ds = 1e-9
Gamma = 1.5e-10

Cout = .1e-3
Cin = 200

beta = Cout / Cin
init = [beta, Gamma/Ds, Dp/Ds]

eta = 10 ** np.linspace(-2, 1, 1000)
eta[0] = 0


# %%
beta, gamma_p, D_p = init
betas = 10**np.linspace(-2, 0, 10)
betas = np.asarray([0.1e-6, 1e-6, 10e-6, 100e-6, 1e-3, 10e-3]) / 200e-3
# betas = np.linspace(1e-6, 0.1, 10)
result_beta = pd.DataFrame(index=betas, columns=['Max', 'Inflexion'])
figure()
for beta in betas:
    res = get_similarity(eta, beta, gamma_p, D_p)
    result_beta.loc[beta, ['Max', 'Inflexion']] = get_max_infl(res)
    plt.semilogx(res.x, res.y[0] / np.max(res.y[0]), '-')

figure()
plt.xlabel('beta')
plt.semilogx(result_beta['Max'], 'x')
plt.plot(result_beta['Inflexion'], 'x')

# plt.plot(betas, max_eta, 'o', c='C0')
# plt.semilogx(betas, infl_eta, 'o', c='C1')

# plot(betas, -betas+1)
# %%
beta, gamma_p, D_p = init

gamma_ps = 10**np.linspace(-2.5, 0, 100)
gamma_ps = [0.02, 0.05, 0.1, 0.2, 0.5]
result_gamma = pd.DataFrame(index=gamma_ps, columns=[
                            'Max', 'Inflexion'], dtype=float)

figure()
for gamma_p in gamma_ps:
    res = get_similarity(eta, beta, gamma_p, D_p)
    result_gamma.loc[gamma_p, ['Max', 'Inflexion']] = get_max_infl(res)
    plt.plot(res.x, res.y[0] / np.max(res.y[0]), '-')
plt.xlim([0, 1])
figure()
for gamma_p in gamma_ps:
    res = get_similarity(eta, beta, gamma_p, D_p)
    result_gamma.loc[gamma_p, ['Max', 'Inflexion']] = get_max_infl(res)
    plt.semilogx(res.x, res.y[0] / np.max(res.y[0]), '-')

figure()
plt.xlabel('gamma_p')
plt.loglog(result_gamma['Max'], 'x')
plt.semilogx(result_gamma['Inflexion'], 'x')

infl_fit = np.polyfit(np.log(gamma_ps), np.log(result_gamma['Inflexion']), 1)
plot(gamma_ps, np.exp(np.poly1d(infl_fit)(np.log(gamma_ps))))

# plot(gamma_ps, 1 / ((gamma_ps)**-0.5 + 1/3), 'C0')

# plt.plot(gamma_ps, max_eta, 'o', c='C0')
# plt.plot(gamma_ps, infl_eta, 'o', c='C1')


# figure()
# plt.xlabel('gamma_p')
# plt.semilogx(gamma_ps, result_gamma['Max'].to_numpy() / max_eta.to_numpy() - 1, 'x')
# plt.semilogx(gamma_ps, result_gamma['Inflexion'].to_numpy() / infl_eta.to_numpy() - 1, 'x')
# plot(gamma_ps, 1 / ((gamma_ps)**-0.5 + 1/3), 'C0')


# plot(gamma_ps, 1 / ((gamma_ps)**-1 + 1/3), 'C0')
# %%

beta, gamma_p, D_p = init

D_ps = 10**np.linspace(-2.5, 0, 10)
D_ps = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
result_D = pd.DataFrame(index=D_ps, columns=['Max', 'Inflexion'])
figure()
for D_p in D_ps:
    res = get_similarity(eta, beta, gamma_p, D_p)
    result_D.loc[D_p, ['Max', 'Inflexion']] = get_max_infl(res)

    plt.plot(res.x, res.y[0] / np.max(res.y[0]), '-')
plt.xlim([0, 1])
figure()
for D_p in D_ps:
    res = get_similarity(eta, beta, gamma_p, D_p)
    result_D.loc[D_p, ['Max', 'Inflexion']] = get_max_infl(res)

    plt.semilogx(res.x, res.y[0] / np.max(res.y[0]), '-')

figure()
plt.xlabel('D_p')
plt.semilogx(result_D['Max'], 'x')
plt.semilogx(result_D['Inflexion'], 'x')
# plot(D_ps, D_ps* 0 + 1 , 'C1'


# %%
figure()
