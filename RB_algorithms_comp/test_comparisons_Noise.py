# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 01:29:15 2021

@author: dv516
"""


import numpy as np
import pickle

import pyro
pyro.enable_validation(True)  # can help with debugging
pyro.set_rng_seed(1)

from algorithms.PyBobyqa_wrapped.Wrapper_for_pybobyqa import PyBobyqaWrapper
from algorithms.Bayesian_opt_Pyro.utilities_full import BayesOpt
from algorithms.nesterov_random.nesterov_random import nesterov_random
from algorithms.simplex.simplex_method import simplex_method
from algorithms.CUATRO.CUATRO import CUATRO
from algorithms.Finite_differences.Finite_differences import finite_Diff_Newton
from algorithms.Finite_differences.Finite_differences import Adam_optimizer
from algorithms.Finite_differences.Finite_differences import BFGS_optimizer

from test_functions import rosenbrock_constrained, quadratic_constrained

import matplotlib.pyplot as plt
import seaborn as sns

import itertools
import pandas as pd
import pickle

def average_from_list(solutions_list):
    N = len(solutions_list)
    f_best_all = np.zeros((N, 100))
    for i in range(N):
        f_best = np.array(solutions_list[i]['f_best_so_far'])
        x_ind = np.array(solutions_list[i]['samples_at_iteration'])
        for j in range(100):
            ind = np.where(x_ind <= j+1)
            if len(ind[0]) == 0:
                f_best_all[i, j] = f_best[0]
            else:
                f_best_all[i, j] = f_best[ind][-1]
    f_median = np.median(f_best_all, axis = 0)
    # f_av = np.average(f_best_all, axis = 0)
    # f_std = np.std(f_best_all, axis = 0)
    f_min = np.min(f_best_all, axis = 0)
    f_max = np.max(f_best_all, axis = 0)
    return f_best_all, f_median, f_min, f_max

def Problem_rosenbrock(x, noise_std, N_SAA):
    f_SAA = 0
    g_SAA1, g_SAA2 = - np.inf, -np.inf
    f1 = rosenbrock_constrained.rosenbrock_f
    g1 = rosenbrock_constrained.rosenbrock_g1
    g2 = rosenbrock_constrained.rosenbrock_g2
    for i in range(N_SAA):
        f_SAA += (f1(x) + np.random.normal(0, noise_std[0]))/N_SAA
        g_SAA1 = max(g1(x) + np.random.normal(0, noise_std[1]), g_SAA1)
        g_SAA2 = max(g2(x) + np.random.normal(0, noise_std[2]), g_SAA2)

    return f_SAA, [g_SAA1, g_SAA2]

n_noise = 6
noise_matrix = np.zeros((n_noise, 3))
for i in range(n_noise):
    noise_matrix[i] = np.array([0.05/3, 0.02/3, 0.02/3])*i

bounds = np.array([[-1.5,1.5],[-1.5,1.5]])
x0 = np.array([-0.5,1.5])

max_f_eval = 100
max_it = 50

N_SAA = 10


N_samples = 20
RBnoise_list_pybbqa = []
for i in range(n_noise):
    print('Iteration ', i, ' of Py-BOBYQA')
    best = []
    for j in range(N_samples):
        f = lambda x: Problem_rosenbrock(x, noise_matrix[i], N_SAA)
        sol = PyBobyqaWrapper().solve(f, x0, bounds=bounds.T, \
                                      maxfun= max_f_eval, constraints=2, \
                                      seek_global_minimum = True)
        best.append(sol['f_best_so_far'][-1])
    RBnoise_list_pybbqa.append(best)

# N_SAA = 1
N_samples = 20
RBnoise_list_splx = []
for i in range(n_noise):
    print('Iteration ', i, ' of Simplex')
    best = []
    for j in range(N_samples):
        f = lambda x: Problem_rosenbrock(x, noise_matrix[i], N_SAA)
        sol = simplex_method(f, x0, bounds, max_iter = 50, \
                            constraints = 2, rnd_seed = j)
        best.append(sol['f_best_so_far'][-1])
    RBnoise_list_splx.append(best)
    
# N_SAA = 1
N_samples = 20
RBnoise_list_CUATROl = []
init_radius = 0.1
for i in range(n_noise):
    print('Iteration ', i, ' of CUATRO_l')
    best = []
    for j in range(N_samples):
        f = lambda x: Problem_rosenbrock(x, noise_matrix[i], N_SAA)
        sol = CUATRO(f, x0, init_radius, bounds = bounds, \
                          N_min_samples = 6, tolerance = 1e-10,\
                          beta_red = 0.9, rnd = j, method = 'local', \
                          constr_handling = 'Fitting')
        best.append(sol['f_best_so_far'][-1])
    RBnoise_list_CUATROl.append(best)
    
# N_SAA = 1
N_samples = 20
RBnoise_list_CUATROg = []
init_radius = 2
for i in range(n_noise):
    print('Iteration ', i, ' of CUATRO_g')
    best = []
    for j in range(N_samples):
        f = lambda x: Problem_rosenbrock(x, noise_matrix[i], N_SAA)
        sol = CUATRO(f, x0, init_radius, bounds = bounds, \
                          N_min_samples = 15, tolerance = 1e-10,\
                          beta_red = 0.9, rnd = j, method = 'global', \
                          constr_handling = 'Discrimination')
        best.append(sol['f_best_so_far'][-1])
    RBnoise_list_CUATROg.append(best)
    


noise = ['%.3f' % noise_matrix[i][0] for i in range(n_noise)]
noise_labels = [[noise[i]]*N_samples for i in range(n_noise)]


convergence = list(itertools.chain(*RBnoise_list_pybbqa)) + \
              list(itertools.chain(*RBnoise_list_splx)) + \
              list(itertools.chain(*RBnoise_list_CUATROl)) + \
              list(itertools.chain(*RBnoise_list_CUATROg))
              
noise = list(itertools.chain(*noise_labels))*4
method = ['Py-BOBYQA']*int(len(noise)/4) + ['Simplex']*int(len(noise)/4) + \
         ['CUATRO_l']*int(len(noise)/4) + ['CUATRO_g']*int(len(noise)/4)

data = {'Convergence': convergence, 'Noise': noise, 'Method': method}
df = pd.DataFrame(data)

# ax = sns.violinplot(x = "Noise", y = "Convergence", hue = "Method", data = df, palette = "muted")
ax = sns.boxplot(x = "Noise", y = "Convergence", hue = "Method", data = df, palette = "muted")
# ax.set_ylim([0.1, 10])
ax.set_yscale("log")



convergence = list(itertools.chain(*RBnoise_list_pybbqa)) + \
              list(itertools.chain(*RBnoise_list_CUATROl)) + \
              list(itertools.chain(*RBnoise_list_CUATROg))
              
noise = list(itertools.chain(*noise_labels))*3
method = ['Py-BOBYQA']*int(len(noise)/3) +  \
         ['CUATRO_l']*int(len(noise)/3) + ['CUATRO_g']*int(len(noise)/3)

data = {'Convergence': convergence, 'Noise': noise, 'Method': method}
df = pd.DataFrame(data)

# ax = sns.violinplot(x = "Noise", y = "Convergence", hue = "Method", data = df, palette = "muted")
ax = sns.boxplot(x = "Noise", y = "Convergence", hue = "Method", data = df, palette = "muted")
# ax.set_ylim([0.1, 10])
ax.set_yscale("log")


convergence = list(itertools.chain(*RBnoise_list_CUATROl)) + \
              list(itertools.chain(*RBnoise_list_CUATROg))
              
noise = list(itertools.chain(*noise_labels))*2
method = ['CUATRO_l']*int(len(noise)/2) + ['CUATRO_g']*int(len(noise)/2)

data = {'Convergence': convergence, 'Noise': noise, 'Method': method}
df = pd.DataFrame(data)

# ax = sns.violinplot(x = "Noise", y = "Convergence", hue = "Method", data = df, palette = "muted")
ax = sns.boxplot(x = "Noise", y = "Convergence", hue = "Method", data = df, palette = "muted")
# ax.set_ylim([0, 4])
# ax.set_yscale("log")
# ax = sns.violinplot(x = "Noise", y = "Convergence", hue = "Method", data = df, palette = "muted")
# ax = sns.violinplot(x = "Noise", y = "Convergence", hue = "Method", data = df, palette = "muted")
# ax.set_ylim([0.75, 1.25])
# plt.boxplot(RBnoise_list_pybbqa.T)


convergence = list(itertools.chain(*RBnoise_list_CUATROg))
              
noise = list(itertools.chain(*noise_labels))
method = ['CUATRO_g']*int(len(noise))

data = {'Convergence': convergence, 'Noise': noise, 'Method': method}
df = pd.DataFrame(data)

# ax = sns.violinplot(x = "Noise", y = "Convergence", hue = "Method", data = df, palette = "muted")
# ax = sns.boxplot(x = "Noise", y = "Convergence", hue = "Method", data = df, palette = "muted")
# ax.set_ylim([0, 4])
# ax.set_yscale("log")
# ax = sns.violinplot(x = "Noise", y = "Convergence", hue = "Method", data = df, palette = "muted")
ax = sns.violinplot(x = "Noise", y = "Convergence", hue = "Method", data = df, palette = "muted")
# ax.set_ylim([0.75, 1.25])
# plt.boxplot(RBnoise_list_pybbqa.T)
# sbs.violinplot(RBnoise_list_pybbqa.T)


# ax = sns.boxplot(x = "Noise", y = "Convergence", hue = "Method", data = df, palette = "muted")




