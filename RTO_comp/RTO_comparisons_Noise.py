# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 23:45:55 2021

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
from algorithms.SQSnobfit_wrapped.Wrapper_for_SQSnobfit import SQSnobFitWrapper
from algorithms.DIRECT_wrapped.Wrapper_for_Direct import DIRECTWrapper

from case_studies.RTO.systems import *

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

def RTO_Noise(x, noise, N_SAA):
    
    plant = WO_system()
    
    f = plant.WO_obj_sys_ca
    g1 = plant.WO_con1_sys_ca
    g2 = plant.WO_con2_sys_ca
    f_SAA = 0
    g1_SAA, g2_SAA = - np.inf, - np.inf
    
    for i in range(N_SAA):
        f_SAA += (f(x) +  5e-1 * np.random.normal(0., noise))/N_SAA
        g1_SAA = max(g1_SAA, g1(x) +  5e-4 * np.random.normal(0., noise))
        g2_SAA = max(g2_SAA, g2(x) +  5e-4 * np.random.normal(0., noise))
    
    return f_SAA, [g1_SAA, g2_SAA]

n_noise = 6
noise_mat = np.zeros(n_noise)
for i in range(n_noise):
    noise_mat[i] = 1/3*i

x0 = [6.9, 83]
bounds  = np.array([[4., 7.], [70., 100.]])

# max_f_eval = 50 ; N_SAA = 1
max_f_eval = 25 ; N_SAA = 2
max_it = 100

#CUATRO local, CUATRO global, simplex, DIRECT

N_samples = 20
RTONoise_list_splx = []
RTOConstraint_list_splx = []
for i in range(n_noise):
    print('Iteration ', i+1, ' of Simplex')
    best = []
    best_constr = []
    for j in range(N_samples):
        f = lambda x: RTO_Noise(x, noise_mat[i], N_SAA)
        sol = simplex_method(f, x0, bounds, max_iter = max_it, \
                            constraints = 2, rnd_seed = j)
        best.append(sol['f_best_so_far'][-1])
        _, g = RTO_Noise(sol['x_best_so_far'][-1], 0, N_SAA)
        best_constr.append(np.sum(np.maximum(g, 0)))
    RTONoise_list_splx.append(best)
    RTOConstraint_list_splx.append(best_constr)

# N_SAA = 1
N_samples = 20
RTONoise_list_CUATROl = []
RTOConstraint_list_CUATROl = []
for i in range(n_noise):
    print('Iteration ', i+1, ' of CUATRO_l')
    best = []
    best_constr = []
    for j in range(N_samples):
        f = lambda x: RTO_Noise(x, noise_mat[i], N_SAA)
        sol = CUATRO(f, x0, 1, bounds = bounds, max_f_eval = max_f_eval, \
                          N_min_samples = 6, tolerance = 1e-10,\
                          beta_red = 0.9, rnd = j, method = 'local', \
                          constr_handling = 'Fitting')
        best.append(sol['f_best_so_far'][-1])
        _, g = RTO_Noise(sol['x_best_so_far'][-1], 0, N_SAA)
        best_constr.append(np.sum(np.maximum(g, 0)))
    RTONoise_list_CUATROl.append(best)
    RTOConstraint_list_CUATROl.append(best_constr)
    
    
# N_SAA = 1
N_samples = 20
RTONoise_list_CUATROg = []
RTOConstraint_list_CUATROg = []
init_radius = 2
for i in range(n_noise):
    print('Iteration ', i+1, ' of CUATRO_g')
    best = []
    best_constr = []
    for j in range(N_samples):
        f = lambda x: RTO_Noise(x, noise_mat[i], N_SAA)
        sol = CUATRO(f, x0, init_radius, bounds = bounds, max_f_eval = max_f_eval, \
                          N_min_samples = 15, tolerance = 1e-10,\
                          beta_red = 0.9, rnd = j, method = 'global', \
                          constr_handling = 'Discrimination')
        best.append(sol['f_best_so_far'][-1])
        _, g = RTO_Noise(sol['x_best_so_far'][-1], 0, N_SAA)
        best_constr.append(np.sum(np.maximum(g, 0)))
    RTONoise_list_CUATROg.append(best)
    RTOConstraint_list_CUATROg.append(best_constr)
    


noise = ['%.3f' % noise_mat[i] for i in range(n_noise)]
noise_labels = [[noise[i]]*N_samples for i in range(n_noise)]


convergence = list(itertools.chain(*RTONoise_list_splx)) + \
              list(itertools.chain(*RTONoise_list_CUATROl)) + \
              list(itertools.chain(*RTONoise_list_CUATROg))
              
constraints = list(itertools.chain(*RTOConstraint_list_splx)) + \
              list(itertools.chain(*RTOConstraint_list_CUATROl)) + \
              list(itertools.chain(*RTOConstraint_list_CUATROg))
              
noise = list(itertools.chain(*noise_labels))*3
method = ['Simplex']*int(len(noise)/3) + ['CUATRO_l']*int(len(noise)/3) + \
         ['CUATRO_g']*int(len(noise)/3)

data = {'Best function evaluation': convergence, \
        "Constraint violation": constraints, \
        "Noise standard deviation": noise, \
        'Method': method}
df = pd.DataFrame(data)


ax = sns.boxplot(x = "Noise standard deviation", y = "Best function evaluation", hue = "Method", data = df, palette = "muted")
plt.savefig('Publication plots/SAA2feval25Convergence.svg', format = "svg")
# ax.set_ylim([0.1, 10])
# ax.set_yscale("log")
plt.show()
plt.clf()

ax = sns.boxplot(x = "Noise standard deviation", y = "Constraint violation", \
                    hue = "Method", data = df, palette = "muted", fliersize = 0)
ax = sns.stripplot(x = "Noise standard deviation", y = "Constraint violation", \
                    hue = "Method", data = df, palette = "muted", dodge = True)
plt.savefig('Publication plots/SAA2feval25Constraints.svg', format = "svg")



