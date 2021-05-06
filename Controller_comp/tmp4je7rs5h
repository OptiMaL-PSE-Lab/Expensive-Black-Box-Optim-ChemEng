# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 23:28:55 2021

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

from case_studies.Controller_tuning.Control_system import reactor_phi_2st, reactor_phi_2stNS

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

def cost_control_noise(x, bounds_abs, noise, N_SAA, x0 = [.116, 368.489], \
                       N = 200, T = 20, NS = False):
    f_SAA = 0 ; g_SAA = -np.inf
    if not NS:
        f = lambda x: reactor_phi_2st(x, bounds_abs, noise, x0 = x0, N = N, \
                                    T = T, return_sys_resp = False)
    else:
        f = lambda x: reactor_phi_2stNS(x, noise, x0 = x0, N = N, \
                                    T = T, return_sys_resp = False)
    for i in range(N_SAA):
        f_sample = f(x)
        f_SAA += f_sample[0]/N_SAA
        g_SAA = np.maximum(g_SAA, float(f_sample[1][0]))
        
    return f_SAA, [g_SAA]

max_f_eval = 100 ; N_SAA = 1
# max_f_eval = 50 ; N_SAA = 2
max_it = 100

pi = [.8746, .0257, -1.43388, -0.00131, 0.00016, 55.8692, 0.7159, .0188, .00017]
pi_init = [.8746, .0257, -1.43388, -0.00131, 0.00016, 0, 0, 0, 0, 0]

bounds_abs = np.zeros((10, 2))

for i in range(5):
    if pi[i] > 0:
        bounds_abs[i] = [pi[i]/2, pi[i]*2]
        bounds_abs[i+5] = [-pi[i]*10, pi[i]*10]
    else:
        bounds_abs[i] = [pi[i]*2, pi[i]/2]
        bounds_abs[i+5] = [pi[i]*10, -pi[i]*10]
        
x0 = (np.array(pi_init) - bounds_abs[:,0]) / (bounds_abs[:,1]-bounds_abs[:,0]) 

noise_previous = [.001, 1]

n_noise = 6
noise_mat = np.zeros((n_noise, 2))
for i in range(n_noise):
    noise_mat[i] = 1/3*np.array(noise_previous)*i
    
bounds = np.array([[0, 1]]*10)
 
x0 = (np.array(pi_init) - bounds_abs[:,0]) / (bounds_abs[:,1]-bounds_abs[:,0])     
x0_abs = np.array(pi_init)



N_samples = 20
ContrSynNoise_list_DIRECT = []
for i in range(n_noise):
    print('Iteration ', i+1, ' of DIRECT')
    best = []
    best_constr = []
    f = lambda x: cost_control_noise(x, bounds_abs, noise_mat[i], N_SAA, \
                                    x0 = [.116, 368.489], N = 200, T = 20)
    ContrSynNoise_DIRECT_f = lambda x, grad: f(x)
    for j in range(N_samples):
        sol = DIRECTWrapper().solve(ContrSynNoise_DIRECT_f, x0, bounds, \
                                    maxfun = max_f_eval, constraints=1)
        best.append(sol['f_best_so_far'][-1])
    ContrSynNoise_list_DIRECT.append(best)

# N_SAA = 1
N_samples = 20
ContrSynNoise_list_CUATROg = []
for i in range(n_noise):
    print('Iteration ', i+1, ' of CUATRO_g')
    best = []
    best_constr = []
    f = lambda x: cost_control_noise(x, bounds_abs, noise_mat[i], N_SAA, \
                                    x0 = [.116, 368.489], N = 200, T = 20)
    for j in range(N_samples):
        sol = CUATRO(f, x0, 0.5, bounds = bounds, max_f_eval = max_f_eval, \
                          N_min_samples = 6, tolerance = 1e-10,\
                          beta_red = 0.9, rnd = j, method = 'global', \
                          constr_handling = 'Discrimination')
        best.append(sol['f_best_so_far'][-1])
    ContrSynNoise_list_CUATROg.append(best)
    
# N_samples = 20
# ContrSynNoise_list_SQSnobFit = []
# for i in range(n_noise):
#     print('Iteration ', i+1, ' of SQSnobfit')
#     best = []
#     best_constr = []
#     f = lambda x: cost_control_noise(x, bounds_abs, noise_mat[i], N_SAA, \
#                                     x0 = [.116, 368.489], N = 200, T = 20, NS = True)
#     for j in range(N_samples):
#         sol = SQSnobFitWrapper().solve(f, x0_abs, bounds_abs, \
#                                     maxfun = max_f_eval, constraints=1)
#         best.append(sol['f_best_so_far'][-1])
#     ContrSynNoise_list_SQSnobFit.append(best)
    
N_samples = 20
ContrSynNoise_list_simplex = []
for i in range(n_noise):
    print('Iteration ', i+1, ' of SQSnobfit')
    best = []
    best_constr = []
    f = lambda x: cost_control_noise(x, bounds_abs, noise_mat[i], N_SAA, \
                                    x0 = [.116, 368.489], N = 200, T = 20)
    for j in range(N_samples):
        sol = simplex_method(f, x0, bounds, max_iter = 50, \
                            constraints = 1, rnd_seed = i, mu_con = 1e6)
        best.append(sol['f_best_so_far'][-1])
    ContrSynNoise_list_simplex.append(best)
    

noise = ['%.3f' % noise_mat[i] for i in range(n_noise)]
noise_labels = [[noise[i]]*N_samples for i in range(n_noise)]


convergence = list(itertools.chain(*ContrSynNoise_list_DIRECT)) + \
              list(itertools.chain(*ContrSynNoise_list_CUATROg)) + \
              list(itertools.chain(*ContrSynNoise_list_simplex))
              
noise = list(itertools.chain(*noise_labels))*3
method = ['DIRECT']*int(len(noise)/3) + ['CUATRO_g']*int(len(noise)/3) + \
         ['Simplex']*int(len(noise)/3)

data = {'Best function evaluation': convergence, \
        "Noise standard deviation": noise, \
        'Method': method}
df = pd.DataFrame(data)


ax = sns.boxplot(x = "Noise standard deviation", y = "Best function evaluation", hue = "Method", data = df, palette = "muted")
plt.savefig('Publication plots/SAA2feval25Convergence.svg', format = "svg")
# ax.set_ylim([0.1, 10])
# ax.set_yscale("log")
plt.show()
plt.clf()






