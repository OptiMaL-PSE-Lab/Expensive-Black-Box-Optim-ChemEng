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

from case_studies.Controller_tuning.Control_system import phi, phi_rand

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

def phi_Noise(x, noise, N_SAA):
    
    f = lambda x: phi_rand(x, deviation = noise)
    f_SAA = 0
    
    for i in range(N_SAA):
        f_SAA += (f(x)[0])/N_SAA
    
    return f_SAA, [0]

n_noise = 6
noise_mat = np.zeros(n_noise)
for i in range(n_noise):
    noise_mat[i] = 1/3*i


x0 = np.array([4, 4, 4, 4])
bounds = np.array([[0, 8], [0, 8], [0, 8], [0, 8]])


max_f_eval = 100 ; N_SAA = 1

N_samples = 20
ContrLinNoise_list_DIRECT = []
for i in range(n_noise):
    print('Iteration ', i+1, ' of DIRECT')
    best = []
    best_constr = []
    for j in range(N_samples):
        f = lambda x: phi_Noise(x, noise_mat[i], N_SAA)
        ContrLinNoise_DIRECT_f = lambda x, grad: f(x)
        sol = DIRECTWrapper().solve(ContrLinNoise_DIRECT_f, x0, bounds, \
                                    maxfun = max_f_eval, constraints=1)
        best.append(sol['f_best_so_far'][-1])
    ContrLinNoise_list_DIRECT.append(best)

# N_SAA = 1
N_samples = 20
ContrLinNoise_list_CUATROl = []
for i in range(n_noise):
    print('Iteration ', i+1, ' of CUATRO_l')
    best = []
    best_constr = []
    for j in range(N_samples):
        f = lambda x: phi_Noise(x, noise_mat[i], N_SAA)
        sol = CUATRO(f, x0, 1, bounds = bounds, max_f_eval = max_f_eval, \
                          N_min_samples = 6, tolerance = 1e-10,\
                          beta_red = 0.9, rnd = j, method = 'local', \
                          constr_handling = 'Fitting')
        best.append(sol['f_best_so_far'][-1])
    ContrLinNoise_list_CUATROl.append(best)
    
# N_SAA = 1
N_samples = 20
ContrLinNoise_list_CUATROg = []
for i in range(n_noise):
    print('Iteration ', i+1, ' of CUATRO_g')
    best = []
    best_constr = []
    for j in range(N_samples):
        f = lambda x: phi_Noise(x, noise_mat[i], N_SAA)
        sol = CUATRO(f, x0, 4, bounds = bounds, max_f_eval = max_f_eval, \
                          N_min_samples = 15, tolerance = 1e-10,\
                          beta_red = 0.9, rnd = j, method = 'global', \
                          constr_handling = 'Discrimination')
        best.append(sol['f_best_so_far'][-1])
    ContrLinNoise_list_CUATROg.append(best)
    
    

noise = ['%.3f' % noise_mat[i] for i in range(n_noise)]
noise_labels = [[noise[i]]*N_samples for i in range(n_noise)]


convergence = list(itertools.chain(*ContrLinNoise_list_DIRECT)) + \
              list(itertools.chain(*ContrLinNoise_list_CUATROl)) + \
              list(itertools.chain(*ContrLinNoise_list_CUATROg))
              
noise = list(itertools.chain(*noise_labels))*3
method = ['DIRECT']*int(len(noise)/3) + ['CUATRO_l']*int(len(noise)/3) + \
         ['CUATRO_g']*int(len(noise)/3)

data = {'Best function evaluation': convergence, \
        "Noise standard deviation": noise, \
        'Method': method}
df = pd.DataFrame(data)


plt.rcParams["font.family"] = "Times New Roman"
ft = int(15)
font = {'size': ft}
plt.rc('font', **font)
params = {'legend.fontsize': 12.5,
              'legend.handlelength': 1.2}
plt.rcParams.update(params)


ax = sns.boxplot(x = "Noise standard deviation", y = "Best function evaluation", hue = "Method", data = df, palette = "muted")
plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=4)
plt.tight_layout()
plt.savefig('Controller_publication_plots/ContrLin_feval100Convergence.svg', format = "svg")
# ax.set_ylim([0.1, 10])
# ax
plt.show()
plt.clf()


max_f_eval = 50 ; N_SAA = 2
max_it = 100

#CUATRO local, CUATRO global, simplex, DIRECT

N_samples = 20
ContrLinNoiseSAA_list_DIRECT = []
for i in range(n_noise):
    print('Iteration ', i+1, ' of DIRECT')
    best = []
    best_constr = []
    for j in range(N_samples):
        f = lambda x: phi_Noise(x, noise_mat[i], N_SAA)
        ContrLinNoise_DIRECT_f = lambda x, grad: f(x)
        sol = DIRECTWrapper().solve(ContrLinNoise_DIRECT_f, x0, bounds, \
                                    maxfun = max_f_eval, constraints=1)
        best.append(sol['f_best_so_far'][-1])
    ContrLinNoiseSAA_list_DIRECT.append(best)

# N_SAA = 1
N_samples = 20
ContrLinNoiseSAA_list_CUATROl = []
for i in range(n_noise):
    print('Iteration ', i+1, ' of CUATRO_l')
    best = []
    best_constr = []
    for j in range(N_samples):
        f = lambda x: phi_Noise(x, noise_mat[i], N_SAA)
        sol = CUATRO(f, x0, 1, bounds = bounds, max_f_eval = max_f_eval, \
                          N_min_samples = 6, tolerance = 1e-10,\
                          beta_red = 0.9, rnd = j, method = 'local', \
                          constr_handling = 'Fitting')
        best.append(sol['f_best_so_far'][-1])
    ContrLinNoiseSAA_list_CUATROl.append(best)
    
# N_SAA = 1
N_samples = 20
ContrLinNoiseSAA_list_CUATROg = []
for i in range(n_noise):
    print('Iteration ', i+1, ' of Simplex')
    best = []
    best_constr = []
    for j in range(N_samples):
        f = lambda x: phi_Noise(x, noise_mat[i], N_SAA)
        sol = CUATRO(f, x0, 4, bounds = bounds, max_f_eval = max_f_eval, \
                          N_min_samples = 15, tolerance = 1e-10,\
                          beta_red = 0.9, rnd = j, method = 'global', \
                          constr_handling = 'Discrimination')
        best.append(sol['f_best_so_far'][-1])
    ContrLinNoiseSAA_list_CUATROg.append(best)
    
    
    


noise = ['%.3f' % noise_mat[i] for i in range(n_noise)]
noise_labels = [[noise[i]]*N_samples for i in range(n_noise)]


convergence = list(itertools.chain(*ContrLinNoiseSAA_list_DIRECT)) + \
              list(itertools.chain(*ContrLinNoiseSAA_list_CUATROl)) + \
              list(itertools.chain(*ContrLinNoiseSAA_list_CUATROg))
              
noise = list(itertools.chain(*noise_labels))*3
method = ['DIRECT']*int(len(noise)/3) + ['CUATRO_l']*int(len(noise)/3) + \
         ['CUATRO_g']*int(len(noise)/3)

data = {'Best function evaluation': convergence, \
        "Noise standard deviation": noise, \
        'Method': method}
df = pd.DataFrame(data)

plt.rcParams["font.family"] = "Times New Roman"
ft = int(15)
font = {'size': ft}
plt.rc('font', **font)
params = {'legend.fontsize': 12.5,
              'legend.handlelength': 1.2}
plt.rcParams.update(params)


ax = sns.boxplot(x = "Noise standard deviation", y = "Best function evaluation", hue = "Method", data = df, palette = "muted")
plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3)
plt.tight_layout()
plt.savefig('Controller_publication_plots/ContrLin_SAA2feval50Convergence.svg', format = "svg")
# ax.set_ylim([0.1, 10])
# ax.set_yscale("log")
plt.show()
plt.clf()
