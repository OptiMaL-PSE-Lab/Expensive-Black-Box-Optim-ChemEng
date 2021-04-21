# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 19:16:41 2021

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

from test_functions import quadratic_constrained

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

def Problem_quadraticNoise(x, noise_std, N_SAA):
    f_SAA = 0 ; g_SAA = - np.inf
    for i in range(N_SAA):
        f_sample = quadratic_constrained.quadratic_f(x)
        f_noise = np.random.normal(0, noise_std[0])
        f_SAA += (f_sample + f_noise)/N_SAA
        g_sample = quadratic_constrained.quadratic_g(x)
        g_noise = np.random.normal(0, noise_std[1])
        g_SAA = max(g_SAA, g_sample + g_noise)

    return f_SAA, [g_SAA]

n_noise = 6
noise_matrix = np.zeros((n_noise, 2))
for i in range(n_noise):
    noise_matrix[i] = np.array([0.05/3, 0.01/3])*i

bounds = np.array([[-1.5,1.5],[-1.5,1.5]])
x0 = np.array([1,1])

max_f_eval = 50 ; N_SAA = 1
# max_f_eval = 25 ; N_SAA = 2
max_it = 100

#CUATRO local, CUATRO global, BO, DIRECT

with open('BayesQuadratic_listNoiseConv.pickle', 'rb') as handle:
    quadraticNoise_list_Bayes = pickle.load(handle)
    
with open('BayesQuadratic_listNoiseConstr.pickle', 'rb') as handle:
    quadraticConstraint_list_Bayes = pickle.load(handle)


# N_SAA = 1
N_samples = 20
quadraticNoise_list_CUATROl = []
quadraticConstraint_list_CUATROl = []
for i in range(n_noise):
    print('Iteration ', i+1, ' of CUATRO_l')
    best = []
    best_constr = []
    for j in range(N_samples):
        f = lambda x: Problem_quadraticNoise(x, noise_matrix[i], N_SAA)
        sol = CUATRO(f, x0, 0.5, bounds = bounds, max_f_eval = max_f_eval, \
                          N_min_samples = 6, tolerance = 1e-10,\
                          beta_red = 0.9, rnd = j, method = 'local', \
                          constr_handling = 'Fitting')
        best.append(sol['f_best_so_far'][-1])
        _, g = Problem_quadraticNoise(sol['x_best_so_far'][-1], [0, 0, 0], N_SAA)
        best_constr.append(np.sum(np.maximum(g, 0)))
    quadraticNoise_list_CUATROl.append(best)
    quadraticConstraint_list_CUATROl.append(best_constr)
    
# N_SAA = 1
N_samples = 20
quadraticNoise_list_DIRECT = []
quadraticConstraint_list_DIRECT = []
init_radius = 0.1
boundsDIR = np.array([[-1.5,1],[-1,1.5]])
for i in range(n_noise):
    print('Iteration ', i+1, ' of DIRECT')
    best = []
    best_constr = []
    for j in range(N_samples):
        f = lambda x: Problem_quadraticNoise(x, noise_matrix[i], N_SAA)
        DIRECT_f = lambda x, grad: f(x)
        sol = DIRECTWrapper().solve(DIRECT_f, x0, bounds, \
                                    maxfun = max_f_eval, constraints=1)
        best.append(sol['f_best_so_far'][-1])
        _, g = Problem_quadraticNoise(sol['x_best_so_far'][-1], [0, 0, 0], N_SAA)
        best_constr.append(np.sum(np.maximum(g, 0)))
    quadraticNoise_list_DIRECT.append(best)
    quadraticConstraint_list_DIRECT.append(best_constr)
    
# N_SAA = 1
N_samples = 20
quadraticNoise_list_CUATROg = []
quadraticConstraint_list_CUATROg = []
init_radius = 2
for i in range(n_noise):
    print('Iteration ', i+1, ' of CUATRO_g')
    best = []
    best_constr = []
    for j in range(N_samples):
        f = lambda x: Problem_quadraticNoise(x, noise_matrix[i], N_SAA)
        sol = CUATRO(f, x0, init_radius, bounds = bounds, max_f_eval = max_f_eval, \
                          N_min_samples = 15, tolerance = 1e-10,\
                          beta_red = 0.9, rnd = j, method = 'global', \
                          constr_handling = 'Discrimination')
        best.append(sol['f_best_so_far'][-1])
        _, g = Problem_quadraticNoise(sol['x_best_so_far'][-1], [0, 0, 0], N_SAA)
        best_constr.append(np.sum(np.maximum(g, 0)))
    quadraticNoise_list_CUATROg.append(best)
    quadraticConstraint_list_CUATROg.append(best_constr)
    


noise = ['%.3f' % noise_matrix[i][0] for i in range(n_noise)]
noise_labels = [[noise[i]]*N_samples for i in range(n_noise)]


convergence = list(itertools.chain(*quadraticNoise_list_Bayes)) + \
              list(itertools.chain(*quadraticNoise_list_CUATROl)) + \
              list(itertools.chain(*quadraticNoise_list_DIRECT)) + \
              list(itertools.chain(*quadraticNoise_list_CUATROg))
              
constraints = list(itertools.chain(*quadraticConstraint_list_Bayes)) + \
              list(itertools.chain(*quadraticConstraint_list_CUATROl)) + \
              list(itertools.chain(*quadraticConstraint_list_DIRECT)) + \
              list(itertools.chain(*quadraticConstraint_list_CUATROg))
              
noise = list(itertools.chain(*noise_labels))*4
method = ['Bayes. Opt.']*int(len(noise)/4) + ['CUATRO_l']*int(len(noise)/4) + \
         ['DIRECT']*int(len(noise)/4) + ['CUATRO_g']*int(len(noise)/4)

data = {'Best function evaluation': convergence, \
        "Constraint violation": constraints, \
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
# plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.legend([])
# plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
#                 mode="expand", borderaxespad=0, ncol=4)
plt.tight_layout()
plt.savefig('Quadratic_publication_plots/Quadratic_feval50Convergence.svg', format = "svg")
plt.show()
# ax.set_ylim([0.1, 10])
# ax.set_yscale("log")
plt.clf()

ax = sns.boxplot(x = "Noise standard deviation", y = "Best function evaluation", hue = "Method", data = df, palette = "muted")
# plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
# plt.legend([])
plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=4)
plt.tight_layout()
plt.savefig('Quadratic_publication_plots/Quadratic_feval50ConvergenceLabel.svg', format = "svg")
plt.show()
# ax.set_ylim([0.1, 10])
# ax.set_yscale("log")
plt.clf()

ax = sns.boxplot(x = "Noise standard deviation", y = "Constraint violation", \
                    hue = "Method", data = df, palette = "muted", fliersize = 0)
ax = sns.stripplot(x = "Noise standard deviation", y = "Constraint violation", \
                    hue = "Method", data = df, palette = "muted", dodge = True)
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.tight_layout()
plt.savefig('Quadratic_publication_plots/Quadratic_feval50Constraints.svg', format = "svg")
plt.show()
plt.clf()



# max_f_eval = 50 ; N_SAA = 1
max_f_eval = 25 ; N_SAA = 2
max_it = 100

#CUATRO local, CUATRO global, BO, DIRECT

with open('BayesQuadrati_listNoiseConvSAA.pickle', 'rb') as handle:
    quadraticSAANoise_list_Bayes = pickle.load(handle)
    
with open('BayesQuadrati_listNoiseConstrSAA.pickle', 'rb') as handle:
    quadraticSAAConstraint_list_Bayes = pickle.load(handle)


# N_SAA = 1
N_samples = 20
quadraticSAANoise_list_CUATROl = []
quadraticSAAConstraint_list_CUATROl = []
for i in range(n_noise):
    print('Iteration ', i+1, ' of CUATRO_l')
    best = []
    best_constr = []
    for j in range(N_samples):
        f = lambda x: Problem_quadraticNoise(x, noise_matrix[i], N_SAA)
        sol = CUATRO(f, x0, 0.5, bounds = bounds, max_f_eval = max_f_eval, \
                          N_min_samples = 6, tolerance = 1e-10,\
                          beta_red = 0.9, rnd = j, method = 'local', \
                          constr_handling = 'Fitting')
        best.append(sol['f_best_so_far'][-1])
        _, g = Problem_quadraticNoise(sol['x_best_so_far'][-1], [0, 0, 0], N_SAA)
        best_constr.append(np.sum(np.maximum(g, 0)))
    quadraticSAANoise_list_CUATROl.append(best)
    quadraticSAAConstraint_list_CUATROl.append(best_constr)
    
# N_SAA = 1
N_samples = 20
quadraticSAANoise_list_DIRECT = []
quadraticSAAConstraint_list_DIRECT = []
for i in range(n_noise):
    print('Iteration ', i+1, ' of DIRECT')
    best = []
    best_constr = []
    for j in range(N_samples):
        f = lambda x: Problem_quadraticNoise(x, noise_matrix[i], N_SAA)
        DIRECT_f = lambda x, grad: f(x)
        sol = DIRECTWrapper().solve(DIRECT_f, x0, bounds, \
                                    maxfun = max_f_eval, constraints=1)
        best.append(sol['f_best_so_far'][-1])
        _, g = Problem_quadraticNoise(sol['x_best_so_far'][-1], [0, 0, 0], N_SAA)
        best_constr.append(np.sum(np.maximum(g, 0)))
    quadraticSAANoise_list_DIRECT.append(best)
    quadraticSAAConstraint_list_DIRECT.append(best_constr)
    
# N_SAA = 1
N_samples = 20
quadraticSAANoise_list_CUATROg = []
quadraticSAAConstraint_list_CUATROg = []
init_radius = 2
for i in range(n_noise):
    print('Iteration ', i+1, ' of CUATRO_g')
    best = []
    best_constr = []
    for j in range(N_samples):
        f = lambda x: Problem_quadraticNoise(x, noise_matrix[i], N_SAA)
        sol = CUATRO(f, x0, init_radius, bounds = bounds, max_f_eval = max_f_eval, \
                          N_min_samples = 15, tolerance = 1e-10,\
                          beta_red = 0.9, rnd = j, method = 'global', \
                          constr_handling = 'Discrimination')
        best.append(sol['f_best_so_far'][-1])
        _, g = Problem_quadraticNoise(sol['x_best_so_far'][-1], [0, 0, 0], N_SAA)
        best_constr.append(np.sum(np.maximum(g, 0)))
    quadraticSAANoise_list_CUATROg.append(best)
    quadraticSAAConstraint_list_CUATROg.append(best_constr)
    


noise = ['%.3f' % noise_matrix[i][0] for i in range(n_noise)]
noise_labels = [[noise[i]]*N_samples for i in range(n_noise)]


convergence = list(itertools.chain(*quadraticSAANoise_list_Bayes)) + \
              list(itertools.chain(*quadraticSAANoise_list_CUATROl)) + \
              list(itertools.chain(*quadraticSAANoise_list_DIRECT)) + \
              list(itertools.chain(*quadraticSAANoise_list_CUATROg))
              
constraints = list(itertools.chain(*quadraticSAAConstraint_list_Bayes)) + \
              list(itertools.chain(*quadraticSAAConstraint_list_CUATROl)) + \
              list(itertools.chain(*quadraticSAAConstraint_list_DIRECT)) + \
              list(itertools.chain(*quadraticSAAConstraint_list_CUATROg))
              
noise = list(itertools.chain(*noise_labels))*4
method = ['Bayes. Opt.']*int(len(noise)/4) + ['CUATRO_l']*int(len(noise)/4) + \
         ['DIRECT']*int(len(noise)/4) + ['CUATRO_g']*int(len(noise)/4)

data = {'Best function evaluation': convergence, \
        "Constraint violation": constraints, \
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
# plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.legend([])
# plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
#                 mode="expand", borderaxespad=0, ncol=4)
plt.tight_layout()
plt.savefig('Quadratic_publication_plots/Quadratic_SAA2feval25Convergence.svg', format = "svg")
plt.show()
# ax.set_ylim([0.1, 10])
# ax.set_yscale("log")
plt.clf()


ax = sns.boxplot(x = "Noise standard deviation", y = "Best function evaluation", hue = "Method", data = df, palette = "muted")
# plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
# plt.legend([])
plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=4)
plt.tight_layout()
plt.savefig('Quadratic_publication_plots/Quadratic_SAA2feval25ConvergenceLabel.svg', format = "svg")
plt.show()
# ax.set_ylim([0.1, 10])
# ax.set_yscale("log")
plt.clf()

ax = sns.boxplot(x = "Noise standard deviation", y = "Constraint violation", \
                    hue = "Method", data = df, palette = "muted", fliersize = 0)
ax = sns.stripplot(x = "Noise standard deviation", y = "Constraint violation", \
                    hue = "Method", data = df, palette = "muted", dodge = True)
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.tight_layout()
plt.savefig('Quadratic_publication_plots/Quadratic_SAA2feval25Constraints.svg', format = "svg")
plt.show()
plt.clf()


