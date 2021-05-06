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

max_f_eval = 50 ; N_SAA = 1
# max_f_eval = 25 ; N_SAA = 2
max_it = 100

#CUATRO local, CUATRO global, SQSnobFit, Bayes

N_samples = 20
RTONoise_list_SQSF = []
RTOConstraint_list_SQSF = []
for i in range(n_noise):
    print('Iteration ', i+1, ' of SQSnobfit')
    best = []
    best_constr = []
    for j in range(N_samples):
        f = lambda x: RTO_Noise(x, noise_mat[i], N_SAA)
        sol = SQSnobFitWrapper().solve(f, x0, bounds, mu_con = 1e6, \
                                    maxfun = max_f_eval, constraints=2)
        best.append(sol['f_best_so_far'][-1])
        _, g = RTO_Noise(sol['x_best_so_far'][-1], 0, N_SAA)
        best_constr.append(np.sum(np.maximum(g, 0)))
    RTONoise_list_SQSF.append(best)
    RTOConstraint_list_SQSF.append(best_constr)

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
        sol = CUATRO(f, x0, 2, bounds = bounds, max_f_eval = max_f_eval, \
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
init_radius = 10
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
    

with open('BayesRTO_listNoiseConv.pickle', 'rb') as handle:
    RTONoise_list_Bayes = pickle.load(handle)
    
with open('BayesRTO_listNoiseConstr.pickle', 'rb') as handle:
    RTOConstraint_list_Bayes = pickle.load(handle)


noise = ['%.3f' % noise_mat[i] for i in range(n_noise)]
noise_labels = [[noise[i]]*N_samples for i in range(n_noise)]


convergence = list(itertools.chain(*RTONoise_list_SQSF)) + \
              list(itertools.chain(*RTONoise_list_CUATROl)) + \
              list(itertools.chain(*RTONoise_list_CUATROg)) + \
              list(itertools.chain(*RTONoise_list_Bayes))    
              
constraints = list(itertools.chain(*RTOConstraint_list_SQSF)) + \
              list(itertools.chain(*RTOConstraint_list_CUATROl)) + \
              list(itertools.chain(*RTOConstraint_list_CUATROg)) + \
              list(itertools.chain(*RTOConstraint_list_Bayes))   
              
noise = list(itertools.chain(*noise_labels))*4
method = ['Snobfit']*int(len(noise)/4) + ['CUATRO_l']*int(len(noise)/4) + \
         ['CUATRO_g']*int(len(noise)/4) + ['Bayes. Opt.']*int(len(noise)/4)

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
plt.savefig('Publication plots/RTO_feval50Convergence.svg', format = "svg")
plt.show()
# ax.set_ylim([0.1, 10])
# ax.set_yscale("log")
plt.clf()



min_list = np.array([np.min([np.min(RTONoise_list_SQSF[i]), 
                  np.min(RTONoise_list_CUATROl[i]),
                  np.min(RTONoise_list_CUATROg[i]), 
                  np.min(RTONoise_list_Bayes[i])]) for i in range(n_noise)])

convergence_test = list(itertools.chain(*np.array(RTONoise_list_SQSF) - min_list.reshape(6,1))) + \
              list(itertools.chain(*np.array(RTONoise_list_CUATROl) - min_list.reshape(6,1))) + \
              list(itertools.chain(*np.array(RTONoise_list_CUATROg) - min_list.reshape(6,1))) + \
              list(itertools.chain(*np.array(RTONoise_list_Bayes) - min_list.reshape(6,1)))    
    

data_test = {'Best function evaluation': convergence_test, \
             "Constraint violation": constraints, \
             "Noise standard deviation": noise, \
             'Method': method}

df_test = pd.DataFrame(data_test)
    
ax = sns.boxplot(x = "Noise standard deviation", y = 'Best function evaluation', hue = "Method", data = df_test, palette = "muted")
# plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
# plt.legend([])
plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=4)
plt.tight_layout()
plt.ylabel(r'$f_{best, sample}$ - $f_{opt, noise}$')
plt.savefig('Publication plots/RTO_feval50ConvergenceLabel.svg', format = "svg")
plt.show()
plt.clf()

ax = sns.boxplot(x = "Noise standard deviation", y = "Constraint violation", \
                    hue = "Method", data = df, palette = "muted", fliersize = 0)
ax = sns.stripplot(x = "Noise standard deviation", y = "Constraint violation", \
                    hue = "Method", data = df, palette = "muted", dodge = True)
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.tight_layout()
plt.savefig('Publication plots/RTO_feval50Constraints.svg', format = "svg")
plt.show()
plt.clf()

max_f_eval = 25 ; N_SAA = 2
max_it = 100

#CUATRO local, CUATRO global, SQSnobFit, Bayes

N_samples = 20
RTOSAANoise_list_SQSF = []
RTOSAAConstraint_list_SQSF = []
for i in range(n_noise):
    print('Iteration ', i+1, ' of SQSnobfit')
    best = []
    best_constr = []
    for j in range(N_samples):
        f = lambda x: RTO_Noise(x, noise_mat[i], N_SAA)
        sol = SQSnobFitWrapper().solve(f, x0, bounds, mu_con = 1e6, \
                                    maxfun = max_f_eval, constraints=2)
        best.append(sol['f_best_so_far'][-1])
        _, g = RTO_Noise(sol['x_best_so_far'][-1], 0, N_SAA)
        best_constr.append(np.sum(np.maximum(g, 0)))
    RTOSAANoise_list_SQSF.append(best)
    RTOSAAConstraint_list_SQSF.append(best_constr)

# N_SAA = 1
N_samples = 20
RTOSAANoise_list_CUATROl = []
RTOSAAConstraint_list_CUATROl = []
for i in range(n_noise):
    print('Iteration ', i+1, ' of CUATRO_l')
    best = []
    best_constr = []
    for j in range(N_samples):
        f = lambda x: RTO_Noise(x, noise_mat[i], N_SAA)
        sol = CUATRO(f, x0, 2, bounds = bounds, max_f_eval = max_f_eval, \
                          N_min_samples = 6, tolerance = 1e-10,\
                          beta_red = 0.9, rnd = j, method = 'local', \
                          constr_handling = 'Fitting')
        best.append(sol['f_best_so_far'][-1])
        _, g = RTO_Noise(sol['x_best_so_far'][-1], 0, N_SAA)
        best_constr.append(np.sum(np.maximum(g, 0)))
    RTOSAANoise_list_CUATROl.append(best)
    RTOSAAConstraint_list_CUATROl.append(best_constr)
    
    
# N_SAA = 1
N_samples = 20
RTOSAANoise_list_CUATROg = []
RTOSAAConstraint_list_CUATROg = []
init_radius = 10
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
    RTOSAANoise_list_CUATROg.append(best)
    RTOSAAConstraint_list_CUATROg.append(best_constr)
    

with open('BayesRTO_listNoiseConvSAA.pickle', 'rb') as handle:
    RTOSAANoise_list_Bayes = pickle.load(handle)
    
with open('BayesRTO_listNoiseConstrSAA.pickle', 'rb') as handle:
    RTOSAAConstraint_list_Bayes = pickle.load(handle)


noise = ['%.3f' % noise_mat[i] for i in range(n_noise)]
noise_labels = [[noise[i]]*N_samples for i in range(n_noise)]


convergence = list(itertools.chain(*RTOSAANoise_list_SQSF)) + \
              list(itertools.chain(*RTOSAANoise_list_CUATROl)) + \
              list(itertools.chain(*RTOSAANoise_list_CUATROg)) + \
              list(itertools.chain(*RTOSAANoise_list_Bayes))    
              
constraints = list(itertools.chain(*RTOSAAConstraint_list_SQSF)) + \
              list(itertools.chain(*RTOSAAConstraint_list_CUATROl)) + \
              list(itertools.chain(*RTOSAAConstraint_list_CUATROg)) + \
              list(itertools.chain(*RTOSAAConstraint_list_Bayes))   
              
noise = list(itertools.chain(*noise_labels))*4
method = ['Snobfit']*int(len(noise)/4) + ['CUATRO_l']*int(len(noise)/4) + \
         ['CUATRO_g']*int(len(noise)/4) + ['Bayes. Opt.']*int(len(noise)/4)

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
plt.savefig('Publication plots/RTO_SAA2feval25Convergence.svg', format = "svg")
plt.show()
# ax.set_ylim([0.1, 10])
# ax.set_yscale("log")
plt.clf()



min_list = np.array([np.min([np.min(RTOSAANoise_list_SQSF[i]), 
                  np.min(RTOSAANoise_list_CUATROl[i]),
                  np.min(RTOSAANoise_list_CUATROg[i]), 
                  np.min(RTOSAANoise_list_Bayes[i])]) for i in range(n_noise)])

convergence_test = list(itertools.chain(*np.array(RTOSAANoise_list_SQSF) - min_list.reshape(6,1))) + \
              list(itertools.chain(*np.array(RTOSAANoise_list_CUATROl) - min_list.reshape(6,1))) + \
              list(itertools.chain(*np.array(RTOSAANoise_list_CUATROg) - min_list.reshape(6,1))) + \
              list(itertools.chain(*np.array(RTOSAANoise_list_Bayes) - min_list.reshape(6,1)))    
    

data_test = {'Best function evaluation': convergence_test, \
             "Constraint violation": constraints, \
             "Noise standard deviation": noise, \
             'Method': method}

df_test = pd.DataFrame(data_test)
    
ax = sns.boxplot(x = "Noise standard deviation", y = 'Best function evaluation', hue = "Method", data = df_test, palette = "muted")
# plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
# plt.legend([])
plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=4)
plt.tight_layout()
plt.ylabel(r'$f_{best, sample}$ - $f_{opt, noise}$')
plt.savefig('Publication plots/RTO_SAA2feval25ConvergenceLabel.svg', format = "svg")
plt.show()
plt.clf()

ax = sns.boxplot(x = "Noise standard deviation", y = "Constraint violation", \
                    hue = "Method", data = df, palette = "muted", fliersize = 0)
ax = sns.stripplot(x = "Noise standard deviation", y = "Constraint violation", \
                    hue = "Method", data = df, palette = "muted", dodge = True)
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.tight_layout()
plt.savefig('Publication plots/RTO_SAA2feval25Constraints.svg', format = "svg")
plt.show()
plt.clf()











