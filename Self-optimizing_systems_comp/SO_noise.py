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

from case_studies.self_optimization import systems

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

def self_opt_react(x, noise_level, N_SAA):
    plant = systems.Static_PDE_reaction_system()
    f = lambda x: plant.objective(x, noise_level[0])
    g = lambda x: plant.constraint_agg_1(x, noise_level[1])
    f_SAA = 0 ; g_SAA = -np.inf
    for i in range(N_SAA):
        f_SAA += f(x)/N_SAA
        g_SAA = np.maximum(g_SAA, g(x))
    
    return f_SAA, [g_SAA]

x0 = np.array([0.0,0.85])
bounds = np.array([[0., 1.], [0., 1.]])

noise_init = [0.001, 0.01]

n_noise = 6
noise_mat = np.zeros((n_noise, 2))
for i in range(n_noise):
    noise_mat[i] = 1/3*i*np.array(noise_init)
    
    

max_f_eval = 50 ; N_SAA = 1
# max_f_eval = 25 ; N_SAA = 2
max_it = 100

#CUATRO local, CUATRO global, SQSnobFit, Bayes

N_samples = 20
# SONoise_list_SQSF = []
# SOConstraint_list_SQSF = []
# for i in range(n_noise):
#     print('Iteration ', i+1, ' of SQSnobfit')
#     best = []
#     best_constr = []
#     f = lambda x: self_opt_react(x, noise_mat[i], N_SAA)
#     for j in range(N_samples):
#         sol = SQSnobFitWrapper().solve(f, x0, bounds, mu_con = 1e3, \
#                                     maxfun = max_f_eval, constraints=2)
#         best.append(sol['f_best_so_far'][-1])
#         _, g = self_opt_react(sol['x_best_so_far'][-1], [0, 0], 1)
#         best_constr.append(np.sum(np.maximum(g, 0)))
#     SONoise_list_SQSF.append(best)
#     SOConstraint_list_SQSF.append(best_constr)

# with open('SQSFSO_listNoiseConv.pickle', 'wb') as handle:
#     pickle.dump(SONoise_list_SQSF, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# with open('SQSFSO_listNoiseConstr.pickle', 'wb') as handle:
#     pickle.dump(SOConstraint_list_SQSF, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('SQSFSO_listNoiseConv.pickle', 'rb') as handle:
    SONoise_list_SQSF = pickle.load(handle)
    
with open('SQSFSO_listNoiseConstr.pickle', 'rb') as handle:
    SOConstraint_list_SQSF = pickle.load(handle)

# # N_SAA = 1
# N_samples = 20
# SONoise_list_DIRECT = []
# SOConstraint_list_DIRECT = []
# for i in range(n_noise):
#     print('Iteration ', i+1, ' of CUATRO_l')
#     best = []
#     best_constr = []
#     f_DIR = lambda x, grad: self_opt_react(x, noise_mat[i], N_SAA)
#     for j in range(N_samples):

#         sol = DIRECTWrapper().solve(f_DIR, x0, bounds, mu_con = 1e3, \
#                                     maxfun = max_f_eval, constraints=1)
#         best.append(sol['f_best_so_far'][-1])
#         _, g = self_opt_react(sol['x_best_so_far'][-1], [0, 0], 1)
#         best_constr.append(np.sum(np.maximum(g, 0)))
#     SONoise_list_DIRECT.append(best)
#     SOConstraint_list_DIRECT.append(best_constr)
    
# with open('DIRECTSO_listNoiseConv.pickle', 'wb') as handle:
#     pickle.dump(SONoise_list_DIRECT, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# with open('DIRECTSO_listNoiseConstr.pickle', 'wb') as handle:
#     pickle.dump(SOConstraint_list_DIRECT, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
with open('DIRECTSO_listNoiseConv.pickle', 'rb') as handle:
    SONoise_list_DIRECT = pickle.load(handle)
    
with open('DIRECTSO_listNoiseConstr.pickle', 'rb') as handle:
    SOConstraint_list_DIRECT = pickle.load(handle)
    
# N_SAA = 1
# N_samples = 20
# SONoise_list_CUATROg = []
# SOConstraint_list_CUATROg = []
# init_radius = 0.5
# for i in range(n_noise):
#     print('Iteration ', i+1, ' of CUATRO_g')
#     best = []
#     best_constr = []
#     f = lambda x: self_opt_react(x, noise_mat[i], N_SAA)
#     for j in range(N_samples):
#         sol = CUATRO(f, x0, init_radius, bounds = bounds, max_f_eval = max_f_eval, \
#                           N_min_samples = 15, tolerance = 1e-10,\
#                           beta_red = 0.9, rnd = j, method = 'global', \
#                           constr_handling = 'Discrimination')
#         best.append(sol['f_best_so_far'][-1])
#         _, g = self_opt_react(sol['x_best_so_far'][-1], [0, 0], 1)
#         best_constr.append(np.sum(np.maximum(g, 0)))
#     SONoise_list_CUATROg.append(best)
#     SOConstraint_list_CUATROg.append(best_constr)
    
# with open('CUATROgSO_listNoiseConv.pickle', 'wb') as handle:
#     pickle.dump(SONoise_list_CUATROg, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# with open('CUATROgSO_listNoiseConstr.pickle', 'wb') as handle:
#     pickle.dump(SOConstraint_list_CUATROg, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('CUATROgSO_listNoiseConv.pickle', 'rb') as handle:
    SONoise_list_CUATROg = pickle.load(handle)
    
with open('CUATROgSO_listNoiseConstr.pickle', 'rb') as handle:
    SOConstraint_list_CUATROg = pickle.load(handle)
    
# N_samples = 20
# SONoise_list_Bayes = []
# SOConstraint_list_Bayes = []
# for i in range(n_noise):
#     print('Iteration ', i+1, ' of SQSnobfit')
#     best = []
#     best_constr = []
#     f = lambda x: self_opt_react(x, noise_mat[i], N_SAA)
#     for j in range(N_samples):
#         Bayes = BayesOpt()
#         pyro.set_rng_seed(j)
#         sol = Bayes.solve(f, x0, acquisition='EI',bounds=bounds.T, \
#                             print_iteration = True, constraints=1, casadi=True, \
#                             maxfun = min(max_f_eval, 30)).output_dict
#         best.append(sol['f_best_so_far'][-1])
#         _, g = self_opt_react(sol['x_best_so_far'][-1], [0, 0], 1)
#         best_constr.append(np.sum(np.maximum(g, 0)))
#     SONoise_list_Bayes.append(best)
#     SOConstraint_list_Bayes.append(best_constr)

# with open('BayesSO_listNoiseConv.pickle', 'wb') as handle:
#     pickle.dump(SONoise_list_Bayes, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# with open('BayesSO_listNoiseConstr.pickle', 'wb') as handle:
#     pickle.dump(SOConstraint_list_Bayes, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('BayesSO_listNoiseConv.pickle', 'rb') as handle:
    SONoise_list_Bayes = pickle.load(handle)
    
with open('BayesSO_listNoiseConstr.pickle', 'rb') as handle:
    SOConstraint_list_Bayes = pickle.load(handle)

noise = ['%.5f' % noise_mat[i][0] for i in range(n_noise)]
noise_labels = [[noise[i]]*N_samples for i in range(n_noise)]


convergence = list(itertools.chain(*SONoise_list_SQSF)) + \
              list(itertools.chain(*SONoise_list_DIRECT)) + \
              list(itertools.chain(*SONoise_list_CUATROg)) + \
              list(itertools.chain(*SONoise_list_Bayes))    
              
constraints = list(itertools.chain(*SOConstraint_list_SQSF)) + \
              list(itertools.chain(*SOConstraint_list_DIRECT)) + \
              list(itertools.chain(*SOConstraint_list_CUATROg)) + \
              list(itertools.chain(*SOConstraint_list_Bayes))   
              
noise = list(itertools.chain(*noise_labels))*4
method = ['SQSnobfit']*int(len(noise)/4) + ['Direct']*int(len(noise)/4) + \
         ['CUATRO_g']*int(len(noise)/4) + ['Bayes Opt.']*int(len(noise)/4)

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
plt.savefig('Publication plots/SO_feval50Convergence.svg', format = "svg")
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
plt.savefig('Publication plots/SO_feval50ConvergenceLabel.svg', format = "svg")
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
plt.savefig('Publication plots/SO_feval50Constraints.svg', format = "svg")
plt.show()
plt.clf()


# max_f_eval = 50 ; N_SAA = 1
max_f_eval = 25 ; N_SAA = 2
max_it = 100

#CUATRO local, CUATRO global, SQSnobFit, Bayes

# N_samples = 20
# SOSAANoise_list_SQSF = []
# SOSAAConstraint_list_SQSF = []
# for i in range(n_noise):
#     print('Iteration ', i+1, ' of SQSnobfit')
#     best = []
#     best_constr = []
#     f = lambda x: self_opt_react(x, noise_mat[i], N_SAA)
#     for j in range(N_samples):
#         sol = SQSnobFitWrapper().solve(f, x0, bounds, mu_con = 1e3, \
#                                     maxfun = max_f_eval, constraints=2)
#         best.append(sol['f_best_so_far'][-1])
#         _, g = self_opt_react(sol['x_best_so_far'][-1], [0, 0], 1)
#         best_constr.append(np.sum(np.maximum(g, 0)))
#     SOSAANoise_list_SQSF.append(best)
#     SOSAAConstraint_list_SQSF.append(best_constr)

# with open('SQSFSOSAA_listNoiseConv.pickle', 'wb') as handle:
#     pickle.dump(SOSAANoise_list_SQSF, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# with open('SQSFSOSAA_listNoiseConstr.pickle', 'wb') as handle:
#     pickle.dump(SOSAAConstraint_list_SQSF, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('SQSFSOSAA_listNoiseConv.pickle', 'rb') as handle:
    SOSAANoise_list_SQSF = pickle.load(handle)
    
with open('SQSFSOSAA_listNoiseConstr.pickle', 'rb') as handle:
    SOSAAConstraint_list_SQSF = pickle.load(handle)

# # N_SAA = 1
# N_samples = 20
# SOSAANoise_list_DIRECT = []
# SOSAAConstraint_list_DIRECT = []
# for i in range(n_noise):
#     print('Iteration ', i+1, ' of CUATRO_l')
#     best = []
#     best_constr = []
#     f_DIR = lambda x, grad: self_opt_react(x, noise_mat[i], N_SAA)
#     for j in range(N_samples):

#         sol = DIRECTWrapper().solve(f_DIR, x0, bounds, mu_con = 1e3, \
#                                     maxfun = max_f_eval, constraints=1)
#         best.append(sol['f_best_so_far'][-1])
#         _, g = self_opt_react(sol['x_best_so_far'][-1], [0, 0], 1)
#         best_constr.append(np.sum(np.maximum(g, 0)))
#     SOSAANoise_list_DIRECT.append(best)
#     SOSAAConstraint_list_DIRECT.append(best_constr)
    
# with open('DIRECTSOSAA_listNoiseConv.pickle', 'wb') as handle:
#     pickle.dump(SOSAANoise_list_DIRECT, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# with open('DIRECTSOSAA_listNoiseConstr.pickle', 'wb') as handle:
#     pickle.dump(SOSAAConstraint_list_DIRECT, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('DIRECTSOSAA_listNoiseConv.pickle', 'rb') as handle:
    SOSAANoise_list_DIRECT = pickle.load(handle)
    
with open('DIRECTSOSAA_listNoiseConstr.pickle', 'rb') as handle:
    SOSAAConstraint_list_DIRECT = pickle.load(handle)

# # N_SAA = 1
# N_samples = 20
# SOSAANoise_list_CUATROg = []
# SOSAAConstraint_list_CUATROg = []
# init_radius = 0.5
# for i in range(n_noise):
#     print('Iteration ', i+1, ' of CUATRO_g')
#     best = []
#     best_constr = []
#     f = lambda x: self_opt_react(x, noise_mat[i], N_SAA)
#     for j in range(N_samples):
#         sol = CUATRO(f, x0, init_radius, bounds = bounds, max_f_eval = max_f_eval, \
#                           N_min_samples = 15, tolerance = 1e-10,\
#                           beta_red = 0.9, rnd = j, method = 'global', \
#                           constr_handling = 'Discrimination')
#         best.append(sol['f_best_so_far'][-1])
#         _, g = self_opt_react(sol['x_best_so_far'][-1], [0, 0], 1)
#         best_constr.append(np.sum(np.maximum(g, 0)))
#     SOSAANoise_list_CUATROg.append(best)
#     SOSAAConstraint_list_CUATROg.append(best_constr)
    
# with open('CUATROgSOSAA_listNoiseConv.pickle', 'wb') as handle:
#     pickle.dump(SOSAANoise_list_CUATROg, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# with open('CUATROgSOSAA_listNoiseConstr.pickle', 'wb') as handle:
#     pickle.dump(SOSAAConstraint_list_CUATROg, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('CUATROgSOSAA_listNoiseConv.pickle', 'rb') as handle:
    SOSAANoise_list_CUATROg = pickle.load(handle)
    
with open('CUATROgSOSAA_listNoiseConstr.pickle', 'rb') as handle:
    SOSAAConstraint_list_CUATROg = pickle.load(handle)
    
# N_samples = 20
# SOSAANoise_list_Bayes = []
# SOSAAConstraint_list_Bayes = []
# for i in range(n_noise):
#     print('Iteration ', i+1, ' of SQSnobfit')
#     best = []
#     best_constr = []
#     f = lambda x: self_opt_react(x, noise_mat[i], N_SAA)
#     for j in range(N_samples):
#         Bayes = BayesOpt()
#         pyro.set_rng_seed(j)
#         sol = Bayes.solve(f, x0, acquisition='EI',bounds=bounds.T, \
#                             print_iteration = True, constraints=1, casadi=True, \
#                             maxfun = min(max_f_eval, 30)).output_dict
#         best.append(sol['f_best_so_far'][-1])
#         _, g = self_opt_react(sol['x_best_so_far'][-1], [0, 0], 1)
#         best_constr.append(np.sum(np.maximum(g, 0)))
#     SOSAANoise_list_Bayes.append(best)
#     SOSAAConstraint_list_Bayes.append(best_constr)

# with open('BayesSOSAA_listNoiseConv.pickle', 'wb') as handle:
#     pickle.dump(SOSAANoise_list_Bayes, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# with open('BayesSOSAA_listNoiseConstr.pickle', 'wb') as handle:
#     pickle.dump(SOSAAConstraint_list_Bayes, handle, protocol=pickle.HIGHEST_PROTOCOL)
 
with open('BayesSOSAA_listNoiseConv.pickle', 'rb') as handle:
    SOSAANoise_list_Bayes = pickle.load(handle)
    
with open('BayesSOSAA_listNoiseConstr.pickle', 'rb') as handle:
    SOSAAConstraint_list_Bayes = pickle.load(handle)   


noise = ['%.5f' % noise_mat[i][0] for i in range(n_noise)]
noise_labels = [[noise[i]]*N_samples for i in range(n_noise)]


convergence = list(itertools.chain(*SOSAANoise_list_SQSF)) + \
              list(itertools.chain(*SOSAANoise_list_DIRECT)) + \
              list(itertools.chain(*SOSAANoise_list_CUATROg)) + \
              list(itertools.chain(*SOSAANoise_list_Bayes))    
              
constraints = list(itertools.chain(*SOSAAConstraint_list_SQSF)) + \
              list(itertools.chain(*SOSAAConstraint_list_DIRECT)) + \
              list(itertools.chain(*SOSAAConstraint_list_CUATROg)) + \
              list(itertools.chain(*SOSAAConstraint_list_Bayes))   
              
noise = list(itertools.chain(*noise_labels))*4
method = ['SQSnobfit']*int(len(noise)/4) + ['Direct']*int(len(noise)/4) + \
         ['CUATRO_g']*int(len(noise)/4) + ['Bayes Opt.']*int(len(noise)/4)

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
plt.savefig('Publication plots/SO_SAA2feval25Convergence.svg', format = "svg")
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
plt.savefig('Publication plots/SO_SAA2feval25ConvergenceLabel.svg', format = "svg")
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
plt.savefig('Publication plots/SO_SAA2feval25Constraints.svg', format = "svg")
plt.show()
plt.clf()

