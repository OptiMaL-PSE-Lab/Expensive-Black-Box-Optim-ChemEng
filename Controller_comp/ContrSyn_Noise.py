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

max_f_eval = 100 ; N_SAA = 1
# max_f_eval = 50 ; N_SAA = 2

N_samples = 20
# ContrSynNoise_list_DIRECT = []
# for i in range(n_noise):
#     print('Iteration ', i+1, ' of DIRECT')
#     best = []
#     best_constr = []
#     f = lambda x: cost_control_noise(x, bounds_abs, noise_mat[i], N_SAA, \
#                                     x0 = [.116, 368.489], N = 200, T = 20)
#     ContrSynNoise_DIRECT_f = lambda x, grad: f(x)
#     for j in range(N_samples):
#         sol = DIRECTWrapper().solve(ContrSynNoise_DIRECT_f, x0, bounds, \
#                                     maxfun = max_f_eval, constraints=1)
#         best.append(sol['f_best_so_far'][-1])
#     ContrSynNoise_list_DIRECT.append(best)

# with open('DIRECTContrSyn_listNoiseConv.pickle', 'wb') as handle:
#     pickle.dump(ContrSynNoise_list_DIRECT, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('DIRECTContrSyn_listNoiseConv.pickle', 'rb') as handle:
    ContrSynNoise_list_DIRECT = pickle.load(handle)

# # N_SAA = 1
# N_samples = 20
# ContrSynNoise_list_CUATROg = []
# for i in range(n_noise):
#     print('Iteration ', i+1, ' of CUATRO_g')
#     best = []
#     best_constr = []
#     f = lambda x: cost_control_noise(x, bounds_abs, noise_mat[i], N_SAA, \
#                                     x0 = [.116, 368.489], N = 200, T = 20)
#     for j in range(N_samples):
#         print(j+1)
#         sol = CUATRO(f, x0, 0.5, bounds = bounds, max_f_eval = max_f_eval, \
#                           N_min_samples = 6, tolerance = 1e-10,\
#                           beta_red = 0.9, rnd = j, method = 'global', \
#                           constr_handling = 'Discrimination')
#         best.append(sol['f_best_so_far'][-1])
#     ContrSynNoise_list_CUATROg.append(best)
    
# with open('CUATROgContrSyn_listNoiseConv.pickle', 'wb') as handle:
#     pickle.dump(ContrSynNoise_list_CUATROg, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('CUATROgContrSyn_listNoiseConv.pickle', 'rb') as handle:
    ContrSynNoise_list_CUATROg = pickle.load(handle)
    
# N_samples = 20
# ContrSynNoise_list_simplex = []
# for i in range(n_noise):
#     print('Iteration ', i+1, ' of Simplex')
#     best = []
#     best_constr = []
#     f = lambda x: cost_control_noise(x, bounds_abs, noise_mat[i], N_SAA, \
#                                     x0 = [.116, 368.489], N = 200, T = 20)
#     for j in range(N_samples):
#         sol = simplex_method(f, x0, bounds, max_iter = 50, \
#                             constraints = 1, rnd_seed = j, mu_con = 1e6)
#         best.append(sol['f_best_so_far'][-1])
#     ContrSynNoise_list_simplex.append(best)
    
# with open('simplexContrSyn_listNoiseConv.pickle', 'wb') as handle:
#     pickle.dump(ContrSynNoise_list_simplex, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('simplexContrSyn_listNoiseConv.pickle', 'rb') as handle:
    ContrSynNoise_list_simplex = pickle.load(handle) 

noise = ['%.3f' % noise_mat[i][1] for i in range(n_noise)]
noise_labels = [[noise[i]]*N_samples for i in range(n_noise)]

min_list = np.array([np.min([np.min(ContrSynNoise_list_DIRECT[i]), 
                  np.min(ContrSynNoise_list_CUATROg[i]),
                  np.min(ContrSynNoise_list_simplex[i])]) for i in range(n_noise)])

convergence = list(itertools.chain(*np.array(ContrSynNoise_list_DIRECT) - min_list.reshape(6,1))) + \
              list(itertools.chain(*np.array(ContrSynNoise_list_CUATROg)- min_list.reshape(6,1))) + \
              list(itertools.chain(*np.array(ContrSynNoise_list_simplex) - min_list.reshape(6,1)))
              
noise = list(itertools.chain(*noise_labels))*3
method = ['DIRECT']*int(len(noise)/3) + ['CUATRO_g']*int(len(noise)/3) + \
         ['Simplex']*int(len(noise)/3)

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

ax = sns.boxplot(x = "Noise standard deviation", y = "Best function evaluation", hue = "Method", data = df, palette = "muted")
plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3)
plt.tight_layout()
plt.ylabel(r'$f_{best, sample}$ - $f_{opt, noise}$')
plt.savefig('ContrSyn_publication_plots/ContrSyn_feval100Convergence.svg', format = "svg")
# ax.set_ylim([0.1, 10])
# ax.set_yscale("log")
plt.show()
plt.clf()


# max_f_eval = 100 ; N_SAA = 1
max_f_eval = 50 ; N_SAA = 2

N_samples = 20
# ContrSynNoiseSAA_list_DIRECT = []
# for i in range(n_noise):
#     print('Iteration ', i+1, ' of DIRECT')
#     best = []
#     best_constr = []
#     f = lambda x: cost_control_noise(x, bounds_abs, noise_mat[i], N_SAA, \
#                                     x0 = [.116, 368.489], N = 200, T = 20)
#     ContrSynNoise_DIRECT_f = lambda x, grad: f(x)
#     for j in range(N_samples):
#         sol = DIRECTWrapper().solve(ContrSynNoise_DIRECT_f, x0, bounds, \
#                                     maxfun = max_f_eval, constraints=1)
#         best.append(sol['f_best_so_far'][-1])
#     ContrSynNoiseSAA_list_DIRECT.append(best)

# with open('DIRECTContrSynSAA_listNoiseConv.pickle', 'wb') as handle:
#     pickle.dump(ContrSynNoiseSAA_list_DIRECT, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('DIRECTContrSynSAA_listNoiseConv.pickle', 'rb') as handle:
    ContrSynNoiseSAA_list_DIRECT = pickle.load(handle)

# N_SAA = 1
N_samples = 20
# ContrSynNoiseSAA_list_CUATROg = []
# for i in range(4):
#     print('Iteration ', i+1, ' of CUATRO_g')
#     best = []
#     best_constr = []
#     f = lambda x: cost_control_noise(x, bounds_abs, noise_mat[i], N_SAA, \
#                                     x0 = [.116, 368.489], N = 200, T = 20)
#     for j in range(N_samples):
#         print(j+1)
#         sol = CUATRO(f, x0, 0.5, bounds = bounds, max_f_eval = max_f_eval, \
#                           N_min_samples = 6, tolerance = 1e-10,\
#                           beta_red = 0.9, rnd = j, method = 'global', \
#                           constr_handling = 'Discrimination')
#         best.append(sol['f_best_so_far'][-1])
#     ContrSynNoiseSAA_list_CUATROg.append(best)
    
# with open('CUATROgContrSynSAA_listNoiseConv.pickle', 'wb') as handle:
#     pickle.dump(ContrSynNoiseSAA_list_CUATROg, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('CUATROgContrSynSAA_listNoiseConv.pickle', 'rb') as handle:
    ContrSynNoiseSAA_list_CUATROg = pickle.load(handle)
    
N_samples = 20
# ContrSynNoiseSAA_list_simplex = []
# for i in range(n_noise):
#     print('Iteration ', i+1, ' of Simplex')
#     best = []
#     best_constr = []
#     f = lambda x: cost_control_noise(x, bounds_abs, noise_mat[i], N_SAA, \
#                                     x0 = [.116, 368.489], N = 200, T = 20)
#     for j in range(N_samples):
#         sol = simplex_method(f, x0, bounds, max_iter = 50, \
#                             constraints = 1, rnd_seed = j, mu_con = 1e6)
#         best.append(sol['f_best_so_far'][-1])
#     ContrSynNoiseSAA_list_simplex.append(best)
    
# with open('simplexContrSynSAA_listNoiseConv.pickle', 'wb') as handle:
#     pickle.dump(ContrSynNoiseSAA_list_simplex, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('simplexContrSynSAA_listNoiseConv.pickle', 'rb') as handle:
    ContrSynNoiseSAA_list_simplex = pickle.load(handle) 

min_list = np.zeros(n_noise)
for i in range(n_noise):
    if i <4:
        min_list[i] = np.array([np.min([np.min(ContrSynNoiseSAA_list_DIRECT[i]), 
                                np.min(ContrSynNoiseSAA_list_CUATROg[i]),
                                np.min(ContrSynNoiseSAA_list_simplex[i])])])
    else:
        min_list[i] = np.array([np.min([np.min(ContrSynNoiseSAA_list_DIRECT[i]), 
                                np.min(ContrSynNoiseSAA_list_simplex[i])])])

noise = ['%.3f' % noise_mat[i][1] for i in range(n_noise)]
noise_labels = [[noise[i]]*N_samples for i in range(n_noise)]


convergence = list(itertools.chain(*np.array(ContrSynNoiseSAA_list_DIRECT) - min_list.reshape(6,1))) + \
              list(itertools.chain(*np.array(ContrSynNoiseSAA_list_CUATROg[:4])- min_list.reshape(6,1)[:4])) + \
              list(itertools.chain(*np.array(ContrSynNoiseSAA_list_simplex) - min_list.reshape(6,1)))
              
noise = list(itertools.chain(*noise_labels)) + list(itertools.chain(*noise_labels[:4])) + \
        list(itertools.chain(*noise_labels))                                         
method = ['DIRECT']*N_samples*n_noise + ['CUATRO_g*']*N_samples*(n_noise-2) + \
         ['Simplex']*N_samples*n_noise

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

ax = sns.boxplot(x = "Noise standard deviation", y = "Best function evaluation", hue = "Method", data = df, palette = "muted")
plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3)
plt.tight_layout()
plt.ylabel(r'$f_{best, sample}$ - $f_{opt, noise}$')
plt.savefig('ContrSyn_publication_plots/ContrSyn_SAA2feval50Convergence.svg', format = "svg")
# ax.set_ylim([0.1, 10])
# ax.set_yscale("log")
plt.show()
plt.clf()










