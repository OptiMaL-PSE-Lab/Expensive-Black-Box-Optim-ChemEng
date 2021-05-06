# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 22:33:03 2021

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

def trust_fig(X, Y, Z, g1):   
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.contour(X, Y, Z, 50)
    ax.contour(X, Y, g1, levels = [0], colors = 'black')
    # ax.contour(X, Y, g2, levels = [0], colors = 'black')
    
    return ax, fig


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


def fix_starting_points(complete_list, x0, init_out, only_starting_point = False):
    if only_starting_point:
        for i in range(len(complete_list)):
            dict_out = complete_list[i]
            f_arr = dict_out['f_best_so_far']
            N_eval = len(f_arr)
            g_arr = dict_out['g_best_so_far']
            dict_out['x_best_so_far'][0] = np.array(x0)
            dict_out['f_best_so_far'][0] = init_out[0]
            dict_out['g_best_so_far'][0] = np.array(init_out[1])
            complete_list[i] = dict_out        
    else:
        for i in range(len(complete_list)):
            dict_out = complete_list[i]
            f_arr = dict_out['f_best_so_far']
            N_eval = len(f_arr)
            g_arr = dict_out['g_best_so_far']
            dict_out['x_best_so_far'][0] = np.array(x0)
            dict_out['f_best_so_far'][0] = init_out[0]
            dict_out['g_best_so_far'][0] = np.array(init_out[1])
        
            for j in range(1, N_eval):
                if (g_arr[j] > 1e-3).any() or (init_out[0] < f_arr[j]):
                    dict_out['x_best_so_far'][j] = np.array(x0)
                    dict_out['f_best_so_far'][j] = init_out[0]
                    dict_out['g_best_so_far'][j] = np.array(init_out[1])
            complete_list[i] = dict_out
            
    return complete_list


def medianx_from_list(solutions_list, x0):
    N = len(solutions_list)
    _, N_x = np.array(solutions_list[0]['x_best_so_far']).shape
    f_best_all = np.zeros((N, 100))
    x_best_all = np.zeros((N, 100, N_x))
    for i in range(N):
        f_best = np.array(solutions_list[i]['f_best_so_far'])
        x_best = np.array(solutions_list[i]['x_best_so_far'])
        x_ind = np.array(solutions_list[i]['samples_at_iteration'])
        for j in range(100):
            ind = np.where(x_ind <= j+1)
            if len(ind[0]) == 0:
                f_best_all[i, j] = f_best[0]
                x_best_all[i,j,:] = np.array(x0)
            else:
                f_best_all[i, j] = f_best[ind][-1]
                x_best_all[i,j,:] = np.array(x_best[ind][-1])
    x_best_all
    x_median = np.median(x_best_all, axis = 0)

    return  x_median

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

noise_init = [0.001, 0.01] ; N_SAA  = 1
max_f_eval = 50

f_rand = lambda x: self_opt_react(x, noise_init, N_SAA)
DIRECT_f_rand = lambda x, grad: self_opt_react(x, noise_init, N_SAA)

initial_outputRand = f_rand(x0)


N = 10

# SO_pybbqa_list = []
# for i in range(N):
#     print('Py-BOBYQA: iteration ', i, ' out of ', N, ' ongoing')
#     SO_pybobyqa = PyBobyqaWrapper().solve(f, x0, bounds=bounds.T, \
#                                               maxfun= max_f_eval, constraints=1, \
#                                               seek_global_minimum = True, \
#                                               scaling_within_bounds = True, \
#                                               mu_con = 1e3)
#     SO_pybbqa_list.append(SO_pybobyqa)   

# with open('pybbqaSO_list.pickle', 'wb') as handle:
#     pickle.dump(SO_pybbqa_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('pybbqaSORand_list.pickle', 'rb') as handle:
    SORand_pybbqa_list = pickle.load(handle)

# SO_simplex_list = []
# for i in range(N):
#     rnd_seed = i
#     print('Simplex: iteration ', i, ' out of ', N, ' ongoing')
#     SO_simplex = simplex_method(f, x0, bounds, max_iter = 50, max_f_eval = max_f_eval, \
#                             constraints = 1, rnd_seed = i, mu_con = 1e3)
#     SO_simplex_list.append(SO_simplex)
# print('Simplex iterations completed')

# with open('simplexSO_list.pickle', 'wb') as handle:
#     pickle.dump(SO_simplex_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('simplexSORand_list.pickle', 'rb') as handle:
    SORand_simplex_list = pickle.load(handle)

# SO_findiff_list = []
# for i in range(N):
#     print('Fin. Diff: iteration ', i, ' out of ', N, ' ongoing')
#     SO_FiniteDiff = finite_Diff_Newton(f, x0, bounds = bounds, \
#                                     con_weight = 1000, max_f_eval=max_f_eval)
#     SO_findiff_list.append(SO_FiniteDiff)
# print('Approx Newton iterations completed')

# with open('findiffSO_list.pickle', 'wb') as handle:
#     pickle.dump(SO_findiff_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('findiffSORand_list.pickle', 'rb') as handle:
    SORand_findiff_list = pickle.load(handle)



# SO_Adam_list = []
# for i in range(N):
#     print('Adam: iteration ', i, ' out of ', N, ' ongoing')
#     SO_Adam = Adam_optimizer(f, x0, method = 'forward', \
#                                       bounds = bounds, alpha = 0.4, \
#                                       beta1 = 0.2, beta2  = 0.1, \
#                                       max_f_eval = 100, con_weight = 1000)
#     SO_Adam_list.append(SO_Adam)
# print('Adam iterations completed')

# with open('AdamSO_list.pickle', 'wb') as handle:
#     pickle.dump(SO_Adam_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

# SO_BFGS_list = []
# for i in range(N):
#     print('BFGS: iteration ', i, ' out of ', N, ' ongoing')
#     SO_BFGS = BFGS_optimizer(f, x0, bounds = bounds, \
#                              con_weight = 1000)
#     SO_BFGS_list.append(SO_BFGS)
# print('BFGS iterations completed')

# with open('BFGSSO_list.pickle', 'wb') as handle:
#     pickle.dump(SO_BFGS_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # print(x0) 
# N_min_s = 15
# init_radius = 0.5
# method = 'Discrimination'
# SO_CUATRO_global_list = []
# for i in range(N):
#     rnd_seed = i
#     print('CUATRO_g: iteration ', i, ' out of ', N, ' ongoing')
#     SO_CUATRO_global = CUATRO(f, x0, init_radius, bounds = bounds, \
#                           N_min_samples = N_min_s, tolerance = 1e-10,\
#                           beta_red = 0.9, rnd = rnd_seed, method = 'global', \
#                           constr_handling = method, max_f_eval= max_f_eval)
#     SO_CUATRO_global_list.append(SO_CUATRO_global)
# print('CUATRO global iterations completed')    

# with open('CUATROgSO_list.pickle', 'wb') as handle:
#     pickle.dump(SO_CUATRO_global_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('CUATROgSORand_list.pickle', 'rb') as handle:
    SORand_CUATRO_global_list = pickle.load(handle)

# N_min_s = 6
# init_radius = 0.1
# method = 'Fitting'
# SO_CUATRO_local_list = []
# for i in range(N):
#     rnd_seed = i
#     print('CUATRO_l: iteration ', i, ' out of ', N, ' ongoing')
#     SO_CUATRO_local = CUATRO(f, x0, init_radius, bounds = bounds, \
#                           N_min_samples = N_min_s, tolerance = 1e-10,\
#                           beta_red = 0.5, rnd = rnd_seed, method = 'local', \
#                           constr_handling = method, max_f_eval= max_f_eval)
#     SO_CUATRO_local_list.append(SO_CUATRO_local)
# print('CUATRO local iterations completed') 

# with open('CUATROlSO_list.pickle', 'wb') as handle:
#     pickle.dump(SO_CUATRO_local_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('CUATROlSORand_list.pickle', 'rb') as handle:
    SORand_CUATRO_local_list = pickle.load(handle)


# SO_SQSnobFit_list = []
# for i in range(N):
#     print('Snobfit: iteration ', i, ' out of ', N, ' ongoing')
#     SO_SQSnobFit = SQSnobFitWrapper().solve(f, x0, bounds, \
#                                    maxfun = max_f_eval, constraints=1, \
#                                    mu_con = 1e3)
#     SO_SQSnobFit_list.append(SO_SQSnobFit)
# print('10 SnobFit iterations completed') 

# with open('SnobfitSO_list.pickle', 'wb') as handle:
#     pickle.dump(SO_SQSnobFit_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('SnobfitSORand_list.pickle', 'rb') as handle:
    SORand_SQSnobFit_list = pickle.load(handle)

# SO_DIRECT_list = []
# for i in range(N):
#     print('DIRECT: iteration ', i, ' out of ', N, ' ongoing')
#     SO_DIRECT =  DIRECTWrapper().solve(DIRECT_f, x0, bounds, mu_con = 1e3, \
#                                     maxfun = max_f_eval, constraints=1)
#     SO_DIRECT_list.append(SO_DIRECT)
# print('10 DIRECT iterations completed')  

# with open('directSO_list.pickle', 'wb') as handle:
#     pickle.dump(SO_DIRECT_list, handle, protocol=pickle.HIGHEST_PROTOCOL)  

with open('DIRECTSORand_list.pickle', 'rb') as handle:
    SORand_DIRECT_list = pickle.load(handle)

# N = 10 ; nbr_feval = 30
# SO_Bayes_list = []
# for i in range(N):
#     Bayes = BayesOpt()
#     pyro.set_rng_seed(i)
#     print('Bayes: iteration ', i, ' out of ', N, ' ongoing')
#     SO_Bayes = Bayes.solve(f, x0, acquisition='EI',bounds=bounds.T, \
#                             print_iteration = True, constraints=1, casadi=True, \
#                             maxfun = nbr_feval, ).output_dict
#     SO_Bayes_list.append(SO_Bayes)
 
# print('10 BayesOpt deterministic iterations completed')

# with open('BayesSO_list.pickle', 'wb') as handle:
#     pickle.dump(SO_Bayes_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('BayesSORand_list.pickle', 'rb') as handle:
    SORand_Bayes_list = pickle.load(handle)
   
SORand_Bayes_list = fix_starting_points(SORand_Bayes_list, x0, initial_outputRand)
SORand_DIRECT_list = fix_starting_points(SORand_DIRECT_list, x0, initial_outputRand) 
SORand_pybbqa_list = fix_starting_points(SORand_pybbqa_list, x0, initial_outputRand) 
SORand_simplex_list = fix_starting_points(SORand_simplex_list, x0, initial_outputRand, only_starting_point=True) 
   
plt.rcParams["font.family"] = "Times New Roman"
ft = int(15)
font = {'size': ft}
plt.rc('font', **font)
params = {'legend.fontsize': 12.5,
              'legend.handlelength': 2}
plt.rcParams.update(params)   
    
plant = systems.Static_PDE_reaction_system()

N_draw = 30
x = np.linspace(0, 1, N_draw)
y = np.linspace(0, 1, N_draw)
X, Y = np.meshgrid(x, y)
# Z_SO = np.zeros((N_draw, N_draw))
# g1_SO = np.zeros((N_draw, N_draw))
# for i in range(len(x)):
#     print(i+1, ' out of ', N_draw)
#     for j in range(len(y)):
    
#         temp = f(np.array([x[i], y[j]]))
#         Z_SO[j][i] = temp[0]
#         g1_SO[j][i] = temp[1][0]

# with open('Z_SO.pickle', 'wb') as handle:
#     pickle.dump(Z_SO, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('g1_SO.pickle', 'wb') as handle:
#     pickle.dump(g1_SO, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('Z_SO.pickle', 'rb') as handle:
    Z_SO = pickle.load(handle)
with open('g1_SO.pickle', 'rb') as handle:
    g1_SO = pickle.load(handle)

plt.rcParams["font.family"] = "Times New Roman"
ft = int(15)
font = {'size': ft}
plt.rc('font', **font)
params = {'legend.fontsize': 12.5,
              'legend.handlelength': 2}
plt.rcParams.update(params)



fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(X, Y, Z_SO, g1_SO)
for i in range(len(SORand_pybbqa_list)):
    x_best = np.array(SORand_pybbqa_list[i]['x_best_so_far'])
    f_best = np.array(SORand_pybbqa_list[i]['f_best_so_far'])
    x_ind = np.array(SORand_pybbqa_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'Py-BOBYQA'+str(i))
    # ax1.plot(x_ind, f_best, label = 'CUATRO_g'+str(i))
    ax2.plot(x_best[:,0], x_best[:,1], '--', label = 'Py-BOBYQA'+str(i))
ax1.legend()
# ax1.set_yscale('log')
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.legend()
ax2.set_xlim(bounds[0])
ax2.set_ylim(bounds[1])
fig1.savefig('Plots/SORand_pybbqa_Convergence_plot.svg', format = "svg")
fig2.savefig('Plots/SORand_pybbqa_2Dspace_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(X, Y, Z_SO, g1_SO)
for i in range(len(SORand_findiff_list)):
    x_best = np.array(SORand_findiff_list[i]['x_best_so_far'])
    f_best = np.array(SORand_findiff_list[i]['f_best_so_far'])
    x_ind = np.array(SORand_findiff_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'Fin. Diff.'+str(i))
    # ax1.plot(x_ind, f_best, label = 'CUATRO_g'+str(i))
    ax2.plot(x_best[:,0], x_best[:,1], '--', label = 'Fin. Diff.'+str(i))
ax1.legend()
# ax1.set_yscale('log')
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.legend()
ax2.set_xlim(bounds[0])
ax2.set_ylim(bounds[1])
fig1.savefig('Plots/SORand_FinDiff_Convergence_plot.svg', format = "svg")
fig2.savefig('Plots/SORand_FinDiff_2Dspace_plot.svg', format = "svg")

# fig1 = plt.figure()
# ax1 = fig1.add_subplot()
# ax2, fig2 = trust_fig(X, Y, Z_SO, g1_SO)
# for i in range(len(SO_Adam_list)):
#     x_best = np.array(SO_Adam_list[i]['x_best_so_far'])
#     f_best = np.array(SO_Adam_list[i]['f_best_so_far'])
#     x_ind = np.array(SO_Adam_list[i]['samples_at_iteration'])
#     ax1.step(x_ind, f_best, where = 'post', label = 'Adam'+str(i))
#     # ax1.plot(x_ind, f_best, label = 'CUATRO_g'+str(i))
#     ax2.plot(x_best[:,0], x_best[:,1], '--', label = 'Adam'+str(i))
# ax1.legend()
# # ax1.set_yscale('log')
# ax1.set_xlabel('Nbr. of function evaluations')
# ax1.set_ylabel('Best function evaluation')
# ax2.set_xlabel('$x_1$')
# ax2.set_ylabel('$x_2$')
# ax2.legend()
# ax2.set_xlim(bounds[0])
# ax2.set_ylim(bounds[1])
# fig1.savefig('Plots/SO_Adam_Convergence_plot.svg', format = "svg")
# fig2.savefig('Plots/SO_Adam_2Dspace_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(X, Y, Z_SO, g1_SO)
for i in range(len(SORand_CUATRO_global_list)):
    x_best = np.array(SORand_CUATRO_global_list[i]['x_best_so_far'])
    f_best = np.array(SORand_CUATRO_global_list[i]['f_best_so_far'])
    x_ind = np.array(SORand_CUATRO_global_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'CUATRO_g'+str(i))
    # ax1.plot(x_ind, f_best, label = 'CUATRO_g'+str(i))
    ax2.plot(x_best[:,0], x_best[:,1], '--', label = 'CUATRO_g'+str(i))
ax1.legend()
# ax1.set_yscale('log')
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.legend()
ax2.set_xlim(bounds[0])
ax2.set_ylim(bounds[1])
fig1.savefig('Plots/SORand_CUATROg_Convergence_plot.svg', format = "svg")
fig2.savefig('Plots/SORand_CUATROg_2Dspace_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(X, Y, Z_SO, g1_SO)
for i in range(len(SORand_CUATRO_local_list)):
    x_best = np.array(SORand_CUATRO_local_list[i]['x_best_so_far'])
    f_best = np.array(SORand_CUATRO_local_list[i]['f_best_so_far'])
    x_ind = np.array(SORand_CUATRO_local_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'CUATRO_l'+str(i))
    # ax1.plot(x_ind, f_best, label = 'CUATRO_l'+str(i))
    ax2.plot(x_best[:,0], x_best[:,1], '--', label = 'CUATRO_l'+str(i))
ax1.legend()
# ax1.set_yscale('log')
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.legend()
ax2.set_xlim(bounds[0])
ax2.set_ylim(bounds[1])
fig1.savefig('Plots/SORand_CUATROl_Convergence_plot.svg', format = "svg")
fig2.savefig('Plots/SORand_CUATROl_2Dspace_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(X, Y, Z_SO, g1_SO)
for i in range(len(SORand_Bayes_list)):
    x_best = np.array(SORand_Bayes_list[i]['x_best_so_far'])
    f_best = np.array(SORand_Bayes_list[i]['f_best_so_far'])
    nbr_feval = len(SORand_Bayes_list[i]['f_store'])
    ax1.step(np.arange(len(f_best)), f_best, where = 'post', \
          label = 'BO'+str(i)+'; #f_eval: ' + str(nbr_feval))
    ax2.plot(x_best[:,0], x_best[:,1], '--', \
          label = 'BO'+str(i)+'; #f_eval: ' + str(nbr_feval))
ax1.legend()
# ax1.set_yscale('log')
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.legend()
ax2.set_xlim(bounds[0])
ax2.set_ylim(bounds[1])
fig1.savefig('Plots/SORand_BO_Convergence_plot.svg', format = "svg")
fig2.savefig('Plots/SORand_BO_2Dspace_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(X, Y, Z_SO, g1_SO)
for i in range(len(SORand_simplex_list)):
    x_best = np.array(SORand_simplex_list[i]['x_best_so_far'])
    f_best = np.array(SORand_simplex_list[i]['f_best_so_far'])
    x_ind = np.array(SORand_simplex_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'Simplex'+str(i))
    ax2.plot(x_best[:,0], x_best[:,1], '--', label = 'Simplex'+str(i))
ax1.legend()
# ax1.set_yscale('log')
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.set_xlim(bounds[0])
ax2.set_ylim(bounds[1])
ax2.legend()
fig1.savefig('Plots/SORand_simplex_Convergence_plot.svg', format = "svg")
fig2.savefig('Plots/SORand_simplex_2Dspace_plot.svg', format = "svg")

## Change to x_best_So_far
fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(X, Y, Z_SO, g1_SO)
for i in range(len(SORand_SQSnobFit_list)):
    x_best = np.array(SORand_SQSnobFit_list[i]['x_best_so_far'])
    f_best = np.array(SORand_SQSnobFit_list[i]['f_best_so_far'])
    x_ind = np.array(SORand_SQSnobFit_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'Snobfit'+str(i))
    ax2.plot(x_best[:,0], x_best[:,1], '--', label = 'Snobfit'+str(i))
ax1.legend()
# ax1.set_yscale('log')
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.set_xlim(bounds[0])
ax2.set_ylim(bounds[1])
ax2.legend()
fig1.savefig('Plots/SORand_SQSnobFit_Convergence_plot.svg', format = "svg")
fig2.savefig('Plots/SORand_SQSnobFit_2Dspace_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(X, Y, Z_SO, g1_SO)
for i in range(len(SORand_DIRECT_list)):
    x_best = np.array(SORand_DIRECT_list[i]['x_best_so_far'])
    f_best = np.array(SORand_DIRECT_list[i]['f_best_so_far'])
    x_ind = np.array(SORand_DIRECT_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'DIRECT'+str(i))
    ax2.plot(x_best[:,0], x_best[:,1], '--', label = 'DIRECT'+str(i))
ax1.legend()
# ax1.set_yscale('log')
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.set_xlim(bounds[0])
ax2.set_ylim(bounds[1])
ax2.legend()
fig1.savefig('Plots/SORand_DIRECT_Convergence_plot.svg', format = "svg")
fig2.savefig('Plots/SORand_DIRECT_2Dspace_plot.svg', format = "svg")



sol_Cg = average_from_list(SORand_CUATRO_global_list)
test_CUATROg, test_av_CUATROg, test_min_CUATROg, test_max_CUATROg = sol_Cg
sol_Cl = average_from_list(SORand_CUATRO_local_list)
test_CUATROl, test_av_CUATROl, test_min_CUATROl, test_max_CUATROl = sol_Cl

sol_Splx = average_from_list(SORand_simplex_list)
test_Splx, test_av_Splx, test_min_Splx, test_max_Splx = sol_Splx
sol_SQSF = average_from_list(SORand_SQSnobFit_list)
test_SQSF, test_av_SQSF, test_min_SQSF, test_max_SQSF = sol_SQSF
sol_DIR = average_from_list(SORand_DIRECT_list)
test_DIR, test_av_DIR, test_min_DIR, test_max_DIR = sol_DIR
sol_pybbyqa = average_from_list(SORand_pybbqa_list)
test_pybbqa, test_av_pybbqa, test_min_pybbqa, test_max_pybbqa = sol_pybbyqa
sol_findiff = average_from_list(SORand_findiff_list)
test_findiff, test_av_findiff, test_min_findiff, test_max_findiff = sol_findiff
# sol_BFGS = average_from_list(SO_BFGS_list)
# test_BFGS, test_av_BFGS, test_min_BFGS, test_max_BFGS = sol_BFGS
# sol_Adam = average_from_list(SO_Adam_list)
# test_Adam, test_av_Adam, test_min_Adam, test_max_Adam = sol_Adam
sol_BO = average_from_list(SORand_Bayes_list)
test_BO, test_av_BO, test_min_BO, test_max_BO = sol_BO



fig = plt.figure()
ax = fig.add_subplot()
ax.step(np.arange(1, 101), test_av_CUATROg, where = 'post', label = 'CUATRO_g', c = 'b')
ax.fill_between(np.arange(1, 101), test_min_CUATROg, \
                test_max_CUATROg, color = 'b', alpha = .5, step = 'post')
ax.step(np.arange(1, 101), test_av_CUATROl, where = 'post', label = 'CUATRO_l', c = 'c')
ax.fill_between(np.arange(1, 101), test_min_CUATROl, \
                test_max_CUATROl, color = 'c', alpha = .5, step = 'post')
ax.step(np.arange(1, 101), test_av_pybbqa, where = 'post', label = 'Py-BOBYQA ', c = 'green')
ax.fill_between(np.arange(1, 101), test_min_pybbqa, \
                test_max_pybbqa, color = 'green', alpha = .5, step = 'post')
ax.step(np.arange(1, 101), test_av_SQSF, where = 'post', label = 'Snobfit', c = 'orange')
ax.fill_between(np.arange(1, 101), test_min_SQSF, \
                test_max_SQSF, color = 'orange', alpha = .5, step = 'post')
ax.step(np.arange(1, 101), test_av_BO, where = 'post', \
          label = 'Bayes. Opt.', c = 'r')
ax.fill_between(np.arange(1, 101), test_min_BO, \
                test_max_BO, color = 'r', alpha = .5, step = 'post')

ax.legend()
# ax.set_yscale('log')
ax.set_xlabel('Number of function evaluations')
ax.set_ylabel('Best function evaluation')
ax.set_xlim([1, 50])
ax.legend(loc = 'upper right')   
fig.savefig('Publication Plots/SORand_Model.svg', format = "svg")


fig = plt.figure()
ax = fig.add_subplot()
# ax.step(np.arange(1, 101), test_av_Nest, where = 'post', label = 'Nesterov', c = 'brown')
# ax.fill_between(np.arange(1, 101), test_min_Nest, \
#                 test_max_Nest, color = 'brown', alpha = .5)
ax.step(np.arange(1, 101), test_av_Splx, where = 'post', label = 'Simplex', c = 'green')
ax.fill_between(np.arange(1, 101), test_min_Splx, \
                test_max_Splx, color = 'green', alpha = .5, step = 'post')
ax.step(np.arange(1, 101), test_av_findiff, where = 'post', label = 'Newton', c = 'grey')
ax.fill_between(np.arange(1, 101), test_min_findiff, \
                test_max_findiff, color = 'grey', alpha = .5, step = 'post')
# ax.step(np.arange(1, 101), test_av_BFGS, where = 'post', label = 'Approx. BFGS', c = 'orange')
# ax.fill_between(np.arange(1, 101), test_min_BFGS, \
#                 test_max_BFGS, color = 'orange', alpha = .5)
# ax.step(np.arange(1, 101), test_av_Adam, where = 'post', label = 'Adam ', c = 'blue')
# ax.fill_between(np.arange(1, 101), test_min_Adam, \
                # test_max_Adam, color = 'blue', alpha = .5)
ax.step(np.arange(1, 101), test_av_DIR, where = 'post', label = 'DIRECT', c = 'violet')
ax.fill_between(np.arange(1, 101), test_min_DIR, \
                test_max_DIR, color = 'violet', alpha = .5, step = 'post')


ax.legend()
# ax.set_yscale('log')
ax.set_xlabel('Number of function evaluations')
ax.set_ylabel('Best function evaluation')
ax.set_xlim([1, 50])
ax.legend(loc = 'upper right')
fig.savefig('Publication Plots/SORand_Others.svg', format = "svg")



plt.rcParams["font.family"] = "Times New Roman"
ft = int(15)
font = {'size': ft}
plt.rc('font', **font)
params = {'legend.fontsize': 12.5,
              'legend.handlelength': 2}
plt.rcParams.update(params)

DIRECT_med = medianx_from_list(SORand_DIRECT_list, x0)
CUATROg_med = medianx_from_list(SORand_CUATRO_global_list, x0)
Bayes_med = medianx_from_list(SORand_Bayes_list, x0)
SQSF_med = medianx_from_list(SORand_SQSnobFit_list, x0)

ax2, fig2 = trust_fig(X, Y, Z_SO, g1_SO)
ax2.plot(DIRECT_med[:,0], DIRECT_med[:,1], label = 'DIRECT', \
            markersize = 4, alpha=.5, marker='o', linewidth = 2)
ax2.plot(CUATROg_med[:,0], CUATROg_med[:,1], label = 'CUATRO_g', \
            markersize = 4, alpha=.5, marker='o', linewidth = 2)
ax2.plot(Bayes_med[:,0], Bayes_med[:,1], label = 'Bayes. Opt.', \
            markersize = 4, alpha=.5, marker='o', linewidth = 2)
ax2.plot(SQSF_med[:,0], SQSF_med[:,1], label = 'Snobfit', \
            markersize = 4, alpha=.5, marker='o', linewidth = 2)
ax2.scatter(x0[0], x0[1], label = 'Init. guess', marker = 'D', s = 40, c = 'k')
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.set_xlim(bounds[0])
ax2.set_ylim(bounds[1])
ax2.legend()
fig2.savefig('Publication plots/SORand_2D_convergence_best.svg', format = "svg")









