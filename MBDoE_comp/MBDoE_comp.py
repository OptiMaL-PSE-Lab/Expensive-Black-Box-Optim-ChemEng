# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 02:29:51 2021

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

import matplotlib.pyplot as plt

from case_studies.MBDoE.construct_MBDoE_funs import set_funcs_mbdoe

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

bounds = np.array([[0,1.],[0,1.],[0,1.],[0,1.]])
x0 = np.array([0.1]*4)

cost = lambda x: set_funcs_mbdoe(x)
DIRECT_cost = lambda x, grad: set_funcs_mbdoe(x)

initial_output = cost(x0)

max_f_eval = 100

N = 10
MBDoE_pybobyqa_list = []
for i in range(N):
    rnd_seed = i
    MBDoE_pybobyqa = PyBobyqaWrapper().solve(cost, x0, bounds=bounds.T, \
                                      maxfun= max_f_eval, constraints=1, \
                                      seek_global_minimum = True)
    MBDoE_pybobyqa_list.append(MBDoE_pybobyqa)
print('10 Py-BOBYQA iterations completed')

N = 10
MBDoE_simplex_list = []
for i in range(N):
    rnd_seed = i
    MBDoE_simplex = simplex_method(cost, x0, bounds, max_iter = 50, \
                            constraints = 1, rnd_seed = i, mu_con = 1e6)
    MBDoE_simplex_list.append(MBDoE_simplex)
print('10 simplex iterations completed')

MBDoE_FiniteDiff = finite_Diff_Newton(cost, x0, bounds = bounds, \
                                    con_weight = 1e6, check_bounds = True)
 
MBDoE_BFGS = BFGS_optimizer(cost, x0, bounds = bounds, \
                               con_weight = 1e6, check_bounds = True)
    
MBDoE_Adam = Adam_optimizer(cost, x0, method = 'forward', \
                                      bounds = bounds, alpha = 0.4, \
                                      beta1 = 0.2, beta2  = 0.1, \
                                      max_f_eval = 100, con_weight = 1e6, \
                                        check_bounds = True)
 
MBDoE_FiniteDiff_list = [] ; MBDoE_BFGS_list = [] ; MBDoE_Adam_list = []
MBDoE_FiniteDiff_list.append(MBDoE_FiniteDiff)
MBDoE_BFGS_list.append(MBDoE_BFGS)    
MBDoE_Adam_list.append(MBDoE_Adam)

N_min_s = 15
init_radius = 0.5
method = 'Discrimination'
N = 10
MBDoE_CUATRO_global_list = []
for i in range(N):
    rnd_seed = i
    MBDoE_CUATRO_global = CUATRO(cost, x0, init_radius, bounds = bounds, \
                          N_min_samples = N_min_s, tolerance = 1e-10,\
                          beta_red = 0.9, rnd = rnd_seed, method = 'global', \
                          constr_handling = method)
    MBDoE_CUATRO_global_list.append(MBDoE_CUATRO_global)
print('10 CUATRO global iterations completed')      
 
N_min_s = 6
init_radius = 0.1
method = 'Fitting'
N = 10
MBDoE_CUATRO_local_list = []
for i in range(N):
    rnd_seed = i
    MBDoE_CUATRO_local = CUATRO(cost, x0, init_radius, bounds = bounds, \
                          N_min_samples = N_min_s, tolerance = 1e-10,\
                          beta_red = 0.9, rnd = rnd_seed, method = 'local', \
                          constr_handling = method)
    MBDoE_CUATRO_local_list.append(MBDoE_CUATRO_local)
print('10 CUATRO local iterations completed') 

N = 10
MBDoE_SQSnobFit_list = []
for i in range(N):
    MBDoE_SQSnobFit = SQSnobFitWrapper().solve(cost, x0, bounds, mu_con = 1e6, \
                                    maxfun = max_f_eval, constraints=1)
    MBDoE_SQSnobFit_list.append(MBDoE_SQSnobFit)
print('10 SnobFit iterations completed') 

N = 10
MBDoE_DIRECT_list = []
for i in range(N):
    MBDoE_DIRECT =  DIRECTWrapper().solve(DIRECT_cost, x0, bounds, mu_con = 1e6, \
                                    maxfun = max_f_eval, constraints=1)
    MBDoE_DIRECT_list.append(MBDoE_DIRECT)
print('10 DIRECT iterations completed')    

with open('BayesMBDoE_list.pickle', 'rb') as handle:
    MBDoE_Bayes_list = pickle.load(handle)

MBDoE_Bayes_list = fix_starting_points(MBDoE_Bayes_list, x0, initial_output)
MBDoE_DIRECT_list = fix_starting_points(MBDoE_DIRECT_list, x0, initial_output)
MBDoE_simplex_list = fix_starting_points(MBDoE_simplex_list, x0, initial_output)
MBDoE_pybobyqa_list = fix_starting_points(MBDoE_pybobyqa_list, x0, initial_output)

plt.rcParams["font.family"] = "Times New Roman"
ft = int(15)
font = {'size': ft}
plt.rc('font', **font)
params = {'legend.fontsize': 12.5,
              'legend.handlelength': 2}
plt.rcParams.update(params)

fig1 = plt.figure()
ax1 = fig1.add_subplot()
for i in range(len(MBDoE_pybobyqa_list)):
    x_best = np.array(MBDoE_pybobyqa_list[i]['x_best_so_far'])
    f_best = np.array(MBDoE_pybobyqa_list[i]['f_best_so_far'])
    x_ind = np.array(MBDoE_pybobyqa_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'Py-BOBYQA'+str(i))
    # ax1.plot(x_ind, f_best, label = 'CUATRO_g'+str(i))
ax1.legend()
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
ax1.set_yscale('log')
fig1.savefig('MBDoE_plots/MBDoE_PyBOBYQA_Convergence_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
for i in range(len(MBDoE_FiniteDiff_list)):
    x_best = np.array(MBDoE_FiniteDiff_list[i]['x_best_so_far'])
    f_best = np.array(MBDoE_FiniteDiff_list[i]['f_best_so_far'])
    x_ind = np.array(MBDoE_FiniteDiff_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'Fin. Diff.'+str(i))
    # ax1.plot(x_ind, f_best, label = 'CUATRO_g'+str(i))
ax1.legend()
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
ax1.set_yscale('log')
fig1.savefig('MBDoE_plots/MBDoE_FiniteDiff_Convergence_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
for i in range(len(MBDoE_BFGS_list)):
    x_best = np.array(MBDoE_BFGS_list[i]['x_best_so_far'])
    f_best = np.array(MBDoE_BFGS_list[i]['f_best_so_far'])
    x_ind = np.array(MBDoE_BFGS_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'BFGS'+str(i))
    # ax1.plot(x_ind, f_best, label = 'CUATRO_g'+str(i))
ax1.legend()
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
ax1.set_yscale('log')
fig1.savefig('MBDoE_plots/MBDoE_BFGS_Convergence_plot.svg', format = "svg")

# fig1 = plt.figure()
# ax1 = fig1.add_subplot()
# for i in range(len(MBDoE_Adam_list)):
#     x_best = np.array(MBDoE_Adam_list[i]['x_best_so_far'])
#     f_best = np.array(MBDoE_Adam_list[i]['f_best_so_far'])
#     x_ind = np.array(MBDoE_Adam_list[i]['samples_at_iteration'])
#     ax1.step(x_ind, f_best, where = 'post', label = 'Adam'+str(i))
#     # ax1.plot(x_ind, f_best, label = 'CUATRO_g'+str(i))
# ax1.legend()
# ax1.set_xlabel('Nbr. of function evaluations')
# ax1.set_ylabel('Best function evaluation')
# ax1.set_yscale('log')
# fig1.savefig('MBDoE_plots/MBDoE_Adam_Convergence_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
for i in range(len(MBDoE_CUATRO_global_list)):
    x_best = np.array(MBDoE_CUATRO_global_list[i]['x_best_so_far'])
    f_best = np.array(MBDoE_CUATRO_global_list[i]['f_best_so_far'])
    x_ind = np.array(MBDoE_CUATRO_global_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'CUATRO_g'+str(i))
    # ax1.plot(x_ind, f_best, label = 'CUATRO_g'+str(i))
ax1.legend()
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
ax1.set_yscale('log')
fig1.savefig('MBDoE_plots/MBDoE_CUATROg_Convergence_plot.svg', format = "svg")


fig1 = plt.figure()
ax1 = fig1.add_subplot()
for i in range(len(MBDoE_CUATRO_local_list)):
    x_best = np.array(MBDoE_CUATRO_local_list[i]['x_best_so_far'])
    f_best = np.array(MBDoE_CUATRO_local_list[i]['f_best_so_far'])
    x_ind = np.array(MBDoE_CUATRO_local_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'CUATRO_l'+str(i))
    # ax1.plot(x_ind, f_best, label = 'CUATRO_l'+str(i))
ax1.legend()
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
ax1.set_yscale('log')
fig1.savefig('MBDoE_plots/MBDoE_CUATROl_Convergence_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
for i in range(len(MBDoE_Bayes_list)):
    x_best = np.array(MBDoE_Bayes_list[i]['x_best_so_far'])
    f_best = np.array(MBDoE_Bayes_list[i]['f_best_so_far'])
    nbr_feval = len(MBDoE_Bayes_list[i]['f_store'])
    ax1.step(np.arange(len(f_best)), f_best, where = 'post', \
          label = 'BO'+str(i))
ax1.legend()
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
ax1.set_yscale('log')
fig1.savefig('MBDoE_plots/MBDoE_BO_Convergence_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
for i in range(len(MBDoE_simplex_list)):
    x_best = np.array(MBDoE_simplex_list[i]['x_best_so_far'])
    f_best = np.array(MBDoE_simplex_list[i]['f_best_so_far'])
    x_ind = np.array(MBDoE_simplex_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'Simplex'+str(i))
ax1.legend()
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
ax1.set_yscale('log')
fig1.savefig('MBDoE_plots/MBDoE_Simplex_Convergence_plot.svg', format = "svg")


# ## Change to x_best_So_far
# fig1 = plt.figure()
# ax1 = fig1.add_subplot()
# for i in range(len(MBDoE_Nest_list)):
#     x_best = np.array(MBDoE_Nest_list[i]['x_best_so_far'])
#     f_best = np.array(MBDoE_Nest_list[i]['f_best_so_far'])
#     x_ind = np.array(MBDoE_Nest_list[i]['samples_at_iteration'])
#     ax1.step(x_ind, f_best, where = 'post', label = 'Nest.'+str(i))
# ax1.legend()
# ax1.set_yscale('log')
# ax1.set_xlabel('Nbr. of function evaluations')
# ax1.set_ylabel('Best function evaluation')
# fig1.savefig('MBDoE_plots/MBDoE_Nesterov_Convergence_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
for i in range(len(MBDoE_SQSnobFit_list)):
    x_best = np.array(MBDoE_SQSnobFit_list[i]['x_best_so_far'])
    f_best = np.array(MBDoE_SQSnobFit_list[i]['f_best_so_far'])
    x_ind = np.array(MBDoE_SQSnobFit_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'SQSnobfit.'+str(i))
ax1.legend()
ax1.set_yscale('log')
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
fig1.savefig('MBDoE_plots/MBDoE_SQSnobFit_Convergence_plot.svg', format = "svg")


fig1 = plt.figure()
ax1 = fig1.add_subplot()
for i in range(len(MBDoE_DIRECT_list)):
    x_best = np.array(MBDoE_DIRECT_list[i]['x_best_so_far'])
    f_best = np.array(MBDoE_DIRECT_list[i]['f_best_so_far'])
    x_ind = np.array(MBDoE_DIRECT_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'DIRECT'+str(i))
ax1.legend()
ax1.set_yscale('log')
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
fig1.savefig('MBDoE_plots/MBDoE_DIRECT_Convergence_plot.svg', format = "svg")


sol_Cg = average_from_list(MBDoE_CUATRO_global_list)
test_CUATROg, test_av_CUATROg, test_min_CUATROg, test_max_CUATROg = sol_Cg
sol_Cl = average_from_list(MBDoE_CUATRO_local_list)
test_CUATROl, test_av_CUATROl, test_min_CUATROl, test_max_CUATROl = sol_Cl
# sol_Nest = average_from_list(MBDoE_Nest_list)
# test_Nest, test_av_Nest, test_min_Nest, test_max_Nest = sol_Nest
sol_Splx = average_from_list(MBDoE_simplex_list)
test_Splx, test_av_Splx, test_min_Splx, test_max_Splx = sol_Splx
sol_SQSF = average_from_list(MBDoE_SQSnobFit_list)
test_SQSF, test_av_SQSF, test_min_SQSF, test_max_SQSF = sol_SQSF
sol_DIR = average_from_list(MBDoE_DIRECT_list)
test_DIR, test_av_DIR, test_min_DIR, test_max_DIR = sol_DIR
sol_pybbyqa = average_from_list(MBDoE_pybobyqa_list)
test_pybbqa, test_av_pybbqa, test_min_pybbqa, test_max_pybbqa = sol_pybbyqa
sol_findiff = average_from_list(MBDoE_FiniteDiff_list)
test_findiff, test_av_findiff, test_min_findiff, test_max_findiff = sol_findiff
sol_BFGS = average_from_list(MBDoE_BFGS_list)
test_BFGS, test_av_BFGS, test_min_BFGS, test_max_BFGS = sol_BFGS
# sol_Adam = average_from_list(MBDoE_Adam_list)
# test_Adam, test_av_Adam, test_min_Adam, test_max_Adam = sol_Adam
sol_BO = average_from_list(MBDoE_Bayes_list)
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
ax.step(np.arange(1, 101), test_av_BO, where = 'post', label = 'Bayes. Opt.', c = 'red')
ax.fill_between(np.arange(1, 101), test_min_BO, \
                test_max_BO, color = 'red', alpha = .5, step = 'post')

ax.legend()
ax.set_xlabel('Number of function evaluations')
ax.set_ylabel('Best function evaluation')
ax.set_yscale('log')
ax.legend(loc = 'upper right')
ax.set_xlim([1, 100])   
ax.set_ylim([0.175, 70]) 
fig.savefig('MBDoE_plots/MBDoE_Model.svg', format = "svg")


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
ax.step(np.arange(1, 101), test_av_BFGS, where = 'post', label = 'BFGS', c = 'orange')
ax.fill_between(np.arange(1, 101), test_min_BFGS, \
                test_max_BFGS, color = 'orange', alpha = .5, step = 'post')
# ax.step(np.arange(1, 101), test_av_Adam, where = 'post', label = 'Adam ', c = 'blue')
# ax.fill_between(np.arange(1, 101), test_min_Adam, \
#                 test_max_Adam, color = 'blue', alpha = .5, step = 'post')
ax.step(np.arange(1, 101), test_av_DIR, where = 'post', label = 'DIRECT', c = 'violet')
ax.fill_between(np.arange(1, 101), test_min_DIR, \
                test_max_DIR, color = 'violet', alpha = .5, step = 'post')

ax.legend()
ax.set_xlabel('Number of function evaluations')
ax.set_ylabel('Best function evaluation')
ax.set_yscale('log')
ax.legend(loc = 'upper right')
ax.set_xlim([1, 100])
ax.set_ylim([0.175, 70]) 
fig.savefig('MBDoE_plots/MBDoE_Others.svg', format = "svg")








