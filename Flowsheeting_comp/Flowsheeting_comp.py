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


### Import your function here
from case_studies.MBDoE.construct_MBDoE_funs import set_funcs_mbdoe
###

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


### Change this
bounds = np.array([[0,1.],[0,1.],[0,1.],[0,1.]])
# x0 = np.array([0.1]*4)
x0 = np.array([.5, .5, .36, .2])
###

### Include function here, remember that it should return f(x), [g_1(x), ...]
f = lambda x: x
###
DIRECT_f = lambda x, grad: f(x)

max_f_eval = 100


### If any of them don't work for now, comment them out for now and let me 
### know which ones aren't working. Will see then if the method is just poor
### or if there's a problem with the code

### I would probably start with a direct method like DIRECT to start with, as
### they are at least not as sensitive to ill-conditioning

### Don't forget to set this to the number of constraints
n_constr = 1
###

# N = 10
# FS_pybobyqa_list = []
# for i in range(N):
#     rnd_seed = i
#     FS_pybobyqa = PyBobyqaWrapper().solve(f, x0, bounds=bounds.T, \
#                                       maxfun= max_f_eval, constraints= n_constr, \
#                                       seek_global_minimum = True)
#     FS_pybobyqa_list.append(FS_pybobyqa)
# print('10 Py-BOBYQA iterations completed')

with open('FS_pybobyqa_list.pickle', 'rb') as handle:
    FS_pybobyqa_list = pickle.load(handle)

# N = 10
# FS_simplex_list = []
# for i in range(N):
#     rnd_seed = i
#     FS_simplex = simplex_method(f, x0, bounds, max_iter = 50, \
#                             constraints = n_constr, rnd_seed = i, mu_con = 1e6)
#     FS_simplex_list.append(FS_simplex)
# print('10 simplex iterations completed')

with open('FS_simplex_list (1).pickle', 'rb') as handle:
    FS_simplex_list = pickle.load(handle)

# FS_FiniteDiff = finite_Diff_Newton(f, x0, bounds = bounds, \
#                                     con_weight = 1e6)

with open('FS_FiniteDiff_list (1).pickle', 'rb') as handle:
    FS_FiniteDiff = pickle.load(handle)
 
# FS_BFGS = BFGS_optimizer(f, x0, bounds = bounds, \
#                                con_weight = 1e6)
 
with open('FS_BFGS_list.pickle', 'rb') as handle:
    FS_BFGS = pickle.load(handle)
   
# FS_Adam = Adam_optimizer(f, x0, method = 'forward', \
#                                       bounds = bounds, alpha = 0.4, \
#                                       beta1 = 0.2, beta2  = 0.1, \
#                                       max_f_eval = 100, con_weight = 1e6)
 
with open('FS_Adam_list (2).pickle', 'rb') as handle:
    FS_Adam = pickle.load(handle)

FS_FiniteDiff_list = []  ; FS_Adam_list = []
FS_FiniteDiff_list.append(FS_FiniteDiff)
FS_BFGS_list = [] ; FS_BFGS_list.append(FS_BFGS)    
FS_Adam_list.append(FS_Adam)

# N_min_s = 15
# ### As rule of thumb, feel free to change
# init_radius = np.max(bounds[:,1] - bounds[:,0]) / 2
# ### 
# method = 'Discrimination'
# N = 10
# FS_CUATRO_global_list = []
# for i in range(N):
#     rnd_seed = i
#     FS_CUATRO_global = CUATRO(f, x0, init_radius, bounds = bounds, \
#                           N_min_samples = N_min_s, tolerance = 1e-10,\
#                           beta_red = 0.9, rnd = rnd_seed, method = 'global', \
#                           constr_handling = method)
#     FS_CUATRO_global_list.append(FS_CUATRO_global)
# print('10 CUATRO global iterations completed')   

with open('FS_CUATRO_global_list (1).pickle', 'rb') as handle:
    FS_CUATRO_global_list = pickle.load(handle)   

initial_output = FS_CUATRO_global_list[0]['f_best_so_far'][0], FS_CUATRO_global_list[0]['g_best_so_far'][0]

# N_min_s = 6
# ### Rule of thumb, feel free to change
# #init_radius = np.max(bounds[:,1] - bounds[:,0]) / 10 ## if well-behaved/convex
# init_radius = np.max(bounds[:,1] - bounds[:,0]) / 4  ## if not
# ### 
# method = 'Fitting'
# N = 10
# FS_CUATRO_local_list = []
# for i in range(N):
#     rnd_seed = i
#     FS_CUATRO_local = CUATRO(f, x0, init_radius, bounds = bounds, \
#                           N_min_samples = N_min_s, tolerance = 1e-10,\
#                           beta_red = 0.9, rnd = rnd_seed, method = 'local', \
#                           constr_handling = method)
#     FS_CUATRO_local_list.append(FS_CUATRO_local)
# print('10 CUATRO local iterations completed') 

with open('FS_CUATRO_local_list (1).pickle', 'rb') as handle:
    FS_CUATRO_local_list = pickle.load(handle)   

# N = 10
# FS_SQSnobFit_list = []
# for i in range(N):
#     FS_SQSnobFit = SQSnobFitWrapper().solve(f, x0, bounds, mu_con = 1e6, \
#                                     maxfun = max_f_eval, constraints= n_constr)
#     FS_SQSnobFit_list.append(FS_SQSnobFit)
# print('10 SnobFit iterations completed') 

with open('FS_SNOBFIT_list.pickle', 'rb') as handle:
    FS_SQSnobFit_list = pickle.load(handle)  

# N = 10
# FS_DIRECT_list = []
# for i in range(N):
#     FS_DIRECT =  DIRECTWrapper().solve(DIRECT_f, x0, bounds, mu_con = 1e6, \
#                                     maxfun = max_f_eval, constraints= n_constr)
#     FS_DIRECT_list.append(FS_DIRECT)
# print('10 DIRECT iterations completed')    

with open('FS_DIRECT_list.pickle', 'rb') as handle:
    FS_DIRECT_list = pickle.load(handle)  

with open('BayesFS_list.pickle', 'rb') as handle:
    FS_Bayes_list = pickle.load(handle)

FS_Bayes_list = fix_starting_points(FS_Bayes_list, x0, initial_output)
FS_DIRECT_list = fix_starting_points(FS_DIRECT_list, x0, initial_output)
FS_simplex_list = fix_starting_points(FS_simplex_list, x0, initial_output)
FS_pybobyqa_list = fix_starting_points(FS_pybobyqa_list, x0, initial_output)

plt.rcParams["font.family"] = "Times New Roman"
ft = int(15)
font = {'size': ft}
plt.rc('font', **font)
params = {'legend.fontsize': 12.5,
              'legend.handlelength': 2}
plt.rcParams.update(params)

fig1 = plt.figure()
ax1 = fig1.add_subplot()
for i in range(len(FS_pybobyqa_list)):
    x_best = np.array(FS_pybobyqa_list[i]['x_best_so_far'])
    f_best = np.array(FS_pybobyqa_list[i]['f_best_so_far'])
    x_ind = np.array(FS_pybobyqa_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'Py-BOBYQA'+str(i))
    # ax1.plot(x_ind, f_best, label = 'CUATRO_g'+str(i))
ax1.legend()
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
# ax1.set_yscale('log')
fig1.savefig('Flowsheeting_plots/FS_PyBOBYQA_Convergence_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
for i in range(len(FS_FiniteDiff_list)):
    x_best = np.array(FS_FiniteDiff_list[i]['x_best_so_far'])
    f_best = np.array(FS_FiniteDiff_list[i]['f_best_so_far'])
    x_ind = np.array(FS_FiniteDiff_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'Fin. Diff.'+str(i))
    # ax1.plot(x_ind, f_best, label = 'CUATRO_g'+str(i))
ax1.legend()
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
# ax1.set_yscale('log')
fig1.savefig('Flowsheeting_plots/FS_FiniteDiff_Convergence_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
for i in range(len(FS_BFGS_list)):
    x_best = np.array(FS_BFGS_list[i]['x_best_so_far'])
    f_best = np.array(FS_BFGS_list[i]['f_best_so_far'])
    x_ind = np.array(FS_BFGS_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'BFGS'+str(i))
    # ax1.plot(x_ind, f_best, label = 'CUATRO_g'+str(i))
ax1.legend()
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
# ax1.set_yscale('log')
fig1.savefig('Flowsheeting_plots/FS_BFGS_Convergence_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
for i in range(len(FS_Adam_list)):
    x_best = np.array(FS_Adam_list[i]['x_best_so_far'])
    f_best = np.array(FS_Adam_list[i]['f_best_so_far'])
    x_ind = np.array(FS_Adam_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'Adam'+str(i))
    # ax1.plot(x_ind, f_best, label = 'CUATRO_g'+str(i))
ax1.legend()
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
# ax1.set_yscale('log')
fig1.savefig('Flowsheeting_plots/FS_Adam_Convergence_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
for i in range(len(FS_CUATRO_global_list)):
    x_best = np.array(FS_CUATRO_global_list[i]['x_best_so_far'])
    f_best = np.array(FS_CUATRO_global_list[i]['f_best_so_far'])
    x_ind = np.array(FS_CUATRO_global_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'CUATRO_g'+str(i))
    # ax1.plot(x_ind, f_best, label = 'CUATRO_g'+str(i))
ax1.legend()
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
# ax1.set_yscale('log')
fig1.savefig('Flowsheeting_plots/FS_CUATROg_Convergence_plot.svg', format = "svg")


fig1 = plt.figure()
ax1 = fig1.add_subplot()
for i in range(len(FS_CUATRO_local_list)):
    x_best = np.array(FS_CUATRO_local_list[i]['x_best_so_far'])
    f_best = np.array(FS_CUATRO_local_list[i]['f_best_so_far'])
    x_ind = np.array(FS_CUATRO_local_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'CUATRO_l'+str(i))
    # ax1.plot(x_ind, f_best, label = 'CUATRO_l'+str(i))
ax1.legend()
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
# ax1.set_yscale('log')
fig1.savefig('Flowsheeting_plots/FS_CUATROl_Convergence_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
for i in range(len(FS_Bayes_list)):
    x_best = np.array(FS_Bayes_list[i]['x_best_so_far'])
    f_best = np.array(FS_Bayes_list[i]['f_best_so_far'])
    nbr_feval = len(FS_Bayes_list[i]['f_store'])
    ax1.step(np.arange(len(f_best)), f_best, where = 'post', \
          label = 'BO'+str(i))
ax1.legend()
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
#ax1.set_yscale('log')
fig1.savefig('Flowsheeting_plots/FS_BO_Convergence_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
for i in range(len(FS_simplex_list)):
    x_best = np.array(FS_simplex_list[i]['x_best_so_far'])
    f_best = np.array(FS_simplex_list[i]['f_best_so_far'])
    x_ind = np.array(FS_simplex_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'Simplex'+str(i))
ax1.legend()
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
# ax1.set_yscale('log')
fig1.savefig('Flowsheeting_plots/FS_Simplex_Convergence_plot.svg', format = "svg")


# ## Change to x_best_So_far
# fig1 = plt.figure()
# ax1 = fig1.add_subplot()
# for i in range(len(FS_Nest_list)):
#     x_best = np.array(FS_Nest_list[i]['x_best_so_far'])
#     f_best = np.array(FS_Nest_list[i]['f_best_so_far'])
#     x_ind = np.array(FS_Nest_list[i]['samples_at_iteration'])
#     ax1.step(x_ind, f_best, where = 'post', label = 'Nest.'+str(i))
# ax1.legend()
# #ax1.set_yscale('log')
# ax1.set_xlabel('Nbr. of function evaluations')
# ax1.set_ylabel('Best function evaluation')
# fig1.savefig('Flowsheeting_plots/FS_Nesterov_Convergence_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
for i in range(len(FS_SQSnobFit_list)):
    x_best = np.array(FS_SQSnobFit_list[i]['x_best_so_far'])
    f_best = np.array(FS_SQSnobFit_list[i]['f_best_so_far'])
    x_ind = np.array(FS_SQSnobFit_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'SQSnobfit.'+str(i))
ax1.legend()
# ax1.set_yscale('log')
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
fig1.savefig('Flowsheeting_plots/FS_SQSnobFit_Convergence_plot.svg', format = "svg")


fig1 = plt.figure()
ax1 = fig1.add_subplot()
for i in range(len(FS_DIRECT_list)):
    x_best = np.array(FS_DIRECT_list[i]['x_best_so_far'])
    f_best = np.array(FS_DIRECT_list[i]['f_best_so_far'])
    x_ind = np.array(FS_DIRECT_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'DIRECT'+str(i))
ax1.legend()
# ax1.set_yscale('log')
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
fig1.savefig('Flowsheeting_plots/FS_DIRECT_Convergence_plot.svg', format = "svg")


sol_Cg = average_from_list(FS_CUATRO_global_list)
test_CUATROg, test_av_CUATROg, test_min_CUATROg, test_max_CUATROg = sol_Cg
sol_Cl = average_from_list(FS_CUATRO_local_list)
test_CUATROl, test_av_CUATROl, test_min_CUATROl, test_max_CUATROl = sol_Cl
# sol_Nest = average_from_list(FS_Nest_list)
# test_Nest, test_av_Nest, test_min_Nest, test_max_Nest = sol_Nest
sol_Splx = average_from_list(FS_simplex_list)
test_Splx, test_av_Splx, test_min_Splx, test_max_Splx = sol_Splx
sol_SQSF = average_from_list(FS_SQSnobFit_list)
test_SQSF, test_av_SQSF, test_min_SQSF, test_max_SQSF = sol_SQSF
sol_DIR = average_from_list(FS_DIRECT_list)
test_DIR, test_av_DIR, test_min_DIR, test_max_DIR = sol_DIR
sol_pybbyqa = average_from_list(FS_pybobyqa_list)
test_pybbqa, test_av_pybbqa, test_min_pybbqa, test_max_pybbqa = sol_pybbyqa
sol_findiff = average_from_list(FS_FiniteDiff_list)
test_findiff, test_av_findiff, test_min_findiff, test_max_findiff = sol_findiff
sol_BFGS = average_from_list(FS_BFGS_list)
test_BFGS, test_av_BFGS, test_min_BFGS, test_max_BFGS = sol_BFGS
sol_Adam = average_from_list(FS_Adam_list)
test_Adam, test_av_Adam, test_min_Adam, test_max_Adam = sol_Adam
sol_BO = average_from_list(FS_Bayes_list)
test_BO, test_av_BO, test_min_BO, test_max_BO = sol_BO



fig = plt.figure()
ax = fig.add_subplot()
ax.step(np.arange(1, 101), test_av_CUATROg, where = 'post', label = 'CUATRO_g', c = 'b')
ax.fill_between(np.arange(1, 101), test_min_CUATROg, \
                test_max_CUATROg, color = 'b', alpha = .5)
ax.step(np.arange(1, 101), test_av_CUATROl, where = 'post', label = 'CUATRO_l', c = 'c')
ax.fill_between(np.arange(1, 101), test_min_CUATROl, \
                test_max_CUATROl, color = 'c', alpha = .5)
ax.step(np.arange(1, 101), test_av_pybbqa, where = 'post', label = 'Py-BOBYQA ', c = 'green')
ax.fill_between(np.arange(1, 101), test_min_pybbqa, \
                test_max_pybbqa, color = 'green', alpha = .5)
ax.step(np.arange(1, 101), test_av_SQSF, where = 'post', label = 'Snobfit', c = 'orange')
ax.fill_between(np.arange(1, 101), test_min_SQSF, \
                test_max_SQSF, color = 'orange', alpha = .5)
ax.step(np.arange(1, 101), test_av_BO, where = 'post', label = 'Bayes. Opt.', c = 'red')
ax.fill_between(np.arange(1, 101), test_min_BO, \
                test_max_BO, color = 'red', alpha = .5)

ax.legend()
# ax.set_yscale('log')
ax.set_xlim([1, 100])    
ax.set_ylim([156.5, 161]) 
ax.set_xlabel('Nbr. of function evaluations')
ax.set_ylabel('Best function evaluation')
ax.legend(loc = 'upper right')
fig.savefig('Flowsheeting_plots/FS_Model.svg', format = "svg")


fig = plt.figure()
ax = fig.add_subplot()
# ax.step(np.arange(1, 101), test_av_Nest, where = 'post', label = 'Nesterov', c = 'brown')
# ax.fill_between(np.arange(1, 101), test_min_Nest, \
#                 test_max_Nest, color = 'brown', alpha = .5)
ax.step(np.arange(1, 101), test_av_Splx, where = 'post', label = 'Simplex', c = 'green')
ax.fill_between(np.arange(1, 101), test_min_Splx, \
                test_max_Splx, color = 'green', alpha = .5)
ax.step(np.arange(1, 101), test_av_findiff, where = 'post', label = 'Newton', c = 'grey')
ax.fill_between(np.arange(1, 101), test_min_findiff, \
                test_max_findiff, color = 'grey', alpha = .5)
ax.step(np.arange(1, 101), test_av_BFGS, where = 'post', label = 'BFGS', c = 'orange')
ax.fill_between(np.arange(1, 101), test_min_BFGS, \
                test_max_BFGS, color = 'orange', alpha = .5)
ax.step(np.arange(1, 101), test_av_Adam, where = 'post', label = 'Adam ', c = 'blue')
ax.fill_between(np.arange(1, 101), test_min_Adam, \
                test_max_Adam, color = 'blue', alpha = .5)
ax.step(np.arange(1, 101), test_av_DIR, where = 'post', label = 'DIRECT', c = 'violet')
ax.fill_between(np.arange(1, 101), test_min_DIR, \
                test_max_DIR, color = 'violet', alpha = .5)

ax.legend()
# ax.set_yscale('log')
ax.set_xlim([1, 100])
ax.set_ylim([156.5, 161]) 
ax.set_xlabel('Nbr. of function evaluations')
ax.set_ylabel('Best function evaluation')
ax.legend(loc = 'upper right')
fig.savefig('Flowsheeting_plots/FS_Others.svg', format = "svg")


def count_feasible_sampling(inpt, list_input = True, threshold = 0):
    if list_input:
        count = 0 ; N_tot = 0
        N = len(inpt)
        best = 0
        for i in range(N):
            arr = np.array(inpt[i]['g_store'])
            count += np.sum(np.product((arr <= threshold).astype(int), axis = 1))
            N_tot += len(arr)
            best += inpt[i]['f_best_so_far'][-1] / N
        return count/N_tot, best
    else:
        arr = np.array(inpt['g_store'])
        N_tot = len(arr)
        count = np.sum(np.product((arr <= threshold).astype(int), axis = 1))
        best = inpt['f_best_so_far'][-1]
        i = -1
        while best == np.inf:
            best = inpt['f_best_so_far'][i-1]
            i -= 1
        return count/N_tot, best
       
FS_constr_list = []
FS_constr_list.append(['Adam', count_feasible_sampling(FS_Adam, list_input = False)])
FS_constr_list.append(['Bayes', count_feasible_sampling(FS_Bayes_list, threshold = 1e-3)])
FS_constr_list.append(['BFGS', count_feasible_sampling(FS_BFGS, list_input = False)])
FS_constr_list.append(['CUATROg', count_feasible_sampling(FS_CUATRO_global_list)])
FS_constr_list.append(['CUATROl', count_feasible_sampling(FS_CUATRO_local_list)])
FS_constr_list.append(['DIRECT', count_feasible_sampling(FS_DIRECT_list)])
FS_constr_list.append(['Newton', count_feasible_sampling(FS_FiniteDiff, list_input = False)])
FS_constr_list.append(['PyBOBYQA', count_feasible_sampling(FS_pybobyqa_list[0], list_input = False)])
FS_constr_list.append(['Simplex', count_feasible_sampling(FS_simplex_list)])
FS_constr_list.append(['SQSnobfit', count_feasible_sampling(FS_SQSnobFit_list)])

with open('FS_constraintSat_list.pickle', 'wb') as handle:
    pickle.dump(FS_constr_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
print(FS_constr_list)








