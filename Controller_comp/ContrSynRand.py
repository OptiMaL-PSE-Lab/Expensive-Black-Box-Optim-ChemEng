# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 01:21:42 2021

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

def scaled_to_absolute(x, bounds):
    absolute = bounds[:,0] + np.array(x)*(bounds[:,1] - bounds[:,0])
    return absolute

def absolute_to_scaled(x, bounds):
    scaled = (np.array(x) - bounds[:,0])/(bounds[:,1] - bounds[:,0])
    return scaled

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

def plot_reactor_respRand(pi, plot, method, bounds, noise, x0 = [.6, 310], 
                          xref = [.666, 308.489], N=200, T=8,  NS = False):
    
    ax1, ax2, ax3, ax4 = plot
    if not NS:
        _, sys_resp, control_resp = reactor_phi_2st(pi, bounds, noise, x0 = x0, N = N, \
                                    T = T, return_sys_resp = True)
    else:
        _, sys_resp, control_resp = reactor_phi_2stNS(pi, noise, x0 = x0, N = N, \
                                    T = T, return_sys_resp = True)
    
    x1 = np.array(sys_resp)[:,0] ; x2 = np.array(sys_resp)[:,1]
    ax1.plot(np.arange(len(x1))/len(x1)*T, x1, label = method + ': $x_1$')
    ax1.plot([0, T], [xref[0], xref[0]], '--k')
    ax2.plot([0, T], [xref[1], xref[1]], '--k')
    ax2.plot(np.arange(len(x2))/len(x2)*T, x2, label =  method + ': $x_2$')

    
    u1 = np.array(control_resp)[:,0] ; u2 = np.array(control_resp)[:,1]
    ax3.plot(np.arange(len(u1))/len(u1)*T, u1, label = method + ': $u_1$')
    ax4.plot(np.arange(len(u2))/len(u2)*T, u2, label = method + ': $u_2$')
    return ax1, ax2, ax3, ax4

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

N_SAA = 1

noise = [.001, 1]

cost_rand = lambda x: cost_control_noise(x, bounds_abs, noise, N_SAA, \
                                    x0 = [.116, 368.489], N = 200, T = 20)
ContrSynRand_DIRECT_cost = lambda x, grad: cost_rand(x)

bounds = np.array([[0, 1]]*10)
 
x0 = (np.array(pi_init) - bounds_abs[:,0]) / (bounds_abs[:,1]-bounds_abs[:,0]) 



x0_abs = np.array(pi_init)

cost_randNS = lambda x: cost_control_noise(x, bounds_abs, noise, N_SAA, \
                                    x0 = [.116, 368.489], N = 200, T = 20, NS = True)

max_f_eval = 100

N = 10
ContrSynRand_pybobyqa_list = []
for i in range(N):
    rnd_seed = i
    ContrSynRand_pybobyqa = PyBobyqaWrapper().solve(cost_rand, x0, bounds=bounds.T, \
                                      maxfun= max_f_eval, constraints=1, \
                                      seek_global_minimum = True)
    ContrSynRand_pybobyqa_list.append(ContrSynRand_pybobyqa)
print('10 Py-BOBYQA iterations completed')
    
N = 10
ContrSynRand_simplex_list = []
for i in range(N):
    rnd_seed = i
    ContrSynRand_simplex = simplex_method(cost_rand, x0, bounds, max_iter = 50, \
                            constraints = 1, rnd_seed = i, mu_con = 1e6)
    ContrSynRand_simplex_list.append(ContrSynRand_simplex)
print('10 simplex iterations completed')

N = 10
ContrSynRand_FiniteDiff_list = []
for i in range(N):
    rnd_seed = i
    ContrSynRand_FiniteDiff = finite_Diff_Newton(cost_randNS, x0_abs, bounds = bounds_abs, \
                                    con_weight = 1e6)
    ContrSynRand_FiniteDiff_list.append(ContrSynRand_FiniteDiff)
print('10 Approx Newton iterations completed')

N = 10
ContrSynRand_BFGS_list = []
for i in range(N):
    rnd_seed = i
    ContrSynRand_BFGS = BFGS_optimizer(cost_randNS, x0_abs, bounds = bounds_abs, \
                                con_weight = 1e6)
    ContrSynRand_BFGS_list.append(ContrSynRand_BFGS)
print('10 BFGS iterations completed')

    
N = 10
ContrSynRand_Adam_list = []
for i in range(N):
    rnd_seed = i
    ContrSynRand_Adam = Adam_optimizer(cost_rand, x0, method = 'forward', \
                                      bounds = bounds, alpha = 0.4, \
                                      beta1 = 0.2, beta2  = 0.1, \
                                      max_f_eval = 100, con_weight = 1e6)
    ContrSynRand_Adam_list.append(ContrSynRand_Adam)
print('10 Adam iterations completed')
 
N_min_s = 15
init_radius = 0.5
method = 'Discrimination'
N = 10
ContrSynRand_CUATRO_global_list = []
for i in range(N):
    rnd_seed = i
    ContrSynRand_CUATRO_global = CUATRO(cost_rand, x0, init_radius, bounds = bounds, \
                          N_min_samples = N_min_s, tolerance = 1e-10,\
                          beta_red = 0.9, rnd = rnd_seed, method = 'global', \
                          constr_handling = method)
    ContrSynRand_CUATRO_global_list.append(ContrSynRand_CUATRO_global)
print('10 CUATRO global iterations completed')      
 
N_min_s = 6
init_radius = 0.5
# method = 'Fitting'
method = 'Discrimination'
N = 10
ContrSynRand_CUATRO_local_list = []
for i in range(N):
    rnd_seed = i
    ContrSynRand_CUATRO_local = CUATRO(cost_rand, x0, init_radius, bounds = bounds, \
                          N_min_samples = N_min_s, tolerance = 1e-10,\
                          beta_red = 0.9, rnd = rnd_seed, method = 'local', \
                          constr_handling = method)
    ContrSynRand_CUATRO_local_list.append(ContrSynRand_CUATRO_local)
print('10 CUATRO local iterations completed') 

N = 10
ContrSynRand_SQSnobFit_list = []
for i in range(N):
    ContrSynRand_SQSnobFit = SQSnobFitWrapper().solve(cost_randNS, x0_abs, bounds_abs, mu_con = 1e6, \
                                    maxfun = max_f_eval, constraints=1)
    ContrSynRand_SQSnobFit_list.append(ContrSynRand_SQSnobFit)
print('10 SnobFit iterations completed') 

N = 10
ContrSynRand_DIRECT_list = []
for i in range(N):
    ContrSynRand_DIRECT =  DIRECTWrapper().solve(ContrSynRand_DIRECT_cost, x0, bounds, mu_con = 1e6, \
                                    maxfun = max_f_eval, constraints=1)
    ContrSynRand_DIRECT_list.append(ContrSynRand_DIRECT)
print('10 DIRECT iterations completed')    





fig1 = plt.figure()
ax1 = fig1.add_subplot()
for i in range(len(ContrSynRand_pybobyqa_list)):
    x_best = np.array(ContrSynRand_pybobyqa_list[i]['x_best_so_far'])
    f_best = np.array(ContrSynRand_pybobyqa_list[i]['f_best_so_far'])
    x_ind = np.array(ContrSynRand_pybobyqa_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'Py-BOBYQA'+str(i))
    # ax1.plot(x_ind, f_best, label = 'CUATRO_g'+str(i))
ax1.legend()
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
ax1.set_yscale('log')
fig1.savefig('ContrSyn_random_plots/ContrSyn_PyBOBYQA_Convergence_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
for i in range(len(ContrSynRand_FiniteDiff_list)):
    x_best = np.array(ContrSynRand_FiniteDiff_list[i]['x_best_so_far'])
    f_best = np.array(ContrSynRand_FiniteDiff_list[i]['f_best_so_far'])
    x_ind = np.array(ContrSynRand_FiniteDiff_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'Fin. Diff.'+str(i))
    # ax1.plot(x_ind, f_best, label = 'CUATRO_g'+str(i))
ax1.legend()
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
ax1.set_yscale('log')
fig1.savefig('ContrSyn_random_plots/ContrSyn_FiniteDiff_Convergence_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
for i in range(len(ContrSynRand_BFGS_list)):
    x_best = np.array(ContrSynRand_BFGS_list[i]['x_best_so_far'])
    f_best = np.array(ContrSynRand_BFGS_list[i]['f_best_so_far'])
    x_ind = np.array(ContrSynRand_BFGS_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'BFGS'+str(i))
    # ax1.plot(x_ind, f_best, label = 'CUATRO_g'+str(i))
ax1.legend()
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
ax1.set_yscale('log')
fig1.savefig('ContrSyn_random_plots/ContrSyn_BFGS_Convergence_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
for i in range(len(ContrSynRand_Adam_list)):
    x_best = np.array(ContrSynRand_Adam_list[i]['x_best_so_far'])
    f_best = np.array(ContrSynRand_Adam_list[i]['f_best_so_far'])
    x_ind = np.array(ContrSynRand_Adam_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'Adam'+str(i))
    # ax1.plot(x_ind, f_best, label = 'CUATRO_g'+str(i))
ax1.legend()
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
ax1.set_yscale('log')
fig1.savefig('ContrSyn_random_plots/ContrSyn_Adam_Convergence_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
for i in range(len(ContrSynRand_CUATRO_global_list)):
    x_best = np.array(ContrSynRand_CUATRO_global_list[i]['x_best_so_far'])
    f_best = np.array(ContrSynRand_CUATRO_global_list[i]['f_best_so_far'])
    x_ind = np.array(ContrSynRand_CUATRO_global_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'CUATRO_g'+str(i))
    # ax1.plot(x_ind, f_best, label = 'CUATRO_g'+str(i))
ax1.legend()
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
ax1.set_yscale('log')
fig1.savefig('ContrSyn_random_plots/ContrSyn_CUATROg_Convergence_plot.svg', format = "svg")


fig1 = plt.figure()
ax1 = fig1.add_subplot()
for i in range(len(ContrSynRand_CUATRO_local_list)):
    x_best = np.array(ContrSynRand_CUATRO_local_list[i]['x_best_so_far'])
    f_best = np.array(ContrSynRand_CUATRO_local_list[i]['f_best_so_far'])
    x_ind = np.array(ContrSynRand_CUATRO_local_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'CUATRO_l'+str(i))
    # ax1.plot(x_ind, f_best, label = 'CUATRO_l'+str(i))
ax1.legend()
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
ax1.set_yscale('log')
fig1.savefig('ContrSyn_random_plots/ContrSyn_CUATROl_Convergence_plot.svg', format = "svg")

# fig1 = plt.figure()
# ax1 = fig1.add_subplot()
# for i in range(len(ContrSynRand_Bayes_list)):
#     x_best = np.array(ContrSynRand_Bayes_list[i]['x_best_so_far'])
#     f_best = np.array(ContrSynRand_Bayes_list[i]['f_best_so_far'])
#     nbr_feval = len(ContrSynRand_Bayes_list[i]['f_store'])
#     ax1.step(np.arange(len(f_best)), f_best, where = 'post', \
#           label = 'BO'+str(i)+'; #f_eval: ' + str(nbr_feval))
# ax1.legend()
# ax1.set_xlabel('Nbr. of function evaluations')
# ax1.set_ylabel('Best function evaluation')
# ax1.set_yscale('log')
# fig1.savefig('ContrSyn_random_plots/ContrSyn_BO_Convergence_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
for i in range(len(ContrSynRand_simplex_list)):
    x_best = np.array(ContrSynRand_simplex_list[i]['x_best_so_far'])
    f_best = np.array(ContrSynRand_simplex_list[i]['f_best_so_far'])
    x_ind = np.array(ContrSynRand_simplex_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'Simplex'+str(i))
ax1.legend()
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
ax1.set_yscale('log')
fig1.savefig('ContrSyn_random_plots/ContrSyn_Simplex_Convergence_plot.svg', format = "svg")


# ## Change to x_best_So_far
# fig1 = plt.figure()
# ax1 = fig1.add_subplot()
# for i in range(len(ContrSynRand_Nest_list)):
#     x_best = np.array(ContrSynRand_Nest_list[i]['x_best_so_far'])
#     f_best = np.array(ContrSynRand_Nest_list[i]['f_best_so_far'])
#     x_ind = np.array(ContrSynRand_Nest_list[i]['samples_at_iteration'])
#     ax1.step(x_ind, f_best, where = 'post', label = 'Nest.'+str(i))
# ax1.legend()
# ax1.set_yscale('log')
# ax1.set_xlabel('Nbr. of function evaluations')
# ax1.set_ylabel('Best function evaluation')
# fig1.savefig('ContrSyn_random_plots/ContrSyn_Nesterov_Convergence_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
for i in range(len(ContrSynRand_SQSnobFit_list)):
    x_best = np.array(ContrSynRand_SQSnobFit_list[i]['x_best_so_far'])
    f_best = np.array(ContrSynRand_SQSnobFit_list[i]['f_best_so_far'])
    x_ind = np.array(ContrSynRand_SQSnobFit_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'SQSnobfit.'+str(i))
ax1.legend()
ax1.set_yscale('log')
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
fig1.savefig('ContrSyn_random_plots/ContrSyn_SQSnobFit_Convergence_plot.svg', format = "svg")


fig1 = plt.figure()
ax1 = fig1.add_subplot()
for i in range(len(ContrSynRand_DIRECT_list)):
    x_best = np.array(ContrSynRand_DIRECT_list[i]['x_best_so_far'])
    f_best = np.array(ContrSynRand_DIRECT_list[i]['f_best_so_far'])
    x_ind = np.array(ContrSynRand_DIRECT_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'DIRECT'+str(i))
ax1.legend()
ax1.set_yscale('log')
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
fig1.savefig('ContrSyn_random_plots/ContrSyn_DIRECT_Convergence_plot.svg', format = "svg")


sol_Cg = average_from_list(ContrSynRand_CUATRO_global_list)
test_CUATROg, test_av_CUATROg, test_min_CUATROg, test_max_CUATROg = sol_Cg
sol_Cl = average_from_list(ContrSynRand_CUATRO_local_list)
test_CUATROl, test_av_CUATROl, test_min_CUATROl, test_max_CUATROl = sol_Cl
# sol_Nest = average_from_list(ContrSynRand_Nest_list)
# test_Nest, test_av_Nest, test_min_Nest, test_max_Nest = sol_Nest
sol_Splx = average_from_list(ContrSynRand_simplex_list)
test_Splx, test_av_Splx, test_min_Splx, test_max_Splx = sol_Splx
sol_SQSF = average_from_list(ContrSynRand_SQSnobFit_list)
test_SQSF, test_av_SQSF, test_min_SQSF, test_max_SQSF = sol_SQSF
sol_DIR = average_from_list(ContrSynRand_DIRECT_list)
test_DIR, test_av_DIR, test_min_DIR, test_max_DIR = sol_DIR
sol_pybbyqa = average_from_list(ContrSynRand_pybobyqa_list)
test_pybbqa, test_av_pybbqa, test_min_pybbqa, test_max_pybbqa = sol_pybbyqa
sol_findiff = average_from_list(ContrSynRand_FiniteDiff_list)
test_findiff, test_av_findiff, test_min_findiff, test_max_findiff = sol_findiff
sol_BFGS = average_from_list(ContrSynRand_BFGS_list)
test_BFGS, test_av_BFGS, test_min_BFGS, test_max_BFGS = sol_BFGS
sol_Adam = average_from_list(ContrSynRand_Adam_list)
test_Adam, test_av_Adam, test_min_Adam, test_max_Adam = sol_Adam



fig = plt.figure()
ax = fig.add_subplot()
ax.step(np.arange(1, 101), test_av_CUATROg, where = 'post', label = 'CUATRO_global', c = 'b')
ax.fill_between(np.arange(1, 101), test_min_CUATROg, \
                test_max_CUATROg, color = 'b', alpha = .5)
ax.step(np.arange(1, 101), test_av_CUATROl, where = 'post', label = 'CUATRO_local', c = 'c')
ax.fill_between(np.arange(1, 101), test_min_CUATROl, \
                test_max_CUATROl, color = 'c', alpha = .5)
ax.step(np.arange(1, 101), test_av_pybbqa, where = 'post', label = 'Py-BOBYQA ', c = 'green')
ax.fill_between(np.arange(1, 101), test_min_pybbqa, \
                test_max_pybbqa, color = 'green', alpha = .5)
ax.step(np.arange(1, 101), test_av_SQSF, where = 'post', label = 'Snobfit', c = 'orange')
ax.fill_between(np.arange(1, 101), test_min_SQSF, \
                test_max_SQSF, color = 'orange', alpha = .5)
## BO placeholder: red

ax.legend()
ax.set_yscale('log')
ax.set_xlim([0, 99])    
fig.savefig('ContrSyn_random_plots/PromisingMethodsRand.svg', format = "svg")


fig = plt.figure()
ax = fig.add_subplot()
# ax.step(np.arange(1, 101), test_av_Nest, where = 'post', label = 'Nesterov', c = 'brown')
# ax.fill_between(np.arange(1, 101), test_min_Nest, \
#                 test_max_Nest, color = 'brown', alpha = .5)
ax.step(np.arange(1, 101), test_av_Splx, where = 'post', label = 'Simplex', c = 'green')
ax.fill_between(np.arange(1, 101), test_min_Splx, \
                test_max_Splx, color = 'green', alpha = .5)
ax.step(np.arange(1, 101), test_av_findiff, where = 'post', label = 'Approx. Newton', c = 'grey')
ax.fill_between(np.arange(1, 101), test_min_findiff, \
                test_max_findiff, color = 'grey', alpha = .5)
ax.step(np.arange(1, 101), test_av_BFGS, where = 'post', label = 'Approx. BFGS', c = 'orange')
ax.fill_between(np.arange(1, 101), test_min_BFGS, \
                test_max_BFGS, color = 'orange', alpha = .5)
ax.step(np.arange(1, 101), test_av_Adam, where = 'post', label = 'Adam ', c = 'blue')
ax.fill_between(np.arange(1, 101), test_min_Adam, \
                test_max_Adam, color = 'blue', alpha = .5)
ax.step(np.arange(1, 101), test_av_DIR, where = 'post', label = 'DIRECT', c = 'violet')
ax.fill_between(np.arange(1, 101), test_min_DIR, \
                test_max_DIR, color = 'violet', alpha = .5)

ax.legend()
ax.set_yscale('log')
ax.set_xlim([0, 99])
fig.savefig('ContrSyn_random_plots/NotSoPromisingMethodsRand.svg', format = "svg")



fig1, fig2, fig3 = plt.figure(), plt.figure(), plt.figure()
ax1, ax2 = fig1.add_subplot(211), fig1.add_subplot(212)
ax4, ax5 = fig3.add_subplot(211), fig3.add_subplot(212)


method = 'Init'
plot = (ax1, ax2, ax4, ax5)
plot = plot_reactor_respRand(x0, plot, method, bounds_abs, noise, x0 = [.116, 368.489], 
                             N = 200*5, T = 20)
method = 'DIRECT'
pi = ContrSynRand_DIRECT_list[9]['x_best_so_far'][-1]
plot = plot_reactor_respRand(pi, plot, method, bounds_abs, noise, x0 = [.116, 368.489],
                             N = 200*5, T = 20)
method = 'SQSnobfit'
pi = absolute_to_scaled(ContrSynRand_SQSnobFit_list[5]['x_best_so_far'][-1], bounds_abs)
plot = plot_reactor_respRand(pi, plot, method, bounds_abs, noise, x0 = [.116, 368.489],
                              N = 200*5, T = 20)
ax1.legend() ; ax2.legend() ; ax4.legend() ; ax5.legend()

fig1.savefig('ContrSyn_random_plots/ContrTrajStatesRand.svg', format = "svg")
fig3.savefig('ContrSyn_random_plots/ContrTrajControlsRand.svg', format = "svg")



fig1, fig2, fig3 = plt.figure(), plt.figure(), plt.figure()
ax1, ax2 = fig1.add_subplot(211), fig1.add_subplot(212)
ax4, ax5 = fig3.add_subplot(211), fig3.add_subplot(212)

method = 'Init'
plot = (ax1, ax2, ax4, ax5)
plot = plot_reactor_respRand(x0, plot, method, bounds_abs, [0, 0], x0 = [.116, 368.489], 
                             N = 200*5, T = 20)
method = 'DIRECT'
pi = ContrSynRand_DIRECT_list[9]['x_best_so_far'][-1]
plot = plot_reactor_respRand(pi, plot, method, bounds_abs, [0, 0], x0 = [.116, 368.489],
                             N = 200*5, T = 20)
method = 'SQSnobfit'
pi = absolute_to_scaled(ContrSynRand_SQSnobFit_list[5]['x_best_so_far'][-1], bounds_abs)
plot = plot_reactor_respRand(pi, plot, method, bounds_abs, [0, 0], x0 = [.116, 368.489],
                             N = 200*5, T = 20)
ax1.legend() ; ax2.legend() ; ax4.legend() ; ax5.legend()

fig1.savefig('ContrSyn_random_plots/ContrTrajStatesDet.svg', format = "svg")
fig3.savefig('ContrSyn_random_plots/ContrTrajControlsDet.svg', format = "svg")





