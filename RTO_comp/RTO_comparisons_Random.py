# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 20:50:15 2021

@author: dv516
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt

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
from algorithms.SQSnobfit_wrapped.Wrapper_for_SQSnobfit import SQSnobFitWrapper
from algorithms.DIRECT_wrapped.Wrapper_for_Direct import DIRECTWrapper
# from algorithms.Finite_differences.Finite_differences import BFGS_optimizer

from case_studies.RTO.systems import *

def trust_fig(X, Y, Z, g1, g2):   
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.contour(X, Y, Z, 50)
    ax.contour(X, Y, g1, levels = [0], colors = 'black')
    ax.contour(X, Y, g2, levels = [0], colors = 'black')
    
    return ax, fig

def scaling(x):
    y = np.zeros(len(x))
    y[0] = (x[0] - 4)/(7 - 4)
    y[1] = (x[1] - 70)/(100 - 70)
    return y

def extract_FT(x):
    x[0] = 4 + (7 - 4)*x[0]
    x[1] = 70 + (100 - 70)*x[1]
    return x

def RTO_rand(x):
    # x = extract_FT(x)
    plant = WO_system()
    f = plant.WO_obj_sys_ca
    g1 = plant.WO_con1_sys_ca
    g2 = plant.WO_con2_sys_ca
    return f(x), [g1(x), g2(x)]


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

## Apply scaling transformation

# bounds = np.array([[0., 1.],[0., 1.]])
# x0 = np.array([0.967, 0.433])

x0 = [6.9, 83]
bounds  = np.array([[4., 7.], [70., 100.]])

max_f_eval = 100
max_it = 100


N = 10
RTORand_pybbqa_list = []
for i in range(N):
    RTORand_pybobyqa = PyBobyqaWrapper().solve(RTO_rand, x0, bounds=bounds.T, \
                                              maxfun= max_f_eval, constraints=2, \
                                              seek_global_minimum = True, \
                                              scaling_within_bounds = True, \
                                              mu_con = 1e6)
    RTORand_pybbqa_list.append(RTORand_pybobyqa)   
print('10 Py-BOBYQA iterations completed')
# RTO_pybobyqa = PyBobyqaWrapper().solve(RTO, x0, bounds=bounds.T, \
#                                       maxfun= max_f_eval, constraints=2)
# print(x0)
N = 10
RTORand_Nest_list = []
for i in range(N):
    rnd_seed = i
    RTORand_Nest = nesterov_random(RTO_rand, x0, bounds, max_iter = 50, \
                               constraints = 2, rnd_seed = i, alpha = 1e-4)
    RTORand_Nest_list.append(RTORand_Nest)
print('10 Nesterov iterations completed')
# print(x0)


N = 10
RTORand_simplex_list = []
for i in range(N):
    rnd_seed = i
    RTORand_simplex = simplex_method(RTO_rand, x0, bounds, max_iter = 50, \
                            constraints = 2, rnd_seed = i, mu_con = 1e6)
    RTORand_simplex_list.append(RTORand_simplex)
print('10 simplex iterations completed')
# print(x0)

N = 10
RTORand_findiff_list = []
for i in range(N):
    RTORand_FiniteDiff = finite_Diff_Newton(RTO_rand, x0, bounds = bounds, \
                                   con_weight = 100)
    RTORand_findiff_list.append(RTORand_FiniteDiff)
print('10 Approx Newton iterations completed')
    
N = 10
RTORand_Adam_list = []
for i in range(N):
    RTORand_Adam = Adam_optimizer(RTO_rand, x0, method = 'forward', \
                                      bounds = bounds, alpha = 0.4, \
                                      beta1 = 0.2, beta2  = 0.1, \
                                      max_f_eval = 100, con_weight = 100)
    RTORand_Adam_list.append(RTORand_Adam)
print('10 Adam iterations completed')


# print(x0) 
N_min_s = 15
# init_radius = 0.5
init_radius = 10
method = 'Discrimination'
N = 10
RTORand_CUATRO_global_list = []
for i in range(N):
    rnd_seed = i
    RTORand_CUATRO_global = CUATRO(RTO_rand, x0, init_radius, bounds = bounds, \
                          N_min_samples = N_min_s, tolerance = 1e-10,\
                          beta_red = 0.9, rnd = rnd_seed, method = 'global', \
                          constr_handling = method)
    RTORand_CUATRO_global_list.append(RTORand_CUATRO_global)
print('10 CUATRO global iterations completed')    
# print(x0)    

N = 10
RTORand_SQSnobFit_list = []
for i in range(N):
    RTORand_SQSnobFit = SQSnobFitWrapper().solve(RTO_rand, x0, bounds, \
                                   maxfun = max_f_eval, constraints=2, \
                                   mu_con = 1e6)
    RTORand_SQSnobFit_list.append(RTORand_SQSnobFit)
print('10 SnobFit iterations completed') 

N = 10
RTORand_DIRECT_list = []
RTO_DIRECT_f = lambda x, grad: RTO_rand(x)
for i in range(N):
    RTORand_DIRECT =  DIRECTWrapper().solve(RTO_DIRECT_f, x0, bounds, \
                                   maxfun = max_f_eval, constraints=2, \
                                   mu_con = 1e6)
    RTORand_DIRECT_list.append(RTORand_DIRECT)
print('10 DIRECT iterations completed')     


N_min_s = 6
# init_radius = 0.05
init_radius = 1
method = 'Fitting'
N = 10
RTORand_CUATRO_local_list = []
for i in range(N):
    rnd_seed = i
    RTORand_CUATRO_local = CUATRO(RTO_rand, x0, init_radius, bounds = bounds, \
                          N_min_samples = N_min_s, tolerance = 1e-10,\
                          beta_red = 0.9, rnd = rnd_seed, method = 'local', \
                          constr_handling = method)
    RTORand_CUATRO_local_list.append(RTORand_CUATRO_local)
print('10 CUATRO local iterations completed') 
# print(x0)

with open('BayesRTO_listRand.pickle', 'rb') as handle:
    RTORand_Bayes_list = pickle.load(handle)


# N = 10
# RTORand_Bayes_list = []
# for i in range(1):
#     Bayes = BayesOpt()
#     pyro.set_rng_seed(i)
    
#     if i<3:
#         nbr_feval = 40
#     elif i<6:
#         nbr_feval = 30
#     else:
#         nbr_feval = 20
    
#     RTORand_Bayes = Bayes.solve(RTO_rand, x0, acquisition='EI',bounds=bounds.T, \
#                             print_iteration = True, constraints=2, casadi=True, \
#                             maxfun = nbr_feval, ).output_dict
#     RTORand_Bayes_list.append(RTORand_Bayes)
 
# print('10 BayesOpt iterations completed')

# with open('BayesRTO_listRand.pickle', 'wb') as handle:
#     pickle.dump(RTORand_Bayes_list, handle, protocol=pickle.HIGHEST_PROTOCOL)


plant = WO_system()

x = np.linspace(4, 7, 50)
y = np.linspace(70, 100, 50)
X, Y = np.meshgrid(x, y)
Z = np.zeros((50, 50))
g1 = np.zeros((50, 50))
g2 = np.zeros((50, 50))
for i in range(len(x)):
    for j in range(len(y)):
        
        Z[j][i] = plant.WO_obj_sys_ca_noise_less([x[i], y[j]])
        g1[j][i] = plant.WO_con1_sys_ca_noise_less([x[i], y[j]])
        g2[j][i] = plant.WO_con2_sys_ca_noise_less([x[i], y[j]])

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(X, Y, Z, g1, g2)
for i in range(len(RTORand_pybbqa_list)):
    x_best = np.array(RTORand_pybbqa_list[i]['x_best_so_far'])
    f_best = np.array(RTORand_pybbqa_list[i]['f_best_so_far'])
    x_ind = np.array(RTORand_pybbqa_list[i]['samples_at_iteration'])
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
fig1.savefig('RTO_Random_plots/RTO_pybbqa_Convergence_plot.svg', format = "svg")
fig2.savefig('RTO_Random_plots/RTO_pybbqa_2Dspace_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(X, Y, Z, g1, g2)
for i in range(len(RTORand_findiff_list)):
    x_best = np.array(RTORand_findiff_list[i]['x_best_so_far'])
    f_best = np.array(RTORand_findiff_list[i]['f_best_so_far'])
    x_ind = np.array(RTORand_findiff_list[i]['samples_at_iteration'])
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
fig1.savefig('RTO_Random_plots/RTO_FinDiff_Convergence_plot.svg', format = "svg")
fig2.savefig('RTO_Random_plots/RTO_FinDiff_2Dspace_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(X, Y, Z, g1, g2)
for i in range(len(RTORand_Adam_list)):
    x_best = np.array(RTORand_Adam_list[i]['x_best_so_far'])
    f_best = np.array(RTORand_Adam_list[i]['f_best_so_far'])
    x_ind = np.array(RTORand_Adam_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'Adam'+str(i))
    # ax1.plot(x_ind, f_best, label = 'CUATRO_g'+str(i))
    ax2.plot(x_best[:,0], x_best[:,1], '--', label = 'Adam'+str(i))
ax1.legend()
# ax1.set_yscale('log')
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.legend()
ax2.set_xlim(bounds[0])
ax2.set_ylim(bounds[1])
fig1.savefig('RTO_Random_plots/RTO_Adam_Convergence_plot.svg', format = "svg")
fig2.savefig('RTO_Random_plots/RTO_Adam_2Dspace_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(X, Y, Z, g1, g2)
for i in range(len(RTORand_CUATRO_global_list)):
    x_best = np.array(RTORand_CUATRO_global_list[i]['x_best_so_far'])
    f_best = np.array(RTORand_CUATRO_global_list[i]['f_best_so_far'])
    x_ind = np.array(RTORand_CUATRO_global_list[i]['samples_at_iteration'])
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
fig1.savefig('RTO_Random_plots/RTO_CUATROg_Convergence_plot.svg', format = "svg")
fig2.savefig('RTO_Random_plots/RTO_CUATROg_2Dspace_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(X, Y, Z, g1, g2)
for i in range(len(RTORand_CUATRO_local_list)):
    x_best = np.array(RTORand_CUATRO_local_list[i]['x_best_so_far'])
    f_best = np.array(RTORand_CUATRO_local_list[i]['f_best_so_far'])
    x_ind = np.array(RTORand_CUATRO_local_list[i]['samples_at_iteration'])
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
fig1.savefig('RTO_Random_plots/RTO_CUATROl_Convergence_plot.svg', format = "svg")
fig2.savefig('RTO_Random_plots/RTO_CUATROl_2Dspace_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(X, Y, Z, g1, g2)
for i in range(len(RTORand_Bayes_list)):
    x_best = np.array(RTORand_Bayes_list[i]['x_best_so_far'])
    f_best = np.array(RTORand_Bayes_list[i]['f_best_so_far'])
    nbr_feval = len(RTORand_Bayes_list[i]['f_store'])
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
fig1.savefig('RTO_Random_plots/RTO_BO_Convergence_plot.svg', format = "svg")
fig2.savefig('RTO_Random_plots/RTO_BO_2Dspace_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(X, Y, Z, g1, g2)
for i in range(len(RTORand_simplex_list)):
    x_best = np.array(RTORand_simplex_list[i]['x_best_so_far'])
    f_best = np.array(RTORand_simplex_list[i]['f_best_so_far'])
    x_ind = np.array(RTORand_simplex_list[i]['samples_at_iteration'])
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
fig1.savefig('RTO_Random_plots/RTO_simplex_Convergence_plot.svg', format = "svg")
fig2.savefig('RTO_Random_plots/RTO_simplex_2Dspace_plot.svg', format = "svg")

## Change to x_best_So_far
fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(X, Y, Z, g1, g2)
for i in range(len(RTORand_Nest_list)):
    x_best = np.array(RTORand_Nest_list[i]['x_best_so_far'])
    f_best = np.array(RTORand_Nest_list[i]['f_best_so_far'])
    x_ind = np.array(RTORand_Nest_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'Nest.'+str(i))
    ax2.plot(x_best[:,0], x_best[:,1], '--', label = 'Nest.'+str(i))
ax1.legend()
# ax1.set_yscale('log')
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.set_xlim(bounds[0])
ax2.set_ylim(bounds[1])
ax2.legend()
fig1.savefig('RTO_Random_plots/RTO_Nesterov_Convergence_plot.svg', format = "svg")
fig2.savefig('RTO_Random_plots/RTO_Nesterov_2Dspace_plot.svg', format = "svg")

## Change to x_best_So_far
fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(X, Y, Z, g1, g2)
for i in range(len(RTORand_SQSnobFit_list)):
    x_best = np.array(RTORand_SQSnobFit_list[i]['x_best_so_far'])
    f_best = np.array(RTORand_SQSnobFit_list[i]['f_best_so_far'])
    x_ind = np.array(RTORand_SQSnobFit_list[i]['samples_at_iteration'])
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
fig1.savefig('RTO_Random_plots/RTO_SQSnobFit_Convergence_plot.svg', format = "svg")
fig2.savefig('RTO_Random_plots/RTO_SQSnobFit_2Dspace_plot.svg', format = "svg")

## Change to x_best_So_far
fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(X, Y, Z, g1, g2)
for i in range(len(RTORand_DIRECT_list)):
    x_best = np.array(RTORand_DIRECT_list[i]['x_best_so_far'])
    f_best = np.array(RTORand_DIRECT_list[i]['f_best_so_far'])
    x_ind = np.array(RTORand_DIRECT_list[i]['samples_at_iteration'])
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
fig1.savefig('RTO_Random_plots/RTO_DIRECT_Convergence_plot.svg', format = "svg")
fig2.savefig('RTO_Random_plots/RTO_DIRECT_2Dspace_plot.svg', format = "svg")


sol_Cg = average_from_list(RTORand_CUATRO_global_list)
test_CUATROg, test_av_CUATROg, test_min_CUATROg, test_max_CUATROg = sol_Cg
sol_Cl = average_from_list(RTORand_CUATRO_local_list)
test_CUATROl, test_av_CUATROl, test_min_CUATROl, test_max_CUATROl = sol_Cl
sol_Nest = average_from_list(RTORand_Nest_list)
test_Nest, test_av_Nest, test_min_Nest, test_max_Nest = sol_Nest
sol_Splx = average_from_list(RTORand_simplex_list)
test_Splx, test_av_Splx, test_min_Splx, test_max_Splx = sol_Splx
sol_SQSF = average_from_list(RTORand_SQSnobFit_list)
test_SQSF, test_av_SQSF, test_min_SQSF, test_max_SQSF = sol_SQSF
sol_DIR = average_from_list(RTORand_DIRECT_list)
test_DIR, test_av_DIR, test_min_DIR, test_max_DIR = sol_DIR
sol_pybbyqa = average_from_list(RTORand_pybbqa_list)
test_pybbqa, test_av_pybbqa, test_min_pybbqa, test_max_pybbqa = sol_pybbyqa
sol_findiff = average_from_list(RTORand_findiff_list)
test_findiff, test_av_findiff, test_min_findiff, test_max_findiff = sol_findiff
# sol_BFGS = average_from_list(RTORand_BFGS_list)
# test_BFGS, test_av_BFGS, test_min_BFGS, test_max_BFGS = sol_BFGS
sol_Adam = average_from_list(RTORand_Adam_list)
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
f_best = np.array(RTORand_Bayes_list[0]['f_best_so_far'])
ax.step(np.arange(len(f_best)), f_best, where = 'post', \
          label = 'BO', c = 'r')

ax.legend()
# ax.set_yscale('log')
ax.set_xlabel('Nbr. of function evaluations')
ax.set_ylabel('Best function evaluation')
ax.set_xlim([0, 99])    
fig.savefig('Publication plots/PromisingMethodsRand.svg', format = "svg")


fig = plt.figure()
ax = fig.add_subplot()
ax.step(np.arange(1, 101), test_av_Nest, where = 'post', label = 'Nesterov', c = 'brown')
ax.fill_between(np.arange(1, 101), test_min_Nest, \
                test_max_Nest, color = 'brown', alpha = .5)
ax.step(np.arange(1, 101), test_av_Splx, where = 'post', label = 'Simplex', c = 'green')
ax.fill_between(np.arange(1, 101), test_min_Splx, \
                test_max_Splx, color = 'green', alpha = .5)
ax.step(np.arange(1, 101), test_av_findiff, where = 'post', label = 'Approx. Newton', c = 'grey')
ax.fill_between(np.arange(1, 101), test_min_findiff, \
                test_max_findiff, color = 'grey', alpha = .5)
# ax.step(np.arange(1, 101), test_av_BFGS, where = 'post', label = 'Approx. BFGS', c = 'orange')
# ax.fill_between(np.arange(1, 101), test_min_BFGS, \
#                 test_max_BFGS, color = 'orange', alpha = .5)
ax.step(np.arange(1, 101), test_av_Adam, where = 'post', label = 'Adam ', c = 'blue')
ax.fill_between(np.arange(1, 101), test_min_Adam, \
                test_max_Adam, color = 'blue', alpha = .5)
ax.step(np.arange(1, 101), test_av_DIR, where = 'post', label = 'DIRECT', c = 'violet')
ax.fill_between(np.arange(1, 101), test_min_DIR, \
                test_max_DIR, color = 'violet', alpha = .5)


ax.legend()
# ax.set_yscale('log')
ax.set_xlabel('Nbr. of function evaluations')
ax.set_ylabel('Best function evaluation')
ax.set_xlim([0, 99])
fig.savefig('Publication plots/NotSoPromisingMethodsRand.svg', format = "svg")


