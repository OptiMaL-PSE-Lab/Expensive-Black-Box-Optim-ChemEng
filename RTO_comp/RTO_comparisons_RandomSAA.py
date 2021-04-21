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

def fix_starting_points(complete_list, x0, init_out):
    for i in range(len(complete_list)):
        dict_out = complete_list[i]
        f_arr = dict_out['f_best_so_far']
        N_eval = len(f_arr)
        g_arr = dict_out['g_best_so_far']
        
        for j in range(N_eval):
            if (g_arr[j] > 1e-3).any() or (init_out[0] < f_arr[j]):
               dict_out['x_best_so_far'][j] = np.array(x0)
               dict_out['f_best_so_far'][j] = init_out[0]
               dict_out['g_best_so_far'][j] = np.array(init_out[1])
        complete_list[i] = dict_out
    return complete_list

def RTO_SAA(x):
    # x = extract_FT(x)
    N_SAA = 5
    
    plant = WO_system()
    
    f = plant.WO_obj_sys_ca
    g1 = plant.WO_con1_sys_ca
    g2 = plant.WO_con2_sys_ca
    f_SAA = 0
    g1_SAA, g2_SAA = - np.inf, - np.inf
    
    for i in range(N_SAA):
        f_SAA += f(x)/N_SAA
        g1_SAA = max(g1_SAA, g1(x))
        g2_SAA = max(g2_SAA, g2(x))
    
    return f_SAA, [g1_SAA, g2_SAA]


## Apply scaling transformation

# bounds = np.array([[0., 1.],[0., 1.]])
# x0 = np.array([0.967, 0.433])

x0 = [6.9, 83]
bounds  = np.array([[4., 7.], [70., 100.]])

max_f_eval = 100
max_it = 100

initial_outputSAA = RTO_SAA(x0)

N = 10
RTOSAA_pybbqa_list = []
for i in range(N):
    RTOSAA_pybobyqa = PyBobyqaWrapper().solve(RTO_SAA, x0, bounds=bounds.T, \
                                              maxfun= max_f_eval, constraints=2, \
                                              seek_global_minimum = True, \
                                              objfun_has_noise=True, \
                                              scaling_within_bounds = True, \
                                              mu_con = 1e6)
    RTOSAA_pybbqa_list.append(RTOSAA_pybobyqa)   
print('10 Py-BOBYQA iterations completed')
# RTO_pybobyqa = PyBobyqaWrapper().solve(RTO, x0, bounds=bounds.T, \
#                                       maxfun= max_f_eval, constraints=2)
# print(x0)
N = 10
RTOSAA_Nest_list = []
for i in range(N):
    rnd_seed = i
    RTOSAA_Nest = nesterov_random(RTO_SAA, x0, bounds, max_iter = 50, \
                               constraints = 2, rnd_seed = i, alpha = 1e-4)
    RTOSAA_Nest_list.append(RTOSAA_Nest)
print('10 Nesterov iterations completed')
# print(x0)


N = 10
RTOSAA_simplex_list = []
for i in range(N):
    rnd_seed = i
    RTOSAA_simplex = simplex_method(RTO_SAA, x0, bounds, max_iter = 50, \
                            constraints = 2, rnd_seed = i, mu_con = 1e6)
    RTOSAA_simplex_list.append(RTOSAA_simplex)
print('10 simplex iterations completed')
# print(x0)

N = 10
RTOSAA_findiff_list = []
for i in range(N):
    RTOSAA_FiniteDiff = finite_Diff_Newton(RTO_SAA, x0, bounds = bounds, \
                                   con_weight = 100)
    RTOSAA_findiff_list.append(RTOSAA_FiniteDiff)
print('10 Approx Newton iterations completed')
    
N = 10
RTOSAA_Adam_list = []
for i in range(N):
    RTOSAA_Adam = Adam_optimizer(RTO_SAA, x0, method = 'forward', \
                                      bounds = bounds, alpha = 0.4, \
                                      beta1 = 0.2, beta2  = 0.1, \
                                      max_f_eval = 100, con_weight = 100)
    RTOSAA_Adam_list.append(RTOSAA_Adam)
print('10 Adam iterations completed')


# print(x0) 
N_min_s = 15
# init_radius = 0.5
init_radius = 10
method = 'Discrimination'
N = 10
RTOSAA_CUATRO_global_list = []
for i in range(N):
    rnd_seed = i
    RTOSAA_CUATRO_global = CUATRO(RTO_SAA, x0, init_radius, bounds = bounds, \
                          N_min_samples = N_min_s, tolerance = 1e-10,\
                          beta_red = 0.9, rnd = rnd_seed, method = 'global', \
                          constr_handling = method)
    RTOSAA_CUATRO_global_list.append(RTOSAA_CUATRO_global)
print('10 CUATRO global iterations completed')    
# print(x0)    

N = 10
RTOSAA_SQSnobFit_list = []
for i in range(N):
    RTOSAA_SQSnobFit = SQSnobFitWrapper().solve(RTO_SAA, x0, bounds, \
                                   maxfun = max_f_eval, constraints=2, \
                                   mu_con = 1e6)
    RTOSAA_SQSnobFit_list.append(RTOSAA_SQSnobFit)
print('10 SnobFit iterations completed') 

N = 10
RTOSAA_DIRECT_list = []
RTO_DIRECT_f = lambda x, grad: RTO_SAA(x)
for i in range(N):
    RTOSAA_DIRECT =  DIRECTWrapper().solve(RTO_DIRECT_f, x0, bounds, \
                                   maxfun = max_f_eval, constraints=2, \
                                   mu_con = 1e6)
    RTOSAA_DIRECT_list.append(RTOSAA_DIRECT)
print('10 DIRECT iterations completed')     


N_min_s = 6
# init_radius = 0.05
init_radius = 1
method = 'Fitting'
N = 10
RTOSAA_CUATRO_local_list = []
for i in range(N):
    rnd_seed = i
    RTOSAA_CUATRO_local = CUATRO(RTO_SAA, x0, init_radius, bounds = bounds, \
                          N_min_samples = N_min_s, tolerance = 1e-10,\
                          beta_red = 0.9, rnd = rnd_seed, method = 'local', \
                          constr_handling = method)
    RTOSAA_CUATRO_local_list.append(RTOSAA_CUATRO_local)
print('10 CUATRO local iterations completed') 
# print(x0)

with open('BayesRTO_listRandSAA.pickle', 'rb') as handle:
    RTOSAA_Bayes_list = pickle.load(handle)

RTOSAA_Bayes_list = fix_starting_points(RTOSAA_Bayes_list, x0, initial_outputSAA)
RTOSAA_DIRECT_list = fix_starting_points(RTOSAA_DIRECT_list, x0, initial_outputSAA)

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

plt.rcParams["font.family"] = "Times New Roman"
ft = int(15)
font = {'size': ft}
plt.rc('font', **font)
params = {'legend.fontsize': 12.5,
              'legend.handlelength': 2}
plt.rcParams.update(params)

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(X, Y, Z, g1, g2)
for i in range(len(RTOSAA_pybbqa_list)):
    x_best = np.array(RTOSAA_pybbqa_list[i]['x_best_so_far'])
    f_best = np.array(RTOSAA_pybbqa_list[i]['f_best_so_far'])
    x_ind = np.array(RTOSAA_pybbqa_list[i]['samples_at_iteration'])
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
fig1.savefig('RTO_RandomSAA_plots/RTO_pybbqa_Convergence_plot.svg', format = "svg")
fig2.savefig('RTO_RandomSAA_plots/RTO_pybbqa_2Dspace_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(X, Y, Z, g1, g2)
for i in range(len(RTOSAA_findiff_list)):
    x_best = np.array(RTOSAA_findiff_list[i]['x_best_so_far'])
    f_best = np.array(RTOSAA_findiff_list[i]['f_best_so_far'])
    x_ind = np.array(RTOSAA_findiff_list[i]['samples_at_iteration'])
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
fig1.savefig('RTO_RandomSAA_plots/RTO_FinDiff_Convergence_plot.svg', format = "svg")
fig2.savefig('RTO_RandomSAA_plots/RTO_FinDiff_2Dspace_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(X, Y, Z, g1, g2)
for i in range(len(RTOSAA_Adam_list)):
    x_best = np.array(RTOSAA_Adam_list[i]['x_best_so_far'])
    f_best = np.array(RTOSAA_Adam_list[i]['f_best_so_far'])
    x_ind = np.array(RTOSAA_Adam_list[i]['samples_at_iteration'])
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
fig1.savefig('RTO_RandomSAA_plots/RTO_Adam_Convergence_plot.svg', format = "svg")
fig2.savefig('RTO_RandomSAA_plots/RTO_Adam_2Dspace_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(X, Y, Z, g1, g2)
for i in range(len(RTOSAA_CUATRO_global_list)):
    x_best = np.array(RTOSAA_CUATRO_global_list[i]['x_best_so_far'])
    f_best = np.array(RTOSAA_CUATRO_global_list[i]['f_best_so_far'])
    x_ind = np.array(RTOSAA_CUATRO_global_list[i]['samples_at_iteration'])
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
fig1.savefig('RTO_RandomSAA_plots/RTO_CUATROg_Convergence_plot.svg', format = "svg")
fig2.savefig('RTO_RandomSAA_plots/RTO_CUATROg_2Dspace_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(X, Y, Z, g1, g2)
for i in range(len(RTOSAA_CUATRO_local_list)):
    x_best = np.array(RTOSAA_CUATRO_local_list[i]['x_best_so_far'])
    f_best = np.array(RTOSAA_CUATRO_local_list[i]['f_best_so_far'])
    x_ind = np.array(RTOSAA_CUATRO_local_list[i]['samples_at_iteration'])
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
fig1.savefig('RTO_RandomSAA_plots/RTO_CUATROl_Convergence_plot.svg', format = "svg")
fig2.savefig('RTO_RandomSAA_plots/RTO_CUATROl_2Dspace_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(X, Y, Z, g1, g2)
for i in range(len(RTOSAA_Bayes_list)):
    x_best = np.array(RTOSAA_Bayes_list[i]['x_best_so_far'])
    f_best = np.array(RTOSAA_Bayes_list[i]['f_best_so_far'])
    nbr_feval = len(RTOSAA_Bayes_list[i]['f_store'])
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
fig1.savefig('RTO_RandomSAA_plots/RTO_BO_Convergence_plot.svg', format = "svg")
fig2.savefig('RTO_RandomSAA_plots/RTO_BO_2Dspace_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(X, Y, Z, g1, g2)
for i in range(len(RTOSAA_simplex_list)):
    x_best = np.array(RTOSAA_simplex_list[i]['x_best_so_far'])
    f_best = np.array(RTOSAA_simplex_list[i]['f_best_so_far'])
    x_ind = np.array(RTOSAA_simplex_list[i]['samples_at_iteration'])
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
fig1.savefig('RTO_RandomSAA_plots/RTO_simplex_Convergence_plot.svg', format = "svg")
fig2.savefig('RTO_RandomSAA_plots/RTO_simplex_2Dspace_plot.svg', format = "svg")

## Change to x_best_So_far
fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(X, Y, Z, g1, g2)
for i in range(len(RTOSAA_Nest_list)):
    x_best = np.array(RTOSAA_Nest_list[i]['x_best_so_far'])
    f_best = np.array(RTOSAA_Nest_list[i]['f_best_so_far'])
    x_ind = np.array(RTOSAA_Nest_list[i]['samples_at_iteration'])
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
fig1.savefig('RTO_RandomSAA_plots/RTO_Nesterov_Convergence_plot.svg', format = "svg")
fig2.savefig('RTO_RandomSAA_plots/RTO_Nesterov_2Dspace_plot.svg', format = "svg")

## Change to x_best_So_far
fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(X, Y, Z, g1, g2)
for i in range(len(RTOSAA_SQSnobFit_list)):
    x_best = np.array(RTOSAA_SQSnobFit_list[i]['x_best_so_far'])
    f_best = np.array(RTOSAA_SQSnobFit_list[i]['f_best_so_far'])
    x_ind = np.array(RTOSAA_SQSnobFit_list[i]['samples_at_iteration'])
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
fig1.savefig('RTO_RandomSAA_plots/RTO_SQSnobFit_Convergence_plot.svg', format = "svg")
fig2.savefig('RTO_RandomSAA_plots/RTO_SQSnobFit_2Dspace_plot.svg', format = "svg")

## Change to x_best_So_far
fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(X, Y, Z, g1, g2)
for i in range(len(RTOSAA_DIRECT_list)):
    x_best = np.array(RTOSAA_DIRECT_list[i]['x_best_so_far'])
    f_best = np.array(RTOSAA_DIRECT_list[i]['f_best_so_far'])
    x_ind = np.array(RTOSAA_DIRECT_list[i]['samples_at_iteration'])
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
fig1.savefig('RTO_RandomSAA_plots/RTO_DIRECT_Convergence_plot.svg', format = "svg")
fig2.savefig('RTO_RandomSAA_plots/RTO_DIRECT_2Dspace_plot.svg', format = "svg")


sol_Cg = average_from_list(RTOSAA_CUATRO_global_list)
test_CUATROg, test_av_CUATROg, test_min_CUATROg, test_max_CUATROg = sol_Cg
sol_Cl = average_from_list(RTOSAA_CUATRO_local_list)
test_CUATROl, test_av_CUATROl, test_min_CUATROl, test_max_CUATROl = sol_Cl
sol_Nest = average_from_list(RTOSAA_Nest_list)
test_Nest, test_av_Nest, test_min_Nest, test_max_Nest = sol_Nest
sol_Splx = average_from_list(RTOSAA_simplex_list)
test_Splx, test_av_Splx, test_min_Splx, test_max_Splx = sol_Splx
sol_SQSF = average_from_list(RTOSAA_SQSnobFit_list)
test_SQSF, test_av_SQSF, test_min_SQSF, test_max_SQSF = sol_SQSF
sol_DIR = average_from_list(RTOSAA_DIRECT_list)
test_DIR, test_av_DIR, test_min_DIR, test_max_DIR = sol_DIR
sol_pybbyqa = average_from_list(RTOSAA_pybbqa_list)
test_pybbqa, test_av_pybbqa, test_min_pybbqa, test_max_pybbqa = sol_pybbyqa
sol_findiff = average_from_list(RTOSAA_findiff_list)
test_findiff, test_av_findiff, test_min_findiff, test_max_findiff = sol_findiff
# sol_BFGS = average_from_list(RTOSAA_BFGS_list)
# test_BFGS, test_av_BFGS, test_min_BFGS, test_max_BFGS = sol_BFGS
sol_Adam = average_from_list(RTOSAA_Adam_list)
test_Adam, test_av_Adam, test_min_Adam, test_max_Adam = sol_Adam
sol_BO = average_from_list(RTOSAA_Bayes_list)
test_BO, test_av_BO, test_min_BO, test_max_BO = sol_BO



fig = plt.figure()
ax = fig.add_subplot()
ax.step(np.arange(1, 101), test_av_CUATROg, where = 'post', label = 'CUATRO_g', c = 'b')
ax.fill_between(np.arange(1, 101), test_min_CUATROg, \
                test_max_CUATROg, color = 'b', alpha = .5)
ax.step(np.arange(1, 101), test_av_CUATROl, where = 'post', label = 'CUATRO_l', c = 'c')
ax.fill_between(np.arange(1, 101), test_min_CUATROl, \
                test_max_CUATROl, color = 'c', alpha = .5)
ax.step(np.arange(1, 101), test_av_pybbqa, where = 'post', label = 'PyBOBYQA ', c = 'green')
ax.fill_between(np.arange(1, 101), test_min_pybbqa, \
                test_max_pybbqa, color = 'green', alpha = .5)
ax.step(np.arange(1, 101), test_av_SQSF, where = 'post', label = 'Snobfit', c = 'orange')
ax.fill_between(np.arange(1, 101), test_min_SQSF, \
                test_max_SQSF, color = 'orange', alpha = .5)
ax.step(np.arange(1, 101), test_av_BO, where = 'post', \
          label = 'BO', c = 'r')
ax.fill_between(np.arange(1, 101), test_min_BO, \
                test_max_BO, color = 'r', alpha = .5)
ax.legend()
# ax.set_yscale('log')
ax.set_xlabel('Nbr. of function evaluations')
ax.set_ylabel('Best function evaluation')
ax.set_xlim([1, 100])    
fig.savefig('Publication plots/RTOSAA_Model.svg', format = "svg")



fig = plt.figure()
ax = fig.add_subplot()
ax.step(np.arange(1, 101), test_av_Nest, where = 'post', label = 'Nesterov', c = 'brown')
ax.fill_between(np.arange(1, 101), test_min_Nest, \
                test_max_Nest, color = 'brown', alpha = .5)
ax.step(np.arange(1, 101), test_av_Splx, where = 'post', label = 'Simplex', c = 'green')
ax.fill_between(np.arange(1, 101), test_min_Splx, \
                test_max_Splx, color = 'green', alpha = .5)
ax.step(np.arange(1, 101), test_av_findiff, where = 'post', label = 'Newton', c = 'grey')
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
ax.legend(loc = 'upper right')
ax.set_xlim([1, 100])
fig.savefig('Publication plots/RTOSAA_Others.svg', format = "svg")


