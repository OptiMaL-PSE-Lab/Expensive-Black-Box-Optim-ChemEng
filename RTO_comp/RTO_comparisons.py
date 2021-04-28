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
from algorithms.Finite_differences.Finite_differences import BFGS_optimizer
from algorithms.SQSnobfit_wrapped.Wrapper_for_SQSnobfit import SQSnobFitWrapper
from algorithms.DIRECT_wrapped.Wrapper_for_Direct import DIRECTWrapper

from case_studies.RTO.systems import *

def trust_fig(X, Y, Z, g1, g2):   
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.contour(X, Y, Z, 50)
    ax.contour(X, Y, g1, levels = [0], colors = 'black')
    ax.contour(X, Y, g2, levels = [0], colors = 'black')
    
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

def RTO(x):
    # x = extract_FT(x)
    plant = WO_system()
    f = plant.WO_obj_sys_ca_noise_less
    g1 = plant.WO_con1_sys_ca_noise_less
    g2 = plant.WO_con2_sys_ca_noise_less
    return f(x), [g1(x), g2(x)]


x0 = [6.9, 83]
bounds  = np.array([[4., 7.], [70., 100.]])

max_f_eval = 100
max_it = 100

initial_output = RTO(x0)

RTO_pybobyqa = PyBobyqaWrapper().solve(RTO, x0, bounds=bounds.T, \
                                      maxfun= max_f_eval, constraints=2, \
                                      scaling_within_bounds = True, \
                                      objfun_has_noise=False, \
                                      mu_con = 1e6)

N = 10
RTO_Nest_list = []
for i in range(N):
    rnd_seed = i
    RTO_Nest = nesterov_random(RTO, x0, bounds, max_iter = 50, \
                                constraints = 2, rnd_seed = i, alpha = 1e-4)
    RTO_Nest_list.append(RTO_Nest)
print('10 Nesterov iterations completed')

N = 10
RTO_simplex_list = []
for i in range(N):
    rnd_seed = i
    RTO_simplex = simplex_method(RTO, x0, bounds, max_iter = 50, \
                            constraints = 2, rnd_seed = i, mu_con = 1e6)
    RTO_simplex_list.append(RTO_simplex)
print('10 simplex iterations completed')

RTO_FiniteDiff = finite_Diff_Newton(RTO, x0, bounds = bounds, \
                                    con_weight = 1e6)
 
# RTO_BFGS = BFGS_optimizer(RTO, x0, bounds = bounds, \
#                           con_weight = 100)
    
RTO_Adam = Adam_optimizer(RTO, x0, method = 'forward', \
                                      bounds = bounds, alpha = 0.4, \
                                      beta1 = 0.2, beta2  = 0.1, \
                                      max_f_eval = 100, con_weight = 1e6)
 
N_min_s = 15
init_radius = 10
method = 'Discrimination'
N = 10
RTO_CUATRO_global_list = []
for i in range(N):
    rnd_seed = i
    RTO_CUATRO_global = CUATRO(RTO, x0, init_radius, bounds = bounds, \
                          N_min_samples = N_min_s, tolerance = 1e-10,\
                          beta_red = 0.9, rnd = rnd_seed, method = 'global', \
                          constr_handling = method)
    RTO_CUATRO_global_list.append(RTO_CUATRO_global)
print('10 CUATRO global iterations completed')      
 
N_min_s = 6
init_radius = 1
method = 'Fitting'
N = 10
RTO_CUATRO_local_list = []
for i in range(N):
    rnd_seed = i
    RTO_CUATRO_local = CUATRO(RTO, x0, init_radius, bounds = bounds, \
                          N_min_samples = N_min_s, tolerance = 1e-10,\
                          beta_red = 0.9, rnd = rnd_seed, method = 'local', \
                          constr_handling = method)
    RTO_CUATRO_local_list.append(RTO_CUATRO_local)
print('10 CUATRO local iterations completed') 

N = 10
RTO_SQSnobFit_list = []
for i in range(N):
    RTO_SQSnobFit = SQSnobFitWrapper().solve(RTO, x0, bounds, mu_con = 1e6, \
                                    maxfun = max_f_eval, constraints=2)
    RTO_SQSnobFit_list.append(RTO_SQSnobFit)
print('10 SnobFit iterations completed') 

N = 10
RTO_DIRECT_list = []
RTO_DIRECT_f = lambda x, grad: RTO(x)
for i in range(N):
    RTO_DIRECT =  DIRECTWrapper().solve(RTO_DIRECT_f, x0, bounds, mu_con = 1e6, \
                                    maxfun = max_f_eval, constraints=2)
    RTO_DIRECT_list.append(RTO_DIRECT)
print('10 DIRECT iterations completed')    

with open('BayesRTO_list.pickle', 'rb') as handle:
    RTO_Bayes_list = pickle.load(handle)

RTO_Bayes_list = fix_starting_points(RTO_Bayes_list, x0, initial_output)
RTO_DIRECT_list = fix_starting_points(RTO_DIRECT_list, x0, initial_output)
RTO_simplex_list = fix_starting_points(RTO_simplex_list, x0, initial_output)
RTO_pybobyqa['x_best_so_far'][0] = np.array(x0)
RTO_pybobyqa['f_best_so_far'][0] = initial_output[0]
RTO_pybobyqa['g_best_so_far'][0] = np.array(initial_output[1])


plt.rcParams["font.family"] = "Times New Roman"
ft = int(15)
font = {'size': ft}
plt.rc('font', **font)
params = {'legend.fontsize': 12.5,
              'legend.handlelength': 2}
plt.rcParams.update(params)


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


x_best_pyBbyqa = np.array(RTO_pybobyqa['x_best_so_far'])
f_best_pyBbyqa = np.array(RTO_pybobyqa['f_best_so_far'])
# x_ind_pyBbyqa = np.array(RTO_pybobyqa['samples_at_iteration'])
# nbr_feval_pyBbyqa = len(RTO_pybobyqa['f_store'])

x_best_finDiff = np.array(RTO_FiniteDiff['x_best_so_far'])
f_best_finDiff = np.array(RTO_FiniteDiff['f_best_so_far'])
x_ind_findDiff = np.array(RTO_FiniteDiff['samples_at_iteration'])

# x_best_BFGS = np.array(RTO_BFGS['x_best_so_far'])
# f_best_BFGS = np.array(RTO_BFGS['f_best_so_far'])
# x_ind_BFGS = np.array(RTO_BFGS['samples_at_iteration'])

x_best_Adam = np.array(RTO_Adam['x_best_so_far'])
f_best_Adam = np.array(RTO_Adam['f_best_so_far'])
x_ind_Adam = np.array(RTO_Adam['samples_at_iteration'])

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax1.step(np.arange(len(f_best_pyBbyqa)), f_best_pyBbyqa, where = 'post', \
         label = 'PyBobyqa')
ax1.step(x_ind_findDiff, f_best_finDiff, where = 'post', \
         label = 'Newton Fin. Diff.')
# ax1.step(np.arange(len(f_best_BFGS)), f_best_BFGS, where = 'post', \
#          label = 'BFGS')
ax1.step(x_ind_Adam, f_best_Adam, where = 'post', \
         label = 'Adam')
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
ax1.legend()
# ax1.set_yscale('log')
fig1.savefig('RTO_plots/RTO_Deterministic_Convergence_plot.svg', format = "svg")

ax2, fig2 = trust_fig(X, Y, Z, g1, g2)
ax2.plot(x_best_pyBbyqa[:,0], x_best_pyBbyqa[:,1], '--x', \
         label = 'PyBobyqa')
ax2.plot(x_best_finDiff[:,0], x_best_finDiff[:,1], '--x', \
         label = 'Newton Fin. Diff.')
# ax2.plot(x_best_BFGS[:,0], x_best_BFGS[:,1], '--x', \
#          label = 'BFGS')
ax2.plot(x_best_Adam[:,0], x_best_Adam[:,1], '--x', \
         label = 'Adam')
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.legend()
ax2.set_xlim(bounds[0])
ax2.set_ylim(bounds[1])
fig2.savefig('RTO_plots/RTO_Deterministic_2Dspace_plot.svg', format = "svg")


fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(X, Y, Z, g1, g2)
for i in range(len(RTO_CUATRO_global_list)):
    x_best = np.array(RTO_CUATRO_global_list[i]['x_best_so_far'])
    f_best = np.array(RTO_CUATRO_global_list[i]['f_best_so_far'])
    x_ind = np.array(RTO_CUATRO_global_list[i]['samples_at_iteration'])
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
fig1.savefig('RTO_plots/RTO_CUATROg_Convergence_plot.svg', format = "svg")
fig2.savefig('RTO_plots/RTO_CUATROg_2Dspace_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(X, Y, Z, g1, g2)
for i in range(len(RTO_CUATRO_local_list)):
    x_best = np.array(RTO_CUATRO_local_list[i]['x_best_so_far'])
    f_best = np.array(RTO_CUATRO_local_list[i]['f_best_so_far'])
    x_ind = np.array(RTO_CUATRO_local_list[i]['samples_at_iteration'])
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
fig1.savefig('RTO_plots/RTO_CUATROl_Convergence_plot.svg', format = "svg")
fig2.savefig('RTO_plots/RTO_CUATROl_2Dspace_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(X, Y, Z, g1, g2)
for i in range(len(RTO_Bayes_list)):
    x_best = np.array(RTO_Bayes_list[i]['x_best_so_far'])
    f_best = np.array(RTO_Bayes_list[i]['f_best_so_far'])
    nbr_feval = len(RTO_Bayes_list[i]['f_store'])
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
fig1.savefig('RTO_plots/RTO_BO_Convergence_plot.svg', format = "svg")
fig2.savefig('RTO_plots/RTO_BO_2Dspace_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(X, Y, Z, g1, g2)
for i in range(len(RTO_simplex_list)):
    x_best = np.array(RTO_simplex_list[i]['x_best_so_far'])
    f_best = np.array(RTO_simplex_list[i]['f_best_so_far'])
    x_ind = np.array(RTO_simplex_list[i]['samples_at_iteration'])
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
fig1.savefig('RTO_plots/RTO_simplex_Convergence_plot.svg', format = "svg")
fig2.savefig('RTO_plots/RTO_simplex_2Dspace_plot.svg', format = "svg")

## Change to x_best_So_far
fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(X, Y, Z, g1, g2)
for i in range(len(RTO_Nest_list)):
    x_best = np.array(RTO_Nest_list[i]['x_best_so_far'])
    f_best = np.array(RTO_Nest_list[i]['f_best_so_far'])
    x_ind = np.array(RTO_Nest_list[i]['samples_at_iteration'])
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
fig1.savefig('RTO_plots/RTO_Nesterov_Convergence_plot.svg', format = "svg")
fig2.savefig('RTO_plots/RTO_Nesterov_2Dspace_plot.svg', format = "svg")

## Change to x_best_So_far
fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(X, Y, Z, g1, g2)
for i in range(len(RTO_SQSnobFit_list)):
    x_best = np.array(RTO_SQSnobFit_list[i]['x_best_so_far'])
    f_best = np.array(RTO_SQSnobFit_list[i]['f_best_so_far'])
    x_ind = np.array(RTO_SQSnobFit_list[i]['samples_at_iteration'])
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
fig1.savefig('RTO_plots/RTO_SQSnobFit_Convergence_plot.svg', format = "svg")
fig2.savefig('RTO_plots/RTO_SQSnobFit_2Dspace_plot.svg', format = "svg")

## Change to x_best_So_far
fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(X, Y, Z, g1, g2)
for i in range(len(RTO_DIRECT_list)):
    x_best = np.array(RTO_DIRECT_list[i]['x_best_so_far'])
    f_best = np.array(RTO_DIRECT_list[i]['f_best_so_far'])
    x_ind = np.array(RTO_DIRECT_list[i]['samples_at_iteration'])
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
fig1.savefig('RTO_plots/RTO_DIRECT_Convergence_plot.svg', format = "svg")
fig2.savefig('RTO_plots/RTO_DIRECT_2Dspace_plot.svg', format = "svg")


sol_Cg = average_from_list(RTO_CUATRO_global_list)
test_CUATROg, test_av_CUATROg, test_min_CUATROg, test_max_CUATROg = sol_Cg
sol_Cl = average_from_list(RTO_CUATRO_local_list)
test_CUATROl, test_av_CUATROl, test_min_CUATROl, test_max_CUATROl = sol_Cl
sol_Nest = average_from_list(RTO_Nest_list)
test_Nest, test_av_Nest, test_min_Nest, test_max_Nest = sol_Nest
sol_Splx = average_from_list(RTO_simplex_list)
test_Splx, test_av_Splx, test_min_Splx, test_max_Splx = sol_Splx
sol_SQSF = average_from_list(RTO_SQSnobFit_list)
test_SQSF, test_av_SQSF, test_min_SQSF, test_max_SQSF = sol_SQSF
sol_DIR = average_from_list(RTO_DIRECT_list)
test_DIR, test_av_DIR, test_min_DIR, test_max_DIR = sol_DIR
sol_BO = average_from_list(RTO_Bayes_list)
test_BO, test_av_BO, test_min_BO, test_max_BO = sol_BO


fig = plt.figure()
ax = fig.add_subplot()
ax.step(np.arange(1, 101), test_av_CUATROg, where = 'post', label = 'CUATRO_g', c = 'b')
ax.fill_between(np.arange(1, 101), test_min_CUATROg, \
                test_max_CUATROg, color = 'b', alpha = .5, step = 'post')
ax.step(np.arange(1, 101), test_av_CUATROl, where = 'post', label = 'CUATRO_l', c = 'c')
ax.fill_between(np.arange(1, 101), test_min_CUATROl, \
                test_max_CUATROl, color = 'c', alpha = .5, step = 'post')
ax.step(np.arange(1, 101), test_av_SQSF, where = 'post', label = 'Snobfit', c = 'orange')
ax.fill_between(np.arange(1, 101), test_min_SQSF, \
                test_max_SQSF, color = 'orange', alpha = .5, step = 'post')
ax.step(np.arange(len(f_best_pyBbyqa)), f_best_pyBbyqa, where = 'post', \
          label = 'Py-BOBYQA', c = 'green')
ax.step(np.arange(1, 101), test_av_BO, where = 'post', \
          label = 'Bayes. Opt.', c = 'r')
ax.fill_between(np.arange(1, 101), test_min_BO, \
                test_max_BO, color = 'r', alpha = .5, step = 'post')

ax.legend()
ax.set_xlabel('Number of function evaluations')
ax.set_ylabel('Best function evaluation')
# ax.set_yscale('log')
ax.set_xlim([1, 100]) 
ax.set_ylim([-85, 75])  

fig.savefig('Publication plots/RTO_Model.svg', format = "svg")
 
    
fig = plt.figure()
ax = fig.add_subplot()
ax.step(np.arange(1, 101), test_av_Nest, where = 'post', label = 'Nesterov', c = 'brown')
ax.fill_between(np.arange(1, 101), test_min_Nest, \
                test_max_Nest, color = 'brown', alpha = .5, step = 'post')
ax.step(np.arange(1, 101), test_av_Splx, where = 'post', label = 'Simplex', c = 'green')
ax.fill_between(np.arange(1, 101), test_min_Splx, \
                test_max_Splx, color = 'green', alpha = .5, step = 'post')
ax.step(x_ind_findDiff, f_best_finDiff, where = 'post', \
          label = 'Newton', c = 'black')
# ax.step(np.arange(f_best_BFGS), f_best_BFGS, where = 'post', \
#           label = 'BFGS', c = 'orange')
ax.step(x_ind_Adam, f_best_Adam, where = 'post', \
          label = 'Adam', c = 'blue')   
ax.step(np.arange(1, 101), test_av_DIR, where = 'post', label = 'DIRECT', c = 'violet')
ax.fill_between(np.arange(1, 101), test_min_DIR, \
                test_max_DIR, color = 'violet', alpha = .5, step = 'post')
# ax.boxplot(test_BO, widths = 0.1, meanline = False, showfliers = False, manage_ticks = False)
# ax.step(np.arange(1, 101), test_av_BO, where = 'post', label = 'Bayes. Opt.')

ax.legend()
ax.set_xlabel('Number of function evaluations')
ax.set_ylabel('Best function evaluation')
# ax.set_yscale('log')
ax.set_xlim([1, 100])
ax.set_ylim([-85, 75])  
ax.legend(loc = 'upper right')
fig.savefig('Publication plots/RTO_Others.svg', format = "svg")


plt.rcParams["font.family"] = "Times New Roman"
ft = int(15)
font = {'size': ft}
plt.rc('font', **font)
params = {'legend.fontsize': 12.5,
              'legend.handlelength': 2}
plt.rcParams.update(params)

CUATROl_med = medianx_from_list(RTO_CUATRO_local_list, x0)
CUATROg_med = medianx_from_list(RTO_CUATRO_global_list, x0)
Bayes_med = medianx_from_list(RTO_Bayes_list, x0)
SQSF_med = medianx_from_list(RTO_SQSnobFit_list, x0)

ax2, fig2 = trust_fig(X, Y, Z, g1, g2)
ax2.plot(CUATROl_med[:,0], CUATROl_med[:,1], label = 'CUATRO_l', \
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
fig2.savefig('Publication plots/RTO_2D_convergence_best.svg', format = "svg")


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
       
RTO_constr_list = []
RTO_constr_list.append(['Adam', count_feasible_sampling(RTO_Adam, list_input = False)])
RTO_constr_list.append(['Bayes', count_feasible_sampling(RTO_Bayes_list, threshold = 1e-3)])
RTO_constr_list.append(['CUATROg', count_feasible_sampling(RTO_CUATRO_global_list)])
RTO_constr_list.append(['CUATROl', count_feasible_sampling(RTO_CUATRO_local_list)])
RTO_constr_list.append(['DIRECT', count_feasible_sampling(RTO_DIRECT_list)])
RTO_constr_list.append(['Newton', count_feasible_sampling(RTO_FiniteDiff, list_input = False)])
RTO_constr_list.append(['Nesterov', count_feasible_sampling(RTO_Nest_list)])
RTO_constr_list.append(['PyBOBYQA', count_feasible_sampling(RTO_pybobyqa, list_input = False)])
RTO_constr_list.append(['Simplex', count_feasible_sampling(RTO_simplex_list)])
RTO_constr_list.append(['SQSnobfit', count_feasible_sampling(RTO_SQSnobFit_list)])

with open('RTO_constraintSat_list.pickle', 'wb') as handle:
    pickle.dump(RTO_constr_list, handle, protocol=pickle.HIGHEST_PROTOCOL)




