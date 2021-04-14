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
            

# def scaling(x):
#     y = np.zeros(len(x))
#     y[0] = (x[0] - 4)/(7 - 4)
#     y[1] = (x[1] - 70)/(100 - 70)
#     return y

# def extract_FT(x):
#     x[0] = 4 + (7 - 4)*x[0]
#     x[1] = 70 + (100 - 70)*x[1]
#     return x

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

RTO_pybobyqa = PyBobyqaWrapper().solve(RTO, x0, bounds=bounds.T, \
                                      maxfun= max_f_eval, constraints=2, \
                                      scaling_within_bounds = True, \
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
                                    con_weight = 100)
 
# RTO_BFGS = BFGS_optimizer(RTO, x0, bounds = bounds, \
#                           con_weight = 100)
    
RTO_Adam = Adam_optimizer(RTO, x0, method = 'forward', \
                                      bounds = bounds, alpha = 0.4, \
                                      beta1 = 0.2, beta2  = 0.1, \
                                      max_f_eval = 100, con_weight = 100)
 
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
ax.step(np.arange(1, 101), test_av_CUATROg, where = 'post', label = 'CUATRO_global', c = 'b')
ax.fill_between(np.arange(1, 101), test_min_CUATROg, \
                test_max_CUATROg, color = 'b', alpha = .5)
ax.step(np.arange(1, 101), test_av_CUATROl, where = 'post', label = 'CUATRO_local', c = 'c')
ax.fill_between(np.arange(1, 101), test_min_CUATROl, \
                test_max_CUATROl, color = 'c', alpha = .5)
ax.step(np.arange(1, 101), test_av_SQSF, where = 'post', label = 'Snobfit', c = 'orange')
ax.fill_between(np.arange(1, 101), test_min_SQSF, \
                test_max_SQSF, color = 'orange', alpha = .5)
ax.step(np.arange(len(f_best_pyBbyqa)), f_best_pyBbyqa, where = 'post', \
          label = 'PyBobyqa', c = 'green')
ax.step(np.arange(1, 101), test_av_BO, where = 'post', \
          label = 'BO', c = 'r')
ax.fill_between(np.arange(1, 101), test_min_BO, \
                test_max_BO, color = 'r', alpha = .5)

ax.legend()
ax.set_xlabel('Nbr. of function evaluations')
ax.set_ylabel('Best function evaluation')
# ax.set_yscale('log')
ax.set_xlim([0, 99])   
fig.savefig('Publication plots/PromisingMethods.svg', format = "svg")
 
    
fig = plt.figure()
ax = fig.add_subplot()
ax.step(np.arange(1, 101), test_av_Nest, where = 'post', label = 'Nesterov', c = 'brown')
ax.fill_between(np.arange(1, 101), test_min_Nest, \
                test_max_Nest, color = 'brown', alpha = .5)
ax.step(np.arange(1, 101), test_av_Splx, where = 'post', label = 'Simplex', c = 'green')
ax.fill_between(np.arange(1, 101), test_min_Splx, \
                test_max_Splx, color = 'green', alpha = .5)
ax.step(x_ind_findDiff, f_best_finDiff, where = 'post', \
          label = 'Newton Fin. Diff.', c = 'black')
# ax.step(np.arange(f_best_BFGS), f_best_BFGS, where = 'post', \
#           label = 'BFGS', c = 'orange')
ax.step(x_ind_Adam, f_best_Adam, where = 'post', \
          label = 'Adam', c = 'blue')   
ax.step(np.arange(1, 101), test_av_DIR, where = 'post', label = 'DIRECT', c = 'violet')
ax.fill_between(np.arange(1, 101), test_min_DIR, \
                test_max_DIR, color = 'violet', alpha = .5)
# ax.boxplot(test_BO, widths = 0.1, meanline = False, showfliers = False, manage_ticks = False)
# ax.step(np.arange(1, 101), test_av_BO, where = 'post', label = 'Bayes. Opt.')

ax.legend()
ax.set_xlabel('Nbr. of function evaluations')
ax.set_ylabel('Best function evaluation')
# ax.set_yscale('log')
ax.set_xlim([0, 99])
fig.savefig('Publication plots/NotSoPromisingMethods.svg', format = "svg")



