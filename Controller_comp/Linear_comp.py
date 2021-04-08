# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 00:14:34 2021

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

from case_studies.Controller_tuning.Control_system import phi

import matplotlib.pyplot as plt


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

def plot_sys_resp(pi, plot, method, x0 = [15, 15], xref = [10, 10], N=200, T=3):
    _, sys_resp, control_resp = phi(pi, x0 = x0, N = N, \
                                    T = T, return_sys_resp = True)
    ax1, ax2 = plot
    x1 = np.array(sys_resp)[:,0] ; x2 = np.array(sys_resp)[:,1]
    u1 = np.array(control_resp)[:,0] ; u2 = np.array(control_resp)[:,1]
    ax1.plot(np.arange(len(x1))/len(x1)*T, x1, label = method + ': $x_1$')
    # ax1.plot([0, T], [10, 10], '--k', label = 'Steady-state for $x_1$ and $x_2$')
    ax1.plot(np.arange(len(x2))/len(x2)*T, x2, label =  method + ': $x_2$')
    ax2.plot(np.arange(len(u1))/len(u1)*T, u1, label = method + ': $u_1$')
    ax2.plot(np.arange(len(u1))/len(u1)*T, u2, label = method + ': $u_2$')
    return ax1, ax2

x0 = np.array([4, 4, 4, 4])
bounds = np.array([[0, 8], [0, 8], [0, 8], [0, 8]])

max_f_eval = 100

ContrLin_pybobyqa = PyBobyqaWrapper().solve(phi, x0, bounds=bounds.T, \
                                      maxfun= max_f_eval, constraints=1, \
                                      seek_global_minimum = True)


N = 10
ContrLin_Nest_list = []
for i in range(N):
    rnd_seed = i
    ContrLin_Nest = nesterov_random(phi, x0, bounds, max_iter = 100, \
                          constraints = 1, rnd_seed = i, alpha = 1e-5, mu = 1e-1, max_f_eval= max_f_eval)
    ContrLin_Nest_list.append(ContrLin_Nest)
print('10 Nesterov iterations completed')

N = 10
ContrLin_simplex_list = []
for i in range(N):
    rnd_seed = i
    ContrLin_simplex = simplex_method(phi, x0, bounds, max_iter = 100, \
                            constraints = 1, rnd_seed = i, max_f_eval= max_f_eval)
    ContrLin_simplex_list.append(ContrLin_simplex)
print('10 simplex iterations completed')

ContrLin_FiniteDiff = finite_Diff_Newton(phi, x0, bounds = bounds, \
                                    con_weight = 100)
    
ContrLin_BFGS = BFGS_optimizer(phi, x0, bounds = bounds, \
                          con_weight = 100)
    
ContrLin_Adam = Adam_optimizer(phi, x0, method = 'forward', \
                                      bounds = bounds, alpha = 0.4, \
                                      beta1 = 0.2, beta2  = 0.1, \
                                      max_f_eval = 100, con_weight = 100)
    
N_min_s = 15
init_radius = 4
method = 'Discrimination'
N = 10
ContrLin_CUATRO_global_list = []
for i in range(N):
    rnd_seed = i
    ContrLin_CUATRO_global = CUATRO(phi, x0, init_radius, bounds = bounds, \
                          N_min_samples = N_min_s, tolerance = 1e-10,\
                          beta_red = 0.9, rnd = rnd_seed, method = 'global', \
                          constr_handling = method)
    ContrLin_CUATRO_global_list.append(ContrLin_CUATRO_global)
print('10 CUATRO global iterations completed')    
    
N_min_s = 6
init_radius = 0.5
method = 'Fitting'
N = 10
ContrLin_CUATRO_local_list = []
for i in range(N):
    rnd_seed = i
    ContrLin_CUATRO_local = CUATRO(phi, x0, init_radius, bounds = bounds, \
                          N_min_samples = N_min_s, tolerance = 1e-10,\
                          beta_red = 0.9, rnd = rnd_seed, method = 'local', \
                          constr_handling = method)
    ContrLin_CUATRO_local_list.append(ContrLin_CUATRO_local)
print('10 CUATRO local iterations completed') 

N = 10
ContrLin_SQSnobFit_list = []
for i in range(N):
    try:
        ContrLin_SQSnobFit = SQSnobFitWrapper().solve(phi, x0, bounds, \
                                    maxfun = max_f_eval, constraints=1)
        ContrLin_SQSnobFit_list.append(ContrLin_SQSnobFit)
    except:
        print('SQSnobfit iteration ', i, ' failed')
print('10 SnobFit iterations completed') 

N = 10
ContrLin_DIRECT_list = []
ContrLin_DIRECT_f = lambda x, grad:phi(x)
for i in range(N):
    ContrLin_DIRECT =  DIRECTWrapper().solve(ContrLin_DIRECT_f, x0, bounds, \
                                    maxfun = max_f_eval, constraints=1)
    ContrLin_DIRECT_list.append(ContrLin_DIRECT)
print('10 DIRECT iterations completed')     

with open('BayesContrLin_list.pickle', 'rb') as handle:
    ContrLin_Bayes_list = pickle.load(handle)


x_best_pyBbyqa = np.array(ContrLin_pybobyqa['x_best_so_far'])
f_best_pyBbyqa = np.array(ContrLin_pybobyqa['f_best_so_far'])
# x_ind_pyBbyqa = np.array(RB_pybobyqa['samples_at_iteration'])
# nbr_feval_pyBbyqa = len(RB_pybobyqa['f_store'])

x_best_finDiff = np.array(ContrLin_FiniteDiff['x_best_so_far'])
f_best_finDiff = np.array(ContrLin_FiniteDiff['f_best_so_far'])
x_ind_findDiff = np.array(ContrLin_FiniteDiff['samples_at_iteration'])

x_best_BFGS = np.array(ContrLin_BFGS['x_best_so_far'])
f_best_BFGS = np.array(ContrLin_BFGS['f_best_so_far'])
x_ind_BFGS = np.array(ContrLin_BFGS['samples_at_iteration'])

x_best_Adam = np.array(ContrLin_Adam['x_best_so_far'])
f_best_Adam = np.array(ContrLin_Adam['f_best_so_far'])
x_ind_Adam = np.array(ContrLin_Adam['samples_at_iteration'])


fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax1.step(np.arange(len(f_best_pyBbyqa)), f_best_pyBbyqa, where = 'post', \
          label = 'PyBobyqa')
ax1.step(x_ind_findDiff, f_best_finDiff, where = 'post', \
          label = 'Newton Fin. Diff.')
ax1.step(x_ind_BFGS, f_best_BFGS, where = 'post', \
          label = 'BFGS')
ax1.step(x_ind_Adam, f_best_Adam, where = 'post', \
          label = 'Adam')
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
ax1.legend()
ax1.set_yscale('log')
fig1.savefig('Controller_plots/Controller_Deterministic_Convergence_plot.svg', format = "svg")



fig1 = plt.figure()
ax1 = fig1.add_subplot()

for i in range(len(ContrLin_CUATRO_global_list)):
    x_best = np.array(ContrLin_CUATRO_global_list[i]['x_best_so_far'])
    f_best = np.array(ContrLin_CUATRO_global_list[i]['f_best_so_far'])
    x_ind = np.array(ContrLin_CUATRO_global_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'CUATRO_g'+str(i))
    # ax1.plot(x_ind, f_best, label = 'CUATRO_g'+str(i))
ax1.legend()
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
ax1.set_yscale('log')
fig1.savefig('Controller_plots/Controller_CUATROg_Convergence_plot.svg', format = "svg")


fig1 = plt.figure()
ax1 = fig1.add_subplot()
for i in range(len(ContrLin_CUATRO_local_list)):
    x_best = np.array(ContrLin_CUATRO_local_list[i]['x_best_so_far'])
    f_best = np.array(ContrLin_CUATRO_local_list[i]['f_best_so_far'])
    x_ind = np.array(ContrLin_CUATRO_local_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'CUATRO_l'+str(i))
    # ax1.plot(x_ind, f_best, label = 'CUATRO_l'+str(i))
ax1.legend()
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
ax1.set_yscale('log')
fig1.savefig('Controller_plots/Controller_CUATROl_Convergence_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
for i in range(len(ContrLin_Bayes_list)):
    x_best = np.array(ContrLin_Bayes_list[i]['x_best_so_far'])
    f_best = np.array(ContrLin_Bayes_list[i]['f_best_so_far'])
    nbr_feval = len(ContrLin_Bayes_list[i]['f_store'])
    ax1.step(np.arange(len(f_best)), f_best, where = 'post', \
          label = 'BO'+str(i)+'; #f_eval: ' + str(nbr_feval))
ax1.legend()
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
ax1.set_yscale('log')
fig1.savefig('Controller_plots/Controller_BO_Convergence_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
for i in range(len(ContrLin_simplex_list)):
    x_best = np.array(ContrLin_simplex_list[i]['x_best_so_far'])
    f_best = np.array(ContrLin_simplex_list[i]['f_best_so_far'])
    x_ind = np.array(ContrLin_simplex_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'Simplex'+str(i))
ax1.legend()
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
ax1.set_yscale('log')
fig1.savefig('Controller_plots/Controller_Simplex_Convergence_plot.svg', format = "svg")


## Change to x_best_So_far
fig1 = plt.figure()
ax1 = fig1.add_subplot()
for i in range(len(ContrLin_Nest_list)):
    x_best = np.array(ContrLin_Nest_list[i]['x_best_so_far'])
    f_best = np.array(ContrLin_Nest_list[i]['f_best_so_far'])
    x_ind = np.array(ContrLin_Nest_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'Nest.'+str(i))
ax1.legend()
ax1.set_yscale('log')
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
fig1.savefig('Controller_plots/Controller_Nesterov_Convergence_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
for i in range(len(ContrLin_SQSnobFit_list)):
    x_best = np.array(ContrLin_SQSnobFit_list[i]['x_best_so_far'])
    f_best = np.array(ContrLin_SQSnobFit_list[i]['f_best_so_far'])
    x_ind = np.array(ContrLin_SQSnobFit_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'Nest.'+str(i))
ax1.legend()
ax1.set_yscale('log')
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
fig1.savefig('Controller_plots/Controller_SQSnobFit_Convergence_plot.svg', format = "svg")


fig1 = plt.figure()
ax1 = fig1.add_subplot()
for i in range(len(ContrLin_DIRECT_list)):
    x_best = np.array(ContrLin_DIRECT_list[i]['x_best_so_far'])
    f_best = np.array(ContrLin_DIRECT_list[i]['f_best_so_far'])
    x_ind = np.array(ContrLin_DIRECT_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'Nest.'+str(i))
ax1.legend()
ax1.set_yscale('log')
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
fig1.savefig('Controller_plots/Controller_DIRECT_Convergence_plot.svg', format = "svg")


sol_Cg = average_from_list(ContrLin_CUATRO_global_list)
test_CUATROg, test_av_CUATROg, test_min_CUATROg, test_max_CUATROg = sol_Cg
sol_Cl = average_from_list(ContrLin_CUATRO_local_list)
test_CUATROl, test_av_CUATROl, test_min_CUATROl, test_max_CUATROl = sol_Cl
sol_Nest = average_from_list(ContrLin_Nest_list)
test_Nest, test_av_Nest, test_min_Nest, test_max_Nest = sol_Nest
sol_Splx = average_from_list(ContrLin_simplex_list)
test_Splx, test_av_Splx, test_min_Splx, test_max_Splx = sol_Splx
sol_SQSF = average_from_list(ContrLin_SQSnobFit_list)
test_SQSF, test_av_SQSF, test_min_SQSF, test_max_SQSF = sol_SQSF
sol_DIR = average_from_list(ContrLin_DIRECT_list)
test_DIR, test_av_DIR, test_min_DIR, test_max_DIR = sol_DIR


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
f_best = np.array(ContrLin_Bayes_list[0]['f_best_so_far'])
ax.step(np.arange(len(f_best)), f_best, where = 'post', \
          label = 'BO', c = 'r')

ax.legend()
ax.set_xlabel('Nbr. of function evaluations')
ax.set_ylabel('Best function evaluation')
ax.set_yscale('log')
ax.set_xlim([0, 99])   
fig.savefig('Controller_publication_plots/PromisingMethods.svg', format = "svg")
 
    
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
ax.step(x_ind_BFGS, f_best_BFGS, where = 'post', \
          label = 'BFGS', c = 'orange')
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
ax.set_yscale('log')
ax.set_xlim([0, 99])
fig.savefig('Controller_publication_plots/NotSoPromisingMethods.svg', format = "svg")

## Plots_test

# T = 3
# pi = [7.636737498696908, 1.8007378706654067, 4.319855283471685, 7.197230390528715]

# fig = plt.figure()
# fig1 = plt.figure()
# ax1 = fig.add_subplot()
# ax2 = fig1.add_subplot()
# ax1.plot([0, T], [10, 10], '--k', label = '$x_{1,ss}$ and $x_{2,ss}$')
# method = 'CUATRO_l'
# plot_stuff = (ax1, ax2)
# plot_intermediate = plot_sys_resp(pi, plot_stuff, method, x0 = [15, 15], xref = [10, 10], N=200, T=3)
# ax1, ax2 = plot_sys_resp([4]*4, plot_intermediate, 'Initial', x0 = [15, 15], xref = [10, 10], N=200, T=3)
# ax1.legend()
# ax2.legend()
# ax1.set_xlabel('T')
# ax2.set_xlabel('T')
# ax1.set_ylabel('x')
# ax2.set_ylabel('u')