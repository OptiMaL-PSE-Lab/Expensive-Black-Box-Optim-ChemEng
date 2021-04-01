# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 11:13:37 2021

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

from test_functions import quadratic_constrained

import matplotlib.pyplot as plt

def trust_fig(oracle, bounds):
    N = 200
    lim = 2
    x = np.linspace(-lim, lim, N)
    y = np.linspace(-lim, lim, N)
    X,Y = np.meshgrid(x, y)
    Z = oracle.sample_obj(X,Y)
    constr = oracle.sample_constr(X,Y)
    
    levels_list = np.logspace(-2, 1.5, 10)
    
    fig = plt.figure(figsize = (6,4))
    ax = fig.add_subplot()
    
    ax.contour(X,Y,Z*constr, levels = levels_list)
    ax.plot([bounds[0,0], bounds[0, 1]], [bounds[1,0], bounds[1, 0]], c = 'k')
    ax.plot([bounds[0,0], bounds[0, 1]], [bounds[1,1], bounds[1, 1]], c = 'k')
    ax.plot([bounds[0,0], bounds[0, 0]], [bounds[1,0], bounds[1, 1]], c = 'k')
    ax.plot([bounds[0,1], bounds[0, 1]], [bounds[1,0], bounds[1, 1]], c = 'k')
    
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

class RB:
    def __init__(self, objective, ineq = []):
        self.obj = objective ; self.ieq = ineq
    def sample_obj(self, x, y):
        return self.obj(x, y)
    def sample_constr(self, x, y):
        if self.ieq == []:
            if (type(x) == float) or (type(x) == int):
                return 1
            else:
                return np.ones(len(x))
        elif (type(x) == float) or (type(x) == int):
            temporary = [int(g(x, y)) for g in self.ieq]
            return np.product(np.array(temporary))
        else:
            temporary = [g(x, y).astype(int) for g in self.ieq]
            return np.product(np.array(temporary), axis = 0)


def Problem_quadratic(x):
    f = quadratic_constrained.quadratic_f
    g = quadratic_constrained.quadratic_g

    return f(x), [g(x)]

bounds = np.array([[-1.5,1.5],[-1.5,1.5]])
x0 = np.array([1, 1])
# x0 = np.array([-0.5, 1.5])


max_f_eval = 100
max_it = 100


quadratic_pybobyqa = PyBobyqaWrapper().solve(Problem_quadratic, x0, bounds=bounds.T, \
                                      maxfun= max_f_eval, constraints=2)
# print(x0)
    
N = 10
quadratic_Nest_list = []
for i in range(N):
    rnd_seed = i
    quadratic_Nest = nesterov_random(Problem_quadratic, x0, bounds, max_iter = 50, \
                          constraints = 1, rnd_seed = i, alpha = 1e-3, mu = 1e-2)
    quadratic_Nest_list.append(quadratic_Nest)
print('10 Nesterov iterations completed')
# print(x0)

N = 10
quadratic_simplex_list = []
for i in range(N):
    rnd_seed = i
    quadratic_simplex = simplex_method(Problem_quadratic, x0, bounds, max_iter = 50, \
                            constraints = 1, rnd_seed = i)
    quadratic_simplex_list.append(quadratic_simplex)
print('10 simplex iterations completed')

quadratic_FiniteDiff = finite_Diff_Newton(Problem_quadratic, x0, bounds = bounds, \
                                   con_weight = 100)
    
quadratic_BFGS = BFGS_optimizer(Problem_quadratic, x0, bounds = bounds, \
                         con_weight = 100)
    
quadratic_Adam = Adam_optimizer(Problem_quadratic, x0, method = 'forward', \
                                      bounds = bounds, alpha = 0.4, \
                                      beta1 = 0.2, beta2  = 0.1, \
                                      max_f_eval = 100, con_weight = 100)
    
N_min_s = 15
init_radius = 2
method = 'Discrimination'
N = 10
quadratic_CUATRO_global_list = []
for i in range(N):
    rnd_seed = i
    quadratic_CUATRO_global = CUATRO(Problem_quadratic, x0, init_radius, bounds = bounds, \
                          N_min_samples = N_min_s, tolerance = 1e-10,\
                          beta_red = 0.9, rnd = rnd_seed, method = 'global', \
                          constr_handling = method)
    quadratic_CUATRO_global_list.append(quadratic_CUATRO_global)
print('10 CUATRO global iterations completed')    

N_min_s = 6
init_radius = 0.1
method = 'Fitting'
N = 10
quadratic_CUATRO_local_list = []
for i in range(N):
    rnd_seed = i
    quadratic_CUATRO_local = CUATRO(Problem_quadratic, x0, init_radius, bounds = bounds, \
                          N_min_samples = N_min_s, tolerance = 1e-10,\
                          beta_red = 0.5, rnd = rnd_seed, method = 'local', \
                          constr_handling = method)
    quadratic_CUATRO_local_list.append(quadratic_CUATRO_local)
print('10 CUATRO local iterations completed') 

N = 10
quadratic_SQSnobFit_list = []
for i in range(N):
    quadratic_SQSnobFit = SQSnobFitWrapper().solve(Problem_quadratic, x0, bounds, \
                                   maxfun = max_f_eval, constraints=2)
    quadratic_SQSnobFit_list.append(quadratic_SQSnobFit)
print('10 SnobFit iterations completed') 

N = 10
quadratic_DIRECT_list = []
quadratic_DIRECT_f = lambda x, grad: Problem_quadratic(x)
for i in range(N):
    quadratic_DIRECT =  DIRECTWrapper().solve(quadratic_DIRECT_f, x0, bounds, \
                                   maxfun = max_f_eval, constraints=2)
    quadratic_DIRECT_list.append(quadratic_DIRECT)
print('10 DIRECT iterations completed')     

# with open('BayesQuadraticTight_list.pickle', 'rb') as handle:
#     quadratic_Bayes_list = pickle.load(handle)

# N = 10
# quadratic_Bayes_list = []
# for i in range(1):
#     Bayes = BayesOpt()
#     pyro.set_rng_seed(i)
    
#     if i<3:
#         nbr_feval = 40
#     elif i<6:
#         nbr_feval = 30
#     else:
#         nbr_feval = 20
    
#     quadratic_Bayes = Bayes.solve(Problem_quadratic, x0, acquisition='EI',bounds=bounds.T, \
#                             print_iteration = True, constraints= 1, casadi=True, \
#                             maxfun = nbr_feval, ).output_dict
#     quadratic_Bayes_list.append(quadratic_Bayes)
 
# print('10 BayesOpt iterations completed')

# with open('BayesQuadraticTight_list.pickle', 'wb') as handle:
#     pickle.dump(quadratic_Bayes_list, handle, protocol=pickle.HIGHEST_PROTOCOL)



x_best_pyBbyqa = np.array(quadratic_pybobyqa['x_best_so_far'])
f_best_pyBbyqa = np.array(quadratic_pybobyqa['f_best_so_far'])
# x_ind_pyBbyqa = np.array(RB_pybobyqa['samples_at_iteration'])
# nbr_feval_pyBbyqa = len(RB_pybobyqa['f_store'])

x_best_finDiff = np.array(quadratic_FiniteDiff['x_best_so_far'])
f_best_finDiff = np.array(quadratic_FiniteDiff['f_best_so_far'])
x_ind_findDiff = np.array(quadratic_FiniteDiff['samples_at_iteration'])

x_best_BFGS = np.array(quadratic_BFGS['x_best_so_far'])
f_best_BFGS = np.array(quadratic_BFGS['f_best_so_far'])
x_ind_BFGS = np.array(quadratic_BFGS['samples_at_iteration'])

x_best_Adam = np.array(quadratic_Adam['x_best_so_far'])
f_best_Adam = np.array(quadratic_Adam['f_best_so_far'])
x_ind_Adam = np.array(quadratic_Adam['samples_at_iteration'])

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
fig1.savefig('Quadratic_plots/Quadratic_Deterministic_Convergence_plot.svg', format = "svg")

quadratic_f = lambda x, y: x**2 + 10*y**2 + x*y
quadratic_g = lambda x, y: 1 - x - y <= 0

oracle = RB(quadratic_f, ineq = [quadratic_g])

ax2, fig2 = trust_fig(oracle, bounds)
ax2.plot(x_best_pyBbyqa[:,0], x_best_pyBbyqa[:,1], '--x', \
         label = 'PyBobyqa')
ax2.plot(x_best_finDiff[:,0], x_best_finDiff[:,1], '--x', \
         label = 'Newton Fin. Diff.')
ax2.plot(x_best_BFGS[:,0], x_best_BFGS[:,1], '--x', \
         label = 'BFGS')
ax2.plot(x_best_Adam[:,0], x_best_Adam[:,1], '--x', \
         label = 'Adam')
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.legend()
ax2.set_xlim(bounds[0])
ax2.set_ylim(bounds[1])
fig2.savefig('Quadratic_plots/Quadratic_Deterministic_2Dspace_plot.svg', format = "svg")


fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(oracle, bounds)
for i in range(len(quadratic_CUATRO_global_list)):
    x_best = np.array(quadratic_CUATRO_global_list[i]['x_best_so_far'])
    f_best = np.array(quadratic_CUATRO_global_list[i]['f_best_so_far'])
    x_ind = np.array(quadratic_CUATRO_global_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'CUATRO_g'+str(i))
    # ax1.plot(x_ind, f_best, label = 'CUATRO_g'+str(i))
    ax2.plot(x_best[:,0], x_best[:,1], '--', label = 'CUATRO_g'+str(i))
ax1.legend()
ax1.set_yscale('log')
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.legend()
ax2.set_xlim(bounds[0])
ax2.set_ylim(bounds[1])
fig1.savefig('Quadratic_plots/Quadratic_CUATROg_Convergence_plot.svg', format = "svg")
fig2.savefig('Quadratic_plots/Quadratic_CUATROg_2Dspace_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(oracle, bounds)
for i in range(len(quadratic_CUATRO_local_list)):
    x_best = np.array(quadratic_CUATRO_local_list[i]['x_best_so_far'])
    f_best = np.array(quadratic_CUATRO_local_list[i]['f_best_so_far'])
    x_ind = np.array(quadratic_CUATRO_local_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'CUATRO_l'+str(i))
    # ax1.plot(x_ind, f_best, label = 'CUATRO_l'+str(i))
    ax2.plot(x_best[:,0], x_best[:,1], '--', label = 'CUATRO_l'+str(i))
ax1.legend()
ax1.set_yscale('log')
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.legend()
ax2.set_xlim(bounds[0])
ax2.set_ylim(bounds[1])
fig1.savefig('Quadratic_plots/Quadratic_CUATROl_Convergence_plot.svg', format = "svg")
fig2.savefig('Quadratic_plots/Quadratic_CUATROl_2Dspace_plot.svg', format = "svg")

# fig1 = plt.figure()
# ax1 = fig1.add_subplot()
# ax2, fig2 = trust_fig(oracle, bounds)
# for i in range(len(quadratic_Bayes_list)):
#     x_best = np.array(quadratic_Bayes_list[i]['x_best_so_far'])
#     f_best = np.array(quadratic_Bayes_list[i]['f_best_so_far'])
#     nbr_feval = len(quadratic_Bayes_list[i]['f_store'])
#     ax1.step(np.arange(len(f_best)), f_best, where = 'post', \
#           label = 'BO'+str(i)+'; #f_eval: ' + str(nbr_feval))
#     ax2.plot(x_best[:,0], x_best[:,1], '--', \
#           label = 'BO'+str(i)+'; #f_eval: ' + str(nbr_feval))
# ax1.legend()
# ax1.set_yscale('log')
# ax1.set_xlabel('Nbr. of function evaluations')
# ax1.set_ylabel('Best function evaluation')
# ax2.set_xlabel('$x_1$')
# ax2.set_ylabel('$x_2$')
# ax2.legend()
# ax2.set_xlim(bounds[0])
# ax2.set_ylim(bounds[1])
# fig1.savefig('RB_plots/quadratic_BO_Convergence_plot.svg', format = "svg")
# fig2.savefig('RB_plots/quadratic_BO_2Dspace_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(oracle, bounds)
for i in range(len(quadratic_simplex_list)):
    x_best = np.array(quadratic_simplex_list[i]['x_best_so_far'])
    f_best = np.array(quadratic_simplex_list[i]['f_best_so_far'])
    x_ind = np.array(quadratic_simplex_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'Simplex'+str(i))
    ax2.plot(x_best[:,0], x_best[:,1], '--', label = 'Simplex'+str(i))
ax1.legend()
ax1.set_yscale('log')
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.set_xlim(bounds[0])
ax2.set_ylim(bounds[1])
ax2.legend()
fig1.savefig('Quadratic_plots/Quadratic_simplex_Convergence_plot.svg', format = "svg")
fig2.savefig('Quadratic_plots/Quadratic_simplex_2Dspace_plot.svg', format = "svg")

## Change to x_best_So_far
fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(oracle, bounds)
for i in range(N):
    x_best = np.array(quadratic_Nest_list[i]['x_best_so_far'])
    f_best = np.array(quadratic_Nest_list[i]['f_best_so_far'])
    x_ind = np.array(quadratic_Nest_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'Nest.'+str(i))
    ax2.plot(x_best[:,0], x_best[:,1], '--', label = 'Nest.'+str(i))
ax1.legend()
ax1.set_yscale('log')
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.set_xlim(bounds[0])
ax2.set_ylim(bounds[1])
ax2.legend()
fig1.savefig('Quadratic_plots/Quadratic_Nesterov_Convergence_plot.svg', format = "svg")
fig2.savefig('Quadratic_plots/Quadratic_Nesterov_2Dspace_plot.svg', format = "svg")

## Change to x_best_So_far
fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(oracle, bounds)
for i in range(N):
    x_best = np.array(quadratic_SQSnobFit_list[i]['x_best_so_far'])
    f_best = np.array(quadratic_SQSnobFit_list[i]['f_best_so_far'])
    x_ind = np.array(quadratic_SQSnobFit_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'Snobfit'+str(i))
    ax2.plot(x_best[:,0], x_best[:,1], '--', label = 'Snobfit'+str(i))
ax1.legend()
ax1.set_yscale('log')
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.set_xlim(bounds[0])
ax2.set_ylim(bounds[1])
ax2.legend()
fig1.savefig('Quadratic_plots/Quadratic_SQSnobFit_Convergence_plot.svg', format = "svg")
fig2.savefig('Quadratic_plots/Quadratic_SQSnobFit_2Dspace_plot.svg', format = "svg")

## Change to x_best_So_far
fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(oracle, bounds)
for i in range(N):
    x_best = np.array(quadratic_DIRECT_list[i]['x_best_so_far'])
    f_best = np.array(quadratic_DIRECT_list[i]['f_best_so_far'])
    x_ind = np.array(quadratic_DIRECT_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'DIRECT'+str(i))
    ax2.plot(x_best[:,0], x_best[:,1], '--', label = 'DIRECT'+str(i))
ax1.legend()
ax1.set_yscale('log')
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.set_xlim(bounds[0])
ax2.set_ylim(bounds[1])
ax2.legend()
fig1.savefig('Quadratic_plots/Quadratic_DIRECT_Convergence_plot.svg', format = "svg")
fig2.savefig('Quadratic_plots/Quadratic_DIRECT_2Dspace_plot.svg', format = "svg")


sol_Cg = average_from_list(quadratic_CUATRO_global_list)
test_CUATROg, test_av_CUATROg, test_min_CUATROg, test_max_CUATROg = sol_Cg
sol_Cl = average_from_list(quadratic_CUATRO_local_list)
test_CUATROl, test_av_CUATROl, test_min_CUATROl, test_max_CUATROl = sol_Cl
sol_Nest = average_from_list(quadratic_Nest_list)
test_Nest, test_av_Nest, test_min_Nest, test_max_Nest = sol_Nest
sol_Splx = average_from_list(quadratic_simplex_list)
test_Splx, test_av_Splx, test_min_Splx, test_max_Splx = sol_Splx
sol_SQSF = average_from_list(quadratic_SQSnobFit_list)
test_SQSF, test_av_SQSF, test_min_SQSF, test_max_SQSF = sol_SQSF
sol_DIR = average_from_list(quadratic_DIRECT_list)
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
# f_best = np.array(quadratic_Bayes_list[0]['f_best_so_far'])
# ax.step(np.arange(len(f_best)), f_best, where = 'post', \
#           label = 'BO', c = 'r')

ax.legend()
ax.set_xlabel('Nbr. of function evaluations')
ax.set_ylabel('Best function evaluation')
ax.set_yscale('log')
ax.set_xlim([0, 99])   
fig.savefig('Quadratic_publication_plots/PromisingMethods.svg', format = "svg")
 
    
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
fig.savefig('Quadratic_publication_plots/NotSoPromisingMethods.svg', format = "svg")


