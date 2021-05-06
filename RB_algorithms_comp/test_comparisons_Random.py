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

from test_functions import rosenbrock_constrained, quadratic_constrained

import matplotlib.pyplot as plt

import pickle

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

def trust_fig(oracle, bounds):
    N = 200
    lim = 2
    x = np.linspace(-lim, lim, N)
    y = np.linspace(-lim, lim, N)
    X,Y = np.meshgrid(x, y)
    Z = oracle.sample_obj(X,Y)
    constr = oracle.sample_constr(X,Y)

    level_list = np.logspace(-0.5, 4, 10)

    fig = plt.figure(figsize = (6,4))
    ax = fig.add_subplot()
    
    ax.contour(X,Y,Z*constr, levels = level_list)
    ax.plot([bounds[0,0], bounds[0, 1]], [bounds[1,0], bounds[1, 0]], c = 'k')
    ax.plot([bounds[0,0], bounds[0, 1]], [bounds[1,1], bounds[1, 1]], c = 'k')
    ax.plot([bounds[0,0], bounds[0, 0]], [bounds[1,0], bounds[1, 1]], c = 'k')
    ax.plot([bounds[0,1], bounds[0, 1]], [bounds[1,0], bounds[1, 1]], c = 'k')
    
    return ax, fig


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

def Problem_rosenbrockRand(x):
    f1 = rosenbrock_constrained.rosenbrock_f
    f_noise = np.random.normal(0, 0.05)
    g1 = rosenbrock_constrained.rosenbrock_g1
    g2 = rosenbrock_constrained.rosenbrock_g2
    g_noise1 = np.random.normal(0, 0.02)
    g_noise2 = np.random.normal(0, 0.02)

    return f1(x) + f_noise, [g1(x) + g_noise1, g2(x) + g_noise2]

bounds = np.array([[-1.5,1.5],[-1.5,1.5]])
x0 = np.array([-0.5,1.5])

max_f_eval = 100
max_it = 50

initial_outputRand = Problem_rosenbrockRand(x0)

N = 10
RBRand_pybbqa_list = []
for i in range(N):
    RBRand_pybobyqa = PyBobyqaWrapper().solve(Problem_rosenbrockRand, x0, bounds=bounds.T, \
                                              maxfun= max_f_eval, constraints=2, \
                                              seek_global_minimum = True, \
                                              objfun_has_noise= True)
    RBRand_pybbqa_list.append(RBRand_pybobyqa)   
print('10 Py-BOBYQA iterations completed')

N = 10
RBRand_Nest_list = []
for i in range(N):
    rnd_seed = i
    RBRand_Nest = nesterov_random(Problem_rosenbrockRand, x0, bounds, max_iter = 50, \
                          constraints = 2, rnd_seed = i, alpha = 1e-4)
    RBRand_Nest_list.append(RBRand_Nest)
print('10 Nesterov iterations completed')

N = 10
RBRand_simplex_list = []
for i in range(N):
    rnd_seed = i
    RBRand_simplex = simplex_method(Problem_rosenbrockRand, x0, bounds, max_iter = 50, \
                            constraints = 2, rnd_seed = i)
    RBRand_simplex_list.append(RBRand_simplex)
print('10 simplex iterations completed')

N = 10
RBRand_findiff_list = []
for i in range(N):
    RBRand_FiniteDiff = finite_Diff_Newton(Problem_rosenbrockRand, x0, bounds = bounds, \
                                   con_weight = 100)
    RBRand_findiff_list.append(RBRand_FiniteDiff)
print('10 Approx Newton iterations completed')
    
N = 10
RBRand_BFGS_list = []
for i in range(N):
    RBRand_BFGS = BFGS_optimizer(Problem_rosenbrockRand, x0, bounds = bounds, \
                         con_weight = 100)
    RBRand_BFGS_list.append(RBRand_BFGS)
print('10 BFGS iterations completed')
    
N = 10
RBRand_Adam_list = []
for i in range(N):
    RBRand_Adam = Adam_optimizer(Problem_rosenbrockRand, x0, method = 'forward', \
                                      bounds = bounds, alpha = 0.4, \
                                      beta1 = 0.2, beta2  = 0.1, \
                                      max_f_eval = 100, con_weight = 100)
    RBRand_Adam_list.append(RBRand_Adam)
print('10 Adam iterations completed')
    
N_min_s = 15
init_radius = 2
method = 'Discrimination'
N = 10
RBRand_CUATRO_global_list = []
for i in range(N):
    rnd_seed = i
    RBRand_CUATRO_global = CUATRO(Problem_rosenbrockRand, x0, init_radius, bounds = bounds, \
                          N_min_samples = N_min_s, tolerance = 1e-10,\
                          beta_red = 0.9, rnd = rnd_seed, method = 'global', \
                          constr_handling = method)
    RBRand_CUATRO_global_list.append(RBRand_CUATRO_global)
print('10 CUATRO global iterations completed')    
    
N_min_s = 6
init_radius = 0.1
method = 'Fitting'
N = 10
RBRand_CUATRO_local_list = []
for i in range(N):
    rnd_seed = i
    RBRand_CUATRO_local = CUATRO(Problem_rosenbrockRand, x0, init_radius, bounds = bounds, \
                          N_min_samples = N_min_s, tolerance = 1e-10,\
                          beta_red = 0.9, rnd = rnd_seed, method = 'local', \
                          constr_handling = method)
    RBRand_CUATRO_local_list.append(RBRand_CUATRO_local)
print('10 CUATRO local iterations completed') 

N = 10
RBRand_SQSnobFit_list = []
for i in range(N):
    RBRand_SQSnobFit = SQSnobFitWrapper().solve(Problem_rosenbrockRand, x0, bounds, \
                                   maxfun = max_f_eval, constraints=2)
    RBRand_SQSnobFit_list.append(RBRand_SQSnobFit)
print('10 SnobFit iterations completed') 

N = 10
boundsDIR = np.array([[-1.5,1],[-1,1.5]])
RBRand_DIRECT_list = []
RB_DIRECT_fRand = lambda x, grad: Problem_rosenbrockRand(x)
for i in range(N):
    RBRand_DIRECT =  DIRECTWrapper().solve(RB_DIRECT_fRand, x0, boundsDIR, \
                                   maxfun = max_f_eval, constraints=2)
    RBRand_DIRECT_list.append(RBRand_DIRECT)
print('10 DIRECT iterations completed')    


with open('BayesRB_listRand.pickle', 'rb') as handle:
    RBRand_Bayes_list = pickle.load(handle)

RBRand_Bayes_list = fix_starting_points(RBRand_Bayes_list, x0, initial_outputRand)
RBRand_DIRECT_list = fix_starting_points(RBRand_DIRECT_list, x0, initial_outputRand)
RBRand_simplex_list = fix_starting_points(RBRand_simplex_list, x0, initial_outputRand)
RBRand_pybbqa_list = fix_starting_points(RBRand_pybbqa_list, x0, initial_outputRand)

plt.rcParams["font.family"] = "Times New Roman"
ft = int(15)
font = {'size': ft}
plt.rc('font', **font)
params = {'legend.fontsize': 12.5,
              'legend.handlelength': 2}
plt.rcParams.update(params)

f_RB = lambda x, y: (1 - x)**2 + 100*(y - x**2)**2
g1_RB = lambda x, y: (x-1)**3 - y + 1 <= 0
g2_RB = lambda x, y: x + y - 1.8 <= 0

oracle = RB(f_RB, ineq = [g1_RB, g2_RB])

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(oracle, bounds)
for i in range(len(RBRand_pybbqa_list)):
    x_best = np.array(RBRand_pybbqa_list[i]['x_best_so_far'])
    f_best = np.array(RBRand_pybbqa_list[i]['f_best_so_far'])
    x_ind = np.array(RBRand_pybbqa_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'Py-BOBYQA'+str(i))
    # ax1.plot(x_ind, f_best, label = 'Py-BOBYQA'+str(i))
    ax2.plot(x_best[:,0], x_best[:,1], '--', label = 'Py-BOBYQA'+str(i))
ax1.legend()
ax1.set_yscale('log')
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.legend()
ax2.set_xlim(bounds[0])
ax2.set_ylim(bounds[1])
fig1.savefig('RB_plots_Random/RBRand_Pybbqa_Convergence_plot.svg', format = "svg")
fig2.savefig('RB_plots_Random/RBRand_Pybbqa_2Dspace_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(oracle, bounds)
for i in range(len(RBRand_findiff_list)):
    x_best = np.array(RBRand_findiff_list[i]['x_best_so_far'])
    f_best = np.array(RBRand_findiff_list[i]['f_best_so_far'])
    x_ind = np.array(RBRand_findiff_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'Approx. Newton'+str(i))
    # ax1.plot(x_ind, f_best, label = 'Approx. Newton'+str(i))
    ax2.plot(x_best[:,0], x_best[:,1], '--', label = 'Approx. Newton'+str(i))
ax1.legend()
ax1.set_yscale('log')
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.legend()
ax2.set_xlim(bounds[0])
ax2.set_ylim(bounds[1])
fig1.savefig('RB_plots_Random/RBRand_Newton_Convergence_plot.svg', format = "svg")
fig2.savefig('RB_plots_Random/RBRand_Newton_2Dspace_plot.svg', format = "svg")


fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(oracle, bounds)
for i in range(len(RBRand_BFGS_list)):
    x_best = np.array(RBRand_BFGS_list[i]['x_best_so_far'])
    f_best = np.array(RBRand_BFGS_list[i]['f_best_so_far'])
    x_ind = np.array(RBRand_BFGS_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'BFGS'+str(i))
    # ax1.plot(x_ind, f_best, label = 'BFGS'+str(i))
    ax2.plot(x_best[:,0], x_best[:,1], '--', label = 'BFGS'+str(i))
ax1.legend()
ax1.set_yscale('log')
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.legend()
ax2.set_xlim(bounds[0])
ax2.set_ylim(bounds[1])
fig1.savefig('RB_plots_Random/RBRand_BFGS_Convergence_plot.svg', format = "svg")
fig2.savefig('RB_plots_Random/RBRand_BFGS_2Dspace_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(oracle, bounds)
for i in range(len(RBRand_BFGS_list)):
    x_best = np.array(RBRand_Adam_list[i]['x_best_so_far'])
    f_best = np.array(RBRand_Adam_list[i]['f_best_so_far'])
    x_ind = np.array(RBRand_Adam_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'Adam'+str(i))
    # ax1.plot(x_ind, f_best, label = 'Adam'+str(i))
    ax2.plot(x_best[:,0], x_best[:,1], '--', label = 'Adam'+str(i))
ax1.legend()
ax1.set_yscale('log')
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.legend()
ax2.set_xlim(bounds[0])
ax2.set_ylim(bounds[1])
fig1.savefig('RB_plots_Random/RBRand_Adam_Convergence_plot.svg', format = "svg")
fig2.savefig('RB_plots_Random/RBRand_Adam_2Dspace_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(oracle, bounds)
for i in range(len(RBRand_CUATRO_global_list)):
    x_best = np.array(RBRand_CUATRO_global_list[i]['x_best_so_far'])
    f_best = np.array(RBRand_CUATRO_global_list[i]['f_best_so_far'])
    x_ind = np.array(RBRand_CUATRO_global_list[i]['samples_at_iteration'])
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
fig1.savefig('RB_plots_Random/RBRand_CUATROg_Convergence_plot.svg', format = "svg")
fig2.savefig('RB_plots_Random/RBRand_CUATROg_2Dspace_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(oracle, bounds)
for i in range(len(RBRand_CUATRO_local_list)):
    x_best = np.array(RBRand_CUATRO_local_list[i]['x_best_so_far'])
    f_best = np.array(RBRand_CUATRO_local_list[i]['f_best_so_far'])
    x_ind = np.array(RBRand_CUATRO_local_list[i]['samples_at_iteration'])
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
fig1.savefig('RB_plots_Random/RBRand_CUATROl_Convergence_plot.svg', format = "svg")
fig2.savefig('RB_plots_Random/RBRand_CUATROl_2Dspace_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(oracle, bounds)
for i in range(len(RBRand_Bayes_list)):
    x_best = np.array(RBRand_Bayes_list[i]['x_best_so_far'])
    f_best = np.array(RBRand_Bayes_list[i]['f_best_so_far'])
    nbr_feval = len(RBRand_Bayes_list[i]['f_store'])
    ax1.step(np.arange(len(f_best)), f_best, where = 'post', \
          label = 'BO'+str(i)+'; #f_eval: ' + str(nbr_feval))
    ax2.plot(x_best[:,0], x_best[:,1], '--', \
          label = 'BO'+str(i)+'; #f_eval: ' + str(nbr_feval))
ax1.legend()
ax1.set_yscale('log')
ax2.legend()
ax1.set_xlabel('Nbr. of function evaluations')
ax1.set_ylabel('Best function evaluation')
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.set_xlim(bounds[0])
ax2.set_ylim(bounds[1])
fig1.savefig('RB_plots_Random/RBRand_BO_Convergence_plot.svg', format = "svg")
fig2.savefig('RB_plots_Random/RBRand_BO_2Dspace_plot.svg', format = "svg")
    

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(oracle, bounds)
for i in range(len(RBRand_simplex_list)):
    x_best = np.array(RBRand_simplex_list[i]['x_best_so_far'])
    f_best = np.array(RBRand_simplex_list[i]['f_best_so_far'])
    x_ind = np.array(RBRand_simplex_list[i]['samples_at_iteration'])
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
fig1.savefig('RB_plots_Random/RBRand_Simplex_Convergence_plot.svg', format = "svg")
fig2.savefig('RB_plots_Random/RBRand_Simplex_2Dspace_plot.svg', format = "svg")


## Change to x_best_So_far
fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(oracle, bounds)
for i in range(len(RBRand_Nest_list)):
    x_best = np.array(RBRand_Nest_list[i]['x_best_so_far'])
    f_best = np.array(RBRand_Nest_list[i]['f_best_so_far'])
    x_ind = np.array(RBRand_Nest_list[i]['samples_at_iteration'])
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
fig1.savefig('RB_plots_Random/RBRand_Nest_Convergence_plot.svg', format = "svg")
fig2.savefig('RB_plots_Random/RBRand_Nest_2Dspace_plot.svg', format = "svg")

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(oracle, bounds)
for i in range(len(RBRand_SQSnobFit_list)):
    x_best = np.array(RBRand_SQSnobFit_list[i]['x_best_so_far'])
    f_best = np.array(RBRand_SQSnobFit_list[i]['f_best_so_far'])
    x_ind = np.array(RBRand_SQSnobFit_list[i]['samples_at_iteration'])
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
fig1.savefig('RB_plots_Random/RBRand_SQSnobFit_Convergence_plot.svg', format = "svg")
fig2.savefig('RB_plots_Random/RBRand_SQSnobFit_2Dspace_plot.svg', format = "svg")

## Change to x_best_So_far
fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2, fig2 = trust_fig(oracle, bounds)
for i in range(len(RBRand_DIRECT_list)):
    x_best = np.array(RBRand_DIRECT_list[i]['x_best_so_far'])
    f_best = np.array(RBRand_DIRECT_list[i]['f_best_so_far'])
    x_ind = np.array(RBRand_DIRECT_list[i]['samples_at_iteration'])
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
fig1.savefig('RB_plots_Random/RBRand_DIRECT_Convergence_plot.svg', format = "svg")
fig2.savefig('RB_plots_Random/RBRand_DIRECT_2Dspace_plot.svg', format = "svg")


sol_Cg = average_from_list(RBRand_CUATRO_global_list)
test_CUATROg, test_av_CUATROg, test_min_CUATROg, test_max_CUATROg = sol_Cg
sol_Cl = average_from_list(RBRand_CUATRO_local_list)
test_CUATROl, test_av_CUATROl, test_min_CUATROl, test_max_CUATROl = sol_Cl
sol_Nest = average_from_list(RBRand_Nest_list)
test_Nest, test_av_Nest, test_min_Nest, test_max_Nest = sol_Nest
sol_Splx = average_from_list(RBRand_simplex_list)
test_Splx, test_av_Splx, test_min_Splx, test_max_Splx = sol_Splx
sol_pybbyqa = average_from_list(RBRand_pybbqa_list)
test_pybbqa, test_av_pybbqa, test_min_pybbqa, test_max_pybbqa = sol_pybbyqa
sol_findiff = average_from_list(RBRand_findiff_list)
test_findiff, test_av_findiff, test_min_findiff, test_max_findiff = sol_findiff
sol_BFGS = average_from_list(RBRand_BFGS_list)
test_BFGS, test_av_BFGS, test_min_BFGS, test_max_BFGS = sol_BFGS
sol_Adam = average_from_list(RBRand_Adam_list)
test_Adam, test_av_Adam, test_min_Adam, test_max_Adam = sol_Adam
sol_SQSF = average_from_list(RBRand_SQSnobFit_list)
test_SQSF, test_av_SQSF, test_min_SQSF, test_max_SQSF = sol_SQSF
sol_DIR = average_from_list(RBRand_DIRECT_list)
test_DIR, test_av_DIR, test_min_DIR, test_max_DIR = sol_DIR
sol_BO = average_from_list(RBRand_Bayes_list)
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
ax.set_yscale('log')
ax.set_xlabel('Number of function evaluations')
ax.set_ylabel('Best function evaluation')
ax.set_xlim([1, 100])   
ax.set_ylim([0.8, 200])  
fig.savefig('Publication plots format/RBRand_Model.svg', format = "svg")
    

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
ax.set_yscale('log')
ax.set_xlabel('Number of function evaluations')
ax.set_ylabel('Best function evaluation')
ax.set_xlim([1, 100])
ax.set_ylim([0.8, 200]) 
fig.savefig('Publication plots format/RBRand_Others.svg', format = "svg")



