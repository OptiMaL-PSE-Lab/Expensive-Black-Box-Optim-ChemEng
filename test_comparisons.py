# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 11:13:37 2021

@author: dv516
"""

import numpy as np

import pyro
pyro.enable_validation(True)  # can help with debugging
pyro.set_rng_seed(1)

from algorithms.PyBobyqa_wrapped.Wrapper_for_pybobyqa import PyBobyqaWrapper
from algorithms.Bayesian_opt_Pyro.utilities_full import BayesOpt
from algorithms.nesterov_random.nesterov_random import nesterov_random
from algorithms.simplex.simplex_method import simplex_method
from algorithms.CUATRO import CUATRO
from algorithms.Finite_differences import finite_Diff_Newton, Adam_optimizer, BFGS_optimizer

from test_functions import rosenbrock_constrained, quadratic_constrained

import matplotlib.pyplot as plt

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
    
    return ax


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


def Problem_rosenbrock(x):
    f1 = rosenbrock_constrained.rosenbrock_f
    g1 = rosenbrock_constrained.rosenbrock_g1
    g2 = rosenbrock_constrained.rosenbrock_g2

    return f1(x), [g1(x), g2(x)]

bounds = np.array([[-1.5,1.5],[-1.5,1.5]])
x0 = np.array([-0.5,1.5])

max_f_eval = 100
max_it = 50


RB_pybobyqa = PyBobyqaWrapper().solve(Problem_rosenbrock, x0, bounds=bounds.T, \
                                      maxfun= max_f_eval, constraints=2)

N = 10
RB_Nest_list = []
for i in range(N):
    rnd_seed = i
    RB_Nest = nesterov_random(Problem_rosenbrock, x0, bounds, max_iter = 50, \
                          constraints = 2, rnd_seed = i, alpha = 1e-4)
    RB_Nest_list.append(RB_Nest)
print('10 Nesterov iterations completed')

N = 10
RB_simplex_list = []
for i in range(N):
    rnd_seed = i
    RB_simplex = simplex_method(Problem_rosenbrock, x0, bounds, max_iter = 50, \
                            constraints = 2, rnd_seed = i)
    RB_simplex_list.append(RB_simplex)
print('10 simplex iterations completed')

RB_FiniteDiff = finite_Diff_Newton(Problem_rosenbrock, x0, bounds = bounds, \
                                   con_weight = 100)
    
RB_BFGS = BFGS_optimizer(Problem_rosenbrock, x0, bounds = bounds, \
                         con_weight = 100)
    
RB_Adam = Adam_optimizer(Problem_rosenbrock, x0, method = 'forward', \
                                      bounds = bounds, alpha = 0.4, \
                                      beta1 = 0.2, beta2  = 0.1, \
                                      max_f_eval = 100, con_weight = 100)
    
N_min_s = 15
init_radius = 2
method = 'Discrimination'
N = 10
RB_CUATRO_global_list = []
for i in range(N):
    rnd_seed = i
    RB_CUATRO_global = CUATRO(Problem_rosenbrock, x0, init_radius, bounds = bounds, \
                          N_min_samples = N_min_s, tolerance = 1e-10,\
                          beta_red = 0.9, rnd = rnd_seed, method = 'global', \
                          constr_handling = method)
    RB_CUATRO_global_list.append(RB_CUATRO_global)
print('10 CUATRO global iterations completed')    
    
N_min_s = 6
init_radius = 0.1
method = 'Fitting'
N = 10
RB_CUATRO_local_list = []
for i in range(N):
    rnd_seed = i
    RB_CUATRO_local = CUATRO(Problem_rosenbrock, x0, init_radius, bounds = bounds, \
                          N_min_samples = N_min_s, tolerance = 1e-10,\
                          beta_red = 0.9, rnd = rnd_seed, method = 'local', \
                          constr_handling = method)
    RB_CUATRO_local_list.append(RB_CUATRO_local)
print('10 CUATRO local iterations completed') 

N = 10
RB_Bayes_list = []
for i in range(N):
    Bayes = BayesOpt()
    pyro.set_rng_seed(i)
    RB_Bayes = Bayes.solve(Problem_rosenbrock, x0, acquisition='EI',bounds=bounds.T, \
                        print_iteration = True, constraints=2, casadi=True, maxfun = 20).output_dict
    RB_Bayes_list.append(RB_Bayes)
 
print('10 BayesOpt iterations completed') 


x_best_pyBbyqa = np.array(RB_pybobyqa['x_best_so_far'])
f_best_pyBbyqa = np.array(RB_pybobyqa['f_best_so_far'])
nbr_feval_pyBbyqa = len(RB_pybobyqa['f_store'])

x_best_finDiff = np.array(RB_FiniteDiff['x_best_so_far'])
f_best_finDiff = np.array(RB_FiniteDiff['f_best_so_far'])
nbr_feval_finDiff = len(RB_FiniteDiff['f_store'])

x_best_BFGS = np.array(RB_BFGS['x_best_so_far'])
f_best_BFGS = np.array(RB_BFGS['f_best_so_far'])
nbr_feval_BFGS = len(RB_BFGS['f_store'])

x_best_Adam = np.array(RB_Adam['x_best_so_far'])
f_best_Adam = np.array(RB_Adam['f_best_so_far'])
nbr_feval_Adam = len(RB_Adam['f_store'])


fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax1.plot(np.arange(len(f_best_pyBbyqa)), f_best_pyBbyqa, \
         label = 'PyBobyqa; #f_eval: ' + str(nbr_feval_pyBbyqa))
ax1.plot(np.arange(len(f_best_finDiff)), f_best_finDiff, \
         label = 'Newton Fin. Diff.; #f_eval: ' + str(nbr_feval_finDiff))
ax1.plot(np.arange(len(f_best_BFGS)), f_best_BFGS, \
         label = 'BFGS; #f_eval: ' + str(nbr_feval_BFGS))
ax1.plot(np.arange(len(f_best_Adam)), f_best_Adam, \
         label = 'Adam; #f_eval: ' + str(nbr_feval_Adam))
ax1.legend()
ax1.set_yscale('log')

f_RB = lambda x, y: (1 - x)**2 + 100*(y - x**2)**2
g1_RB = lambda x, y: (x-1)**3 - y + 1 <= 0
g2_RB = lambda x, y: x + y - 1.8 <= 0

oracle = RB(f_RB, ineq = [g1_RB, g2_RB])

ax2 = trust_fig(oracle, bounds)
ax2.plot(x_best_pyBbyqa[:,0], x_best_pyBbyqa[:,1], '--x', \
         label = 'PyBobyqa; #f_eval: ' + str(nbr_feval_pyBbyqa))
ax2.plot(x_best_finDiff[:,0], x_best_finDiff[:,1], '--x', \
         label = 'Newton Fin. Diff.; #f_eval: ' + str(nbr_feval_finDiff))
ax2.plot(x_best_BFGS[:,0], x_best_BFGS[:,1], '--x', \
         label = 'BFGS; #f_eval: ' + str(nbr_feval_BFGS))
ax2.plot(x_best_Adam[:,0], x_best_Adam[:,1], '--x', \
         label = 'Adam; #f_eval: ' + str(nbr_feval_Adam))
ax2.legend()
ax2.set_xlim(bounds[0])
ax2.set_ylim(bounds[1])


fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2 = trust_fig(oracle, bounds)
for i in range(N):
    x_best = np.array(RB_CUATRO_global_list[i]['x_best_so_far'])
    f_best = np.array(RB_CUATRO_global_list[i]['f_best_so_far'])
    nbr_feval = len(RB_CUATRO_global_list[i]['f_store'])
    ax1.plot(np.arange(len(f_best)), f_best, \
         label = 'CUATRO_g'+str(i)+'; #f_eval: ' + str(nbr_feval))
    ax2.plot(x_best[:,0], x_best[:,1], '--', \
         label = 'CUATRO_g'+str(i)+'; #f_eval: ' + str(nbr_feval))
ax1.legend()
ax1.set_yscale('log')
ax2.legend()
ax2.set_xlim(bounds[0])
ax2.set_ylim(bounds[1])

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2 = trust_fig(oracle, bounds)
for i in range(N):
    x_best = np.array(RB_CUATRO_local_list[i]['x_best_so_far'])
    f_best = np.array(RB_CUATRO_local_list[i]['f_best_so_far'])
    nbr_feval = len(RB_CUATRO_local_list[i]['f_store'])
    ax1.plot(np.arange(len(f_best)), f_best, \
         label = 'CUATRO_l'+str(i)+'; #f_eval: ' + str(nbr_feval))
    ax2.plot(x_best[:,0], x_best[:,1], '--', \
         label = 'CUATRO_l'+str(i)+'; #f_eval: ' + str(nbr_feval))
ax1.legend()
ax1.set_yscale('log')
ax2.legend()
ax2.set_xlim(bounds[0])
ax2.set_ylim(bounds[1])

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2 = trust_fig(oracle, bounds)
for i in range(len(RB_Bayes_list)):
    x_best = np.array(RB_Bayes_list[i]['x_best_so_far'])
    f_best = np.array(RB_Bayes_list[i]['f_best_so_far'])
    nbr_feval = len(RB_Bayes_list[i]['f_store'])
    ax1.plot(np.arange(len(f_best)), f_best, \
         label = 'BO'+str(i)+'; #f_eval: ' + str(nbr_feval))
    ax2.plot(x_best[:,0], x_best[:,1], '--', \
         label = 'BO'+str(i)+'; #f_eval: ' + str(nbr_feval))
ax1.legend()
ax1.set_yscale('log')
ax2.legend()
ax2.set_xlim(bounds[0])
ax2.set_ylim(bounds[1])
    

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2 = trust_fig(oracle, bounds)
for i in range(N):
    x_best = np.array(RB_simplex_list[i]['x_best_so_far'])
    f_best = np.array(RB_simplex_list[i]['f_best_so_far'])
    nbr_feval = RB_simplex_list[i]['N_evals']
    ax1.plot(np.arange(len(f_best)), f_best, \
         label = 'Simplex'+str(i)+'; #f_eval: ' + str(nbr_feval))
    ax2.plot(x_best[:,0], x_best[:,1], '--', \
         label = 'Simplex'+str(i)+'; #f_eval: ' + str(nbr_feval))
ax1.legend()
ax1.set_yscale('log')
ax2.legend()


## Change to x_best_So_far
fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2 = trust_fig(oracle, bounds)
for i in range(N):
    x_best = np.array(RB_Nest_list[i]['x_store'])
    f_best = np.array(RB_Nest_list[i]['f_store'])
    nbr_feval = RB_Nest_list[i]['N_evals']
    ax1.plot(np.arange(len(f_best)), f_best, \
         label = 'Nest.'+str(i)+'; #f_eval: ' + str(nbr_feval))
    ax2.plot(x_best[:,0], x_best[:,1], '--', \
         label = 'Nest.'+str(i)+'; #f_eval: ' + str(nbr_feval))
ax1.legend()
ax1.set_yscale('log')
ax2.legend()







