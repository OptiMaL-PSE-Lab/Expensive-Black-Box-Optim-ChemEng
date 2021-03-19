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


def Problem_quadratic(x):
    f = quadratic_constrained.quadratic_f
    g = quadratic_constrained.quadratic_g

    return f(x), [g(x)]

bounds = np.array([[-1.5,1.5],[-1.5,1.5]])
# x0 = np.array([1, 1])
x0 = np.array([-0.5, 1.5])


max_f_eval = 100
max_it = 50


quadratic_pybobyqa = PyBobyqaWrapper().solve(Problem_quadratic, x0, bounds=bounds.T, \
                                      maxfun= max_f_eval, constraints=2)

N = 10
quadratic_Nest_list = []
for i in range(N):
    rnd_seed = i
    quadratic_Nest = nesterov_random(Problem_quadratic, x0, bounds, max_iter = 50, \
                          constraints = 2, rnd_seed = i, alpha = 1e-4)
    quadratic_Nest_list.append(quadratic_Nest)
print('10 Nesterov iterations completed')

N = 10
quadratic_simplex_list = []
for i in range(N):
    rnd_seed = i
    quadratic_simplex = simplex_method(Problem_quadratic, x0, bounds, max_iter = 50, \
                            constraints = 2, rnd_seed = i)
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
                          beta_red = 0.9, rnd = rnd_seed, method = 'local', \
                          constr_handling = method)
    quadratic_CUATRO_local_list.append(quadratic_CUATRO_local)
print('10 CUATRO local iterations completed') 

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
#                             print_iteration = True, constraints=2, casadi=True, \
#                             maxfun = nbr_feval, ).output_dict
#     quadratic_Bayes_list.append(quadratic_Bayes)
 
# print('10 BayesOpt iterations completed')

# with open('BayesQuadratic_list.pickle', 'wb') as handle:
#     pickle.dump(quadratic_Bayes_list, handle, protocol=pickle.HIGHEST_PROTOCOL)



x_best_pyBbyqa = np.array(quadratic_pybobyqa['x_best_so_far'])
f_best_pyBbyqa = np.array(quadratic_pybobyqa['f_best_so_far'])

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
ax1.legend()
ax1.set_yscale('log')

quadratic_f = lambda x, y: x**2 + 10*y**2 + x*y
quadratic_g = lambda x, y: 1 - x - y <= 0

oracle = RB(quadratic_f, ineq = [quadratic_g])

ax2 = trust_fig(oracle, bounds)
ax2.plot(x_best_pyBbyqa[:,0], x_best_pyBbyqa[:,1], '--x', \
         label = 'PyBobyqa')
ax2.plot(x_best_finDiff[:,0], x_best_finDiff[:,1], '--x', \
         label = 'Newton Fin. Diff.')
ax2.plot(x_best_BFGS[:,0], x_best_BFGS[:,1], '--x', \
         label = 'BFGS')
ax2.plot(x_best_Adam[:,0], x_best_Adam[:,1], '--x', \
         label = 'Adam')
ax2.legend()
ax2.set_xlim(bounds[0])
ax2.set_ylim(bounds[1])


fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2 = trust_fig(oracle, bounds)
for i in range(len(quadratic_CUATRO_global_list)):
    x_best = np.array(quadratic_CUATRO_global_list[i]['x_best_so_far'])
    f_best = np.array(quadratic_CUATRO_global_list[i]['f_best_so_far'])
    x_ind = np.array(quadratic_CUATRO_global_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'CUATRO_g'+str(i))
    # ax1.plot(x_ind, f_best, label = 'CUATRO_g'+str(i))
    ax2.plot(x_best[:,0], x_best[:,1], '--', label = 'CUATRO_g'+str(i))
ax1.legend()
ax1.set_yscale('log')
ax2.legend()
ax2.set_xlim(bounds[0])
ax2.set_ylim(bounds[1])


fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2 = trust_fig(oracle, bounds)
for i in range(len(quadratic_CUATRO_local_list)):
    x_best = np.array(quadratic_CUATRO_local_list[i]['x_best_so_far'])
    f_best = np.array(quadratic_CUATRO_local_list[i]['f_best_so_far'])
    x_ind = np.array(quadratic_CUATRO_local_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'CUATRO_l'+str(i))
    # ax1.plot(x_ind, f_best, label = 'CUATRO_l'+str(i))
    ax2.plot(x_best[:,0], x_best[:,1], '--', label = 'CUATRO_l'+str(i))
ax1.legend()
ax1.set_yscale('log')
ax2.legend()
ax2.set_xlim(bounds[0])
ax2.set_ylim(bounds[1])

# fig1 = plt.figure()
# ax1 = fig1.add_subplot()
# ax2 = trust_fig(oracle, bounds)
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
# ax2.legend()
# ax2.set_xlim(bounds[0])
# ax2.set_ylim(bounds[1])
    

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2 = trust_fig(oracle, bounds)
for i in range(len(quadratic_simplex_list)):
    x_best = np.array(quadratic_simplex_list[i]['x_best_so_far'])
    f_best = np.array(quadratic_simplex_list[i]['f_best_so_far'])
    x_ind = np.array(quadratic_simplex_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'Simplex'+str(i))
    ax2.plot(x_best[:,0], x_best[:,1], '--', label = 'Simplex'+str(i))
ax1.legend()
ax1.set_yscale('log')
ax2.legend()


## Change to x_best_So_far
fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2 = trust_fig(oracle, bounds)
for i in range(N):
    x_best = np.array(quadratic_Nest_list[i]['x_best_so_far'])
    f_best = np.array(quadratic_Nest_list[i]['f_best_so_far'])
    x_ind = np.array(quadratic_Nest_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'Nest.'+str(i))
    ax2.plot(x_best[:,0], x_best[:,1], '--', label = 'Nest.'+str(i))
ax1.legend()
ax1.set_yscale('log')
ax2.legend()



# with open('BayesQuadratic_list.pickle', 'rb') as handle:
#     Quadratic_Bayes_list_ = pickle.load(handle)




