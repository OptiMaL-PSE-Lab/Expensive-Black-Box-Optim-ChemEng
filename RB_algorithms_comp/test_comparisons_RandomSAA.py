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

from test_functions import rosenbrock_constrained, quadratic_constrained

import matplotlib.pyplot as plt

import pickle

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


def Problem_rosenbrock(x):
    N_SAA = 10
    f_SAA = 0
    g_SAA1, g_SAA2 = - np.inf, -np.inf
    f1 = rosenbrock_constrained.rosenbrock_f
    g1 = rosenbrock_constrained.rosenbrock_g1
    g2 = rosenbrock_constrained.rosenbrock_g2
    for i in range(N_SAA):
        f_SAA += (f1(x) + np.random.normal(0, 0.05))/N_SAA
        g_SAA1 = max(g1(x) + np.random.normal(0, 0.02), g_SAA1)
        g_SAA2 = max(g2(x) + np.random.normal(0, 0.02), g_SAA2)

    return f_SAA, [g_SAA1, g_SAA2]


bounds = np.array([[-1.5,1.5],[-1.5,1.5]])
x0 = np.array([-0.5,1.5])

max_f_eval = 100
max_it = 50

N = 10
RBSAA_pybbqa_list = []
for i in range(N):
    RBSAA_pybobyqa = PyBobyqaWrapper().solve(Problem_rosenbrock, x0, bounds=bounds.T, \
                                              maxfun= max_f_eval, constraints=2, \
                                              seek_global_minimum = True)
    RBSAA_pybbqa_list.append(RBSAA_pybobyqa)   
print('10 Py-BOBYQA iterations completed')

N = 10
RBSAA_Nest_list = []
for i in range(N):
    rnd_seed = i
    RBSAA_Nest = nesterov_random(Problem_rosenbrock, x0, bounds, max_iter = 50, \
                          constraints = 2, rnd_seed = i, alpha = 1e-4)
    RBSAA_Nest_list.append(RBSAA_Nest)
print('10 Nesterov iterations completed')

N = 10
RBSAA_simplex_list = []
for i in range(N):
    rnd_seed = i
    RBSAA_simplex = simplex_method(Problem_rosenbrock, x0, bounds, max_iter = 50, \
                            constraints = 2, rnd_seed = i)
    RBSAA_simplex_list.append(RBSAA_simplex)
print('10 simplex iterations completed')

N = 10
RBSAA_findiff_list = []
for i in range(N):
    RBSAA_FiniteDiff = finite_Diff_Newton(Problem_rosenbrock, x0, bounds = bounds, \
                                   con_weight = 100)
    RBSAA_findiff_list.append(RBSAA_FiniteDiff)
print('10 Approx Newton iterations completed')
    
N = 10
RBSAA_BFGS_list = []
for i in range(N):
    RBSAA_BFGS = BFGS_optimizer(Problem_rosenbrock, x0, bounds = bounds, \
                         con_weight = 100)
    RBSAA_BFGS_list.append(RBSAA_BFGS)
print('10 BFGS iterations completed')
    
N = 10
RBSAA_Adam_list = []
for i in range(N):
    RBSAA_Adam = Adam_optimizer(Problem_rosenbrock, x0, method = 'forward', \
                                      bounds = bounds, alpha = 0.4, \
                                      beta1 = 0.2, beta2  = 0.1, \
                                      max_f_eval = 100, con_weight = 100)
    RBSAA_Adam_list.append(RBSAA_Adam)
print('10 Adam iterations completed')
    
N_min_s = 15
init_radius = 2
method = 'Discrimination'
N = 10
RBSAA_CUATRO_global_list = []
for i in range(N):
    rnd_seed = i
    RBSAA_CUATRO_global = CUATRO(Problem_rosenbrock, x0, init_radius, bounds = bounds, \
                          N_min_samples = N_min_s, tolerance = 1e-10,\
                          beta_red = 0.9, rnd = rnd_seed, method = 'global', \
                          constr_handling = method)
    RBSAA_CUATRO_global_list.append(RBSAA_CUATRO_global)
print('10 CUATRO global iterations completed')    
    
N_min_s = 6
init_radius = 0.1
method = 'Fitting'
N = 10
RBSAA_CUATRO_local_list = []
for i in range(N):
    rnd_seed = i
    RBSAA_CUATRO_local = CUATRO(Problem_rosenbrock, x0, init_radius, bounds = bounds, \
                          N_min_samples = N_min_s, tolerance = 1e-10,\
                          beta_red = 0.9, rnd = rnd_seed, method = 'local', \
                          constr_handling = method)
    RBSAA_CUATRO_local_list.append(RBSAA_CUATRO_local)
print('10 CUATRO local iterations completed') 

# N = 10
# RB_Bayes_list = []
# for i in range(N):
#     Bayes = BayesOpt()
#     pyro.set_rng_seed(i)
    
#     if i<3:
#         nbr_feval = 40
#     elif i<6:
#         nbr_feval = 30
#     else:
#         nbr_feval = 20
    
#     RB_Bayes = Bayes.solve(Problem_rosenbrock, x0, acquisition='EI',bounds=bounds.T, \
#                             print_iteration = True, constraints=2, casadi=True, \
#                             maxfun = nbr_feval, ).output_dict
#     RB_Bayes_list.append(RB_Bayes)
 
# print('10 BayesOpt iterations completed')

# with open('BayesRB_listRand.pickle', 'wb') as handle:
#     pickle.dump(RB_Bayes_list, handle, protocol=pickle.HIGHEST_PROTOCOL)



f_RB = lambda x, y: (1 - x)**2 + 100*(y - x**2)**2
g1_RB = lambda x, y: (x-1)**3 - y + 1 <= 0
g2_RB = lambda x, y: x + y - 1.8 <= 0

oracle = RB(f_RB, ineq = [g1_RB, g2_RB])

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2 = trust_fig(oracle, bounds)
for i in range(len(RBSAA_pybbqa_list)):
    x_best = np.array(RBSAA_pybbqa_list[i]['x_best_so_far'])
    f_best = np.array(RBSAA_pybbqa_list[i]['f_best_so_far'])
    x_ind = np.array(RBSAA_pybbqa_list[i]['samples_at_iteration'])
    # ax1.step(x_ind, f_best, where = 'post', label = 'CUATRO_g'+str(i))
    ax1.plot(x_ind, f_best, label = 'Py-BOBYQA'+str(i))
    ax2.plot(x_best[:,0], x_best[:,1], '--', label = 'Py-BOBYQA'+str(i))
ax1.legend()
ax1.set_yscale('log')
ax2.legend()
ax2.set_xlim(bounds[0])
ax2.set_ylim(bounds[1])

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2 = trust_fig(oracle, bounds)
for i in range(len(RBSAA_findiff_list)):
    x_best = np.array(RBSAA_findiff_list[i]['x_best_so_far'])
    f_best = np.array(RBSAA_findiff_list[i]['f_best_so_far'])
    x_ind = np.array(RBSAA_findiff_list[i]['samples_at_iteration'])
    # ax1.step(x_ind, f_best, where = 'post', label = 'CUATRO_g'+str(i))
    ax1.plot(x_ind, f_best, label = 'Approx. Newton'+str(i))
    ax2.plot(x_best[:,0], x_best[:,1], '--', label = 'Approx. Newton'+str(i))
ax1.legend()
ax1.set_yscale('log')
ax2.legend()
ax2.set_xlim(bounds[0])
ax2.set_ylim(bounds[1])

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2 = trust_fig(oracle, bounds)
for i in range(len(RBSAA_BFGS_list)):
    x_best = np.array(RBSAA_BFGS_list[i]['x_best_so_far'])
    f_best = np.array(RBSAA_BFGS_list[i]['f_best_so_far'])
    x_ind = np.array(RBSAA_BFGS_list[i]['samples_at_iteration'])
    # ax1.step(x_ind, f_best, where = 'post', label = 'CUATRO_g'+str(i))
    ax1.plot(x_ind, f_best, label = 'BFGS'+str(i))
    ax2.plot(x_best[:,0], x_best[:,1], '--', label = 'BFGS'+str(i))
ax1.legend()
ax1.set_yscale('log')
ax2.legend()
ax2.set_xlim(bounds[0])
ax2.set_ylim(bounds[1])

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2 = trust_fig(oracle, bounds)
for i in range(len(RBSAA_BFGS_list)):
    x_best = np.array(RBSAA_Adam_list[i]['x_best_so_far'])
    f_best = np.array(RBSAA_Adam_list[i]['f_best_so_far'])
    x_ind = np.array(RBSAA_Adam_list[i]['samples_at_iteration'])
    # ax1.step(x_ind, f_best, where = 'post', label = 'CUATRO_g'+str(i))
    ax1.plot(x_ind, f_best, label = 'Adam'+str(i))
    ax2.plot(x_best[:,0], x_best[:,1], '--', label = 'Adam'+str(i))
ax1.legend()
ax1.set_yscale('log')
ax2.legend()
ax2.set_xlim(bounds[0])
ax2.set_ylim(bounds[1])

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2 = trust_fig(oracle, bounds)
for i in range(len(RBSAA_CUATRO_global_list)):
    x_best = np.array(RBSAA_CUATRO_global_list[i]['x_best_so_far'])
    f_best = np.array(RBSAA_CUATRO_global_list[i]['f_best_so_far'])
    x_ind = np.array(RBSAA_CUATRO_global_list[i]['samples_at_iteration'])
    # ax1.step(x_ind, f_best, where = 'post', label = 'CUATRO_g'+str(i))
    ax1.plot(x_ind, f_best, label = 'CUATRO_g'+str(i))
    ax2.plot(x_best[:,0], x_best[:,1], '--', label = 'CUATRO_g'+str(i))
ax1.legend()
ax1.set_yscale('log')
ax2.legend()
ax2.set_xlim(bounds[0])
ax2.set_ylim(bounds[1])


fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2 = trust_fig(oracle, bounds)
for i in range(len(RBSAA_CUATRO_local_list)):
    x_best = np.array(RBSAA_CUATRO_local_list[i]['x_best_so_far'])
    f_best = np.array(RBSAA_CUATRO_local_list[i]['f_best_so_far'])
    x_ind = np.array(RBSAA_CUATRO_local_list[i]['samples_at_iteration'])
    # ax1.step(x_ind, f_best, where = 'post', label = 'CUATRO_l'+str(i))
    ax1.plot(x_ind, f_best, label = 'CUATRO_l'+str(i))
    ax2.plot(x_best[:,0], x_best[:,1], '--', label = 'CUATRO_l'+str(i))
ax1.legend()
ax1.set_yscale('log')
ax2.legend()
ax2.set_xlim(bounds[0])
ax2.set_ylim(bounds[1])

# fig1 = plt.figure()
# ax1 = fig1.add_subplot()
# ax2 = trust_fig(oracle, bounds)
# for i in range(len(RB_Bayes_list)):
#     x_best = np.array(RB_Bayes_list[i]['x_best_so_far'])
#     f_best = np.array(RB_Bayes_list[i]['f_best_so_far'])
#     nbr_feval = len(RB_Bayes_list[i]['f_store'])
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
for i in range(len(RBSAA_simplex_list)):
    x_best = np.array(RBSAA_simplex_list[i]['x_best_so_far'])
    f_best = np.array(RBSAA_simplex_list[i]['f_best_so_far'])
    x_ind = np.array(RBSAA_simplex_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'Simplex'+str(i))
    ax2.plot(x_best[:,0], x_best[:,1], '--', label = 'Simplex'+str(i))
ax1.legend()
ax1.set_yscale('log')
ax2.legend()


## Change to x_best_So_far
fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2 = trust_fig(oracle, bounds)
for i in range(len(RBSAA_Nest_list)):
    x_best = np.array(RBSAA_Nest_list[i]['x_best_so_far'])
    f_best = np.array(RBSAA_Nest_list[i]['f_best_so_far'])
    x_ind = np.array(RBSAA_Nest_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'Nest.'+str(i))
    ax2.plot(x_best[:,0], x_best[:,1], '--', label = 'Nest.'+str(i))
ax1.legend()
ax1.set_yscale('log')
ax2.legend()



sol_Cg = average_from_list(RBSAA_CUATRO_global_list)
test_CUATROg, test_av_CUATROg, test_min_CUATROg, test_max_CUATROg = sol_Cg
sol_Cl = average_from_list(RBSAA_CUATRO_local_list)
test_CUATROl, test_av_CUATROl, test_min_CUATROl, test_max_CUATROl = sol_Cl
sol_Nest = average_from_list(RBSAA_Nest_list)
test_Nest, test_av_Nest, test_min_Nest, test_max_Nest = sol_Nest
sol_Splx = average_from_list(RBSAA_simplex_list)
test_Splx, test_av_Splx, test_min_Splx, test_max_Splx = sol_Splx
sol_pybbyqa = average_from_list(RBSAA_pybbqa_list)
test_pybbqa, test_av_pybbqa, test_min_pybbqa, test_max_pybbqa = sol_pybbyqa
sol_findiff = average_from_list(RBSAA_findiff_list)
test_findiff, test_av_findiff, test_min_findiff, test_max_findiff = sol_findiff
sol_BFGS = average_from_list(RBSAA_BFGS_list)
test_BFGS, test_av_BFGS, test_min_BFGS, test_max_BFGS = sol_BFGS
sol_Adam = average_from_list(RBSAA_Adam_list)
test_Adam, test_av_Adam, test_min_Adam, test_max_Adam = sol_Adam



fig = plt.figure()
ax = fig.add_subplot()
ax.step(np.arange(1, 101), test_av_CUATROg, where = 'post', label = 'CUATRO_global', c = 'b')
ax.fill_between(np.arange(1, 101), test_min_CUATROg, \
                test_max_CUATROg, color = 'b', alpha = .5)
ax.step(np.arange(1, 101), test_av_CUATROl, where = 'post', label = 'CUATRO_local', c = 'r')
ax.fill_between(np.arange(1, 101), test_min_CUATROl, \
                test_max_CUATROl, color = 'r', alpha = .5)
ax.step(np.arange(1, 101), test_av_findiff, where = 'post', label = 'Approx. Newton', c = 'grey')
ax.fill_between(np.arange(1, 101), test_min_findiff, \
                test_max_findiff, color = 'grey', alpha = .5)
ax.step(np.arange(1, 101), test_av_BFGS, where = 'post', label = 'Approx. BFGS', c = 'orange')
ax.fill_between(np.arange(1, 101), test_min_BFGS, \
                test_max_BFGS, color = 'orange', alpha = .5)
ax.step(np.arange(1, 101), test_av_Adam, where = 'post', label = 'Adam ', c = 'green')
ax.fill_between(np.arange(1, 101), test_min_Adam, \
                test_max_Adam, color = 'green', alpha = .5)

ax.legend()
ax.set_yscale('log')
ax.set_xlim([0, 99])    
    
fig = plt.figure()
ax = fig.add_subplot()
ax.step(np.arange(1, 101), test_av_Nest, where = 'post', label = 'Nesterov', c = 'brown')
ax.fill_between(np.arange(1, 101), test_min_Nest, \
                test_max_Nest, color = 'brown', alpha = .5)
ax.step(np.arange(1, 101), test_av_Splx, where = 'post', label = 'Simplex', c = 'green')
ax.fill_between(np.arange(1, 101), test_min_Splx, \
                test_max_Splx, color = 'green', alpha = .5)
# ax.boxplot(test_BO, widths = 0.1, meanline = False, showfliers = False, manage_ticks = False)
# ax.step(np.arange(1, 101), test_av_BO, where = 'post', label = 'Bayes. Opt.')

ax.step(np.arange(1, 101), test_av_pybbqa, where = 'post', label = 'Py-BOBYQA ', c = 'blue')
ax.fill_between(np.arange(1, 101), test_min_pybbqa, \
                test_max_pybbqa, color = 'blue', alpha = .5)
# f_best = np.array(RB_Bayes_list[0]['f_best_so_far'])
# ax.step(np.arange(len(f_best)), f_best, where = 'post', \
#           label = 'BO', c = 'blue')

ax.legend()
ax.set_yscale('log')
ax.set_xlim([0, 99])



# with open('BayesRB_list.pickle', 'rb') as handle:
#     RB_Bayes_list_ = pickle.load(handle)






# with open('BayesRB_list.pickle', 'rb') as handle:
#     RB_Bayes_list_ = pickle.load(handle)




