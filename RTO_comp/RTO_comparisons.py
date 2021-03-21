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

from case_studies.RTO.systems import *

def trust_fig(X, Y, Z, g1, g2):   
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.contour(X, Y, Z, 50)
    ax.contour(X, Y, g1, levels = [0], colors = 'black')
    ax.contour(X, Y, g2, levels = [0], colors = 'black')
    
    return ax

def scaling(x):
    y = np.zeros(len(x))
    y[0] = (x[0] - 4)/(7 - 4)
    y[1] = (x[1] - 70)/(100 - 70)
    return y

def extract_FT(x):
    x[0] = 4 + (7 - 4)*x[0]
    x[1] = 70 + (100 - 70)*x[1]
    return x

def RTO(x):
    # x = extract_FT(x)
    plant = WO_system()
    f = plant.WO_obj_sys_ca_noise_less
    g1 = plant.WO_con1_sys_ca_noise_less
    g2 = plant.WO_con2_sys_ca_noise_less
    return f(x), [g1(x), g2(x)]


## Apply scaling transformation

# bounds = np.array([[0., 1.],[0., 1.]])
# x0 = np.array([0.967, 0.433])

x0 = [6.9, 83]
bounds  = np.array([[4., 7.], [70., 100.]])

max_f_eval = 100
max_it = 100



RTO_pybobyqa = PyBobyqaWrapper().solve(RTO, x0, bounds=bounds.T, \
                                      maxfun= max_f_eval, constraints=2, \
                                      scaling_within_bounds = True, \
                                      mu_con = 1e6)
# RTO_pybobyqa = PyBobyqaWrapper().solve(RTO, x0, bounds=bounds.T, \
#                                       maxfun= max_f_eval, constraints=2)
# print(x0)
N = 10
RTO_Nest_list = []
for i in range(N):
    rnd_seed = i
    RTO_Nest = nesterov_random(RTO, x0, bounds, max_iter = 50, \
                               constraints = 2, rnd_seed = i, alpha = 1e-4)
    RTO_Nest_list.append(RTO_Nest)
print('10 Nesterov iterations completed')
# print(x0)


N = 10
RTO_simplex_list = []
for i in range(N):
    rnd_seed = i
    RTO_simplex = simplex_method(RTO, x0, bounds, max_iter = 50, \
                            constraints = 2, rnd_seed = i, mu_con = 1e6)
    RTO_simplex_list.append(RTO_simplex)
print('10 simplex iterations completed')
# print(x0)

RTO_FiniteDiff = finite_Diff_Newton(RTO, x0, bounds = bounds, \
                                   con_weight = 100)
# print(x0)  
# RTO_BFGS = BFGS_optimizer(RTO, x0, bounds = bounds, \
#                          con_weight = 100)
    
RTO_Adam = Adam_optimizer(RTO, x0, method = 'forward', \
                                      bounds = bounds, alpha = 0.4, \
                                      beta1 = 0.2, beta2  = 0.1, \
                                      max_f_eval = 100, con_weight = 100)
# print(x0) 
N_min_s = 15
# init_radius = 0.5
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
# print(x0)   
 
N_min_s = 6
# init_radius = 0.05
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
# print(x0)

N = 10
RTO_Bayes_list = []
for i in range(1):
    Bayes = BayesOpt()
    pyro.set_rng_seed(i)
    
    if i<3:
        nbr_feval = 40
    elif i<6:
        nbr_feval = 30
    else:
        nbr_feval = 20
    
    RTO_Bayes = Bayes.solve(RTO, x0, acquisition='EI',bounds=bounds.T, \
                            print_iteration = True, constraints=2, casadi=True, \
                            maxfun = nbr_feval, ).output_dict
    RTO_Bayes_list.append(RTO_Bayes)
 
print('10 BayesOpt iterations completed')

with open('BayesRTO_list.pickle', 'wb') as handle:
    pickle.dump(RTO_Bayes_list, handle, protocol=pickle.HIGHEST_PROTOCOL)


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
# x_ind_pyBbyqa = np.array(RB_pybobyqa['samples_at_iteration'])
# nbr_feval_pyBbyqa = len(RB_pybobyqa['f_store'])

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
# ax1.step(x_ind_BFGS, f_best_BFGS, where = 'post', \
#          label = 'BFGS')
ax1.step(x_ind_Adam, f_best_Adam, where = 'post', \
         label = 'Adam')
ax1.legend()
# ax1.set_yscale('log')


ax2 = trust_fig(X, Y, Z, g1, g2)
ax2.plot(x_best_pyBbyqa[:,0], x_best_pyBbyqa[:,1], '--x', \
          label = 'PyBobyqa')
ax2.plot(x_best_finDiff[:,0], x_best_finDiff[:,1], '--x', \
         label = 'Newton Fin. Diff.')
# ax2.plot(x_best_BFGS[:,0], x_best_BFGS[:,1], '--x', \
#          label = 'BFGS')
ax2.plot(x_best_Adam[:,0], x_best_Adam[:,1], '--x', \
         label = 'Adam')
ax2.legend()
ax2.set_xlim(bounds[0])
ax2.set_ylim(bounds[1])


fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2 = trust_fig(X, Y, Z, g1, g2)
for i in range(len(RTO_CUATRO_global_list)):
    x_best = np.array(RTO_CUATRO_global_list[i]['x_best_so_far'])
    f_best = np.array(RTO_CUATRO_global_list[i]['f_best_so_far'])
    x_ind = np.array(RTO_CUATRO_global_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'CUATRO_g'+str(i))
    # ax1.plot(x_ind, f_best, label = 'CUATRO_g'+str(i))
    ax2.plot(x_best[:,0], x_best[:,1], '--', label = 'CUATRO_g'+str(i))
ax1.legend()
# ax1.set_yscale('log')
ax2.legend()
ax2.set_xlim(bounds[0])
ax2.set_ylim(bounds[1])


fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2 = trust_fig(X, Y, Z, g1, g2)
for i in range(len(RTO_CUATRO_local_list)):
    x_best = np.array(RTO_CUATRO_local_list[i]['x_best_so_far'])
    f_best = np.array(RTO_CUATRO_local_list[i]['f_best_so_far'])
    x_ind = np.array(RTO_CUATRO_local_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'CUATRO_l'+str(i))
    # ax1.plot(x_ind, f_best, label = 'CUATRO_l'+str(i))
    ax2.plot(x_best[:,0], x_best[:,1], '--', label = 'CUATRO_l'+str(i))
ax1.legend()
# ax1.set_yscale('log')
ax2.legend()
ax2.set_xlim(bounds[0])
ax2.set_ylim(bounds[1])

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2 = trust_fig(X, Y, Z, g1, g2)
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
ax2.legend()
ax2.set_xlim(bounds[0])
ax2.set_ylim(bounds[1])
    

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2 = trust_fig(X, Y, Z, g1, g2)
for i in range(len(RTO_simplex_list)):
    x_best = np.array(RTO_simplex_list[i]['x_best_so_far'])
    f_best = np.array(RTO_simplex_list[i]['f_best_so_far'])
    x_ind = np.array(RTO_simplex_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'Simplex'+str(i))
    ax2.plot(x_best[:,0], x_best[:,1], '--', label = 'Simplex'+str(i))
ax1.legend()
# ax1.set_yscale('log')
ax2.legend()


## Change to x_best_So_far
fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax2 = trust_fig(X, Y, Z, g1, g2)
for i in range(N):
    x_best = np.array(RTO_Nest_list[i]['x_best_so_far'])
    f_best = np.array(RTO_Nest_list[i]['f_best_so_far'])
    x_ind = np.array(RTO_Nest_list[i]['samples_at_iteration'])
    ax1.step(x_ind, f_best, where = 'post', label = 'Nest.'+str(i))
    ax2.plot(x_best[:,0], x_best[:,1], '--', label = 'Nest.'+str(i))
ax1.legend()
# ax1.set_yscale('log')
ax2.legend()



# with open('BayesRB_list.pickle', 'rb') as handle:
#     RB_Bayes_list_ = pickle.load(handle)








