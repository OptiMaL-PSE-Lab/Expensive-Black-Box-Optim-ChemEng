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
                          constraints = 2)
    RB_Nest_list.append(RB_Nest)
print('10 Nesterov iterations completed')

N = 10
RB_simplex_list = []
for i in range(N):
    rnd_seed = i
    RB_simplex = simplex_method(Problem_rosenbrock, x0, bounds, max_iter = 50, \
                            constraints = 2)
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
                        print_iteration = True, constraints=2, casadi=True, maxfun = 40).output_dict
    RB_Bayes_list.append(RB_Bayes)
 
print('10 BayesOpt iterations completed') 

# RB_Bayes_noCdi_list = []
# for i in range(N):
#     Bayes = BayesOpt()
#     pyro.set_rng_seed(1)
#     RB_Bayes_noCdi = Bayes.solve(Problem_rosenbrock, x0, acquisition='EI',bounds=bounds.T, \
#                         print_iteration = True, constraints=2, casadi=False, maxfun = 40)
#     RB_Bayes_noCdi_list.append(RB_Bayes_noCdi)
    
    

    

    
plt.plot()
    