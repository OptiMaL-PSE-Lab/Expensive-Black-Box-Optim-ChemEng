# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 14:54:37 2021

@author: dv516
"""

from algorithms.Bayesian_opt_Pyro.utilities_full import BayesOpt
from test_functions import rosenbrock_constrained

import numpy as np
import pickle

import pyro
pyro.enable_validation(True)  # can help with debugging

def Problem_rosenbrock(x):
    f1 = rosenbrock_constrained.rosenbrock_f
    g1 = rosenbrock_constrained.rosenbrock_g1
    g2 = rosenbrock_constrained.rosenbrock_g2

    return f1(x), [g1(x), g2(x)]

def Problem_rosenbrockRand(x):
    f1 = rosenbrock_constrained.rosenbrock_f
    f_noise = np.random.normal(0, 0.05)
    g1 = rosenbrock_constrained.rosenbrock_g1
    g2 = rosenbrock_constrained.rosenbrock_g2
    g_noise1 = np.random.normal(0, 0.02)
    g_noise2 = np.random.normal(0, 0.02)

    return f1(x) + f_noise, [g1(x) + g_noise1, g2(x) + g_noise2]

def Problem_rosenbrockSAA(x):
    N_SAA = 5
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

def Problem_rosenbrockNoise(x, noise_std, N_SAA):
    f_SAA = 0
    g_SAA1, g_SAA2 = - np.inf, -np.inf
    f1 = rosenbrock_constrained.rosenbrock_f
    g1 = rosenbrock_constrained.rosenbrock_g1
    g2 = rosenbrock_constrained.rosenbrock_g2
    for i in range(N_SAA):
        f_SAA += (f1(x) + np.random.normal(0, noise_std[0]))/N_SAA
        g_SAA1 = max(g1(x) + np.random.normal(0, noise_std[1]), g_SAA1)
        g_SAA2 = max(g2(x) + np.random.normal(0, noise_std[2]), g_SAA2)

    return f_SAA, [g_SAA1, g_SAA2]

bounds = np.array([[-1.5,1.5],[-1.5,1.5]])
x0 = np.array([-0.5,1.5])

# max_f_eval = 100
# max_it = 50
nbr_feval = 50

N = 10
RB_Bayes_list = []
for i in range(N):
    Bayes = BayesOpt()
    pyro.set_rng_seed(i)
    
    RB_Bayes = Bayes.solve(Problem_rosenbrock, x0, acquisition='EI',bounds=bounds.T, \
                            print_iteration = True, constraints=2, casadi=True, \
                            maxfun = nbr_feval, ).output_dict
    RB_Bayes_list.append(RB_Bayes)
 
print('10 BayesOpt deterministic iterations completed')

with open('BayesRB_list.pickle', 'wb') as handle:
    pickle.dump(RB_Bayes_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

N = 10
RBRand_Bayes_list = []
for i in range(N):
    Bayes = BayesOpt()
    pyro.set_rng_seed(i)
    
    RBRand_Bayes = Bayes.solve(Problem_rosenbrockRand, x0, acquisition='EI',bounds=bounds.T, \
                            print_iteration = True, constraints=2, casadi=True, \
                            maxfun = nbr_feval, ).output_dict
    RBRand_Bayes_list.append(RB_Bayes)
 
print('10 BayesOpt random iterations completed')

with open('BayesRB_listRand.pickle', 'wb') as handle:
    pickle.dump(RBRand_Bayes_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
N = 10
RBSAA_Bayes_list = []
for i in range(N):
    Bayes = BayesOpt()
    pyro.set_rng_seed(i)
    
    RBSAA_Bayes = Bayes.solve(Problem_rosenbrockSAA, x0, acquisition='EI',bounds=bounds.T, \
                            print_iteration = True, constraints=2, casadi=True, \
                            maxfun = nbr_feval, ).output_dict
    RBSAA_Bayes_list.append(RBSAA_Bayes)
 
print('10 BayesOpt iterations completed')

with open('BayesRB_listRandSAA.pickle', 'wb') as handle:
    pickle.dump(RBSAA_Bayes_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

n_noise = 6
noise_matrix = np.zeros((n_noise, 3))
for i in range(n_noise):
    noise_matrix[i] = np.array([0.05/3, 0.02/3, 0.02/3])*i

bounds = np.array([[-1.5,1.5],[-1.5,1.5]])
x0 = np.array([-0.5,1.5])
max_f_eval = 50 ; N_SAA = 1

 
N_SAA = 1
N_samples = 20
RBnoise_list_Bayes = []
RBconstraint_list_Bayes = []
for i in range(n_noise):
    print('Outer Iteration ', i+1, ' out of ', n_noise,' of BayesOpt')
    best = []
    best_constr = []
    for j in range(N_samples):
        f = lambda x: Problem_rosenbrockNoise(x, noise_matrix[i], N_SAA)
        sol = Bayes.solve(f, x0, acquisition='EI',bounds=bounds.T, \
                            print_iteration = True, constraints=2, casadi=True, \
                            maxfun = nbr_feval, ).output_dict
        best.append(sol['f_best_so_far'][-1])
        _, g = Problem_rosenbrockNoise(sol['x_best_so_far'][-1], [0, 0, 0], N_SAA)
        best_constr.append(np.sum(np.maximum(g, 0)))
    RBnoise_list_Bayes.append(best)
    RBconstraint_list_Bayes.append(best_constr)

with open('BayesRB_listNoiseConv.pickle', 'wb') as handle:
    pickle.dump(RBnoise_list_Bayes, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('BayesRB_listNoiseConstr.pickle', 'wb') as handle:
    pickle.dump(RBconstraint_list_Bayes, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

N_SAA = 2
N_samples = 20
RBnoiseSAA_list_Bayes = []
RBconstraintSAA_list_Bayes = []
for i in range(n_noise):
    print('Outer Iteration ', i+1, ' out of ', n_noise,' of BayesOpt')
    best = []
    best_constr = []
    for j in range(N_samples):
        f = lambda x: Problem_rosenbrockNoise(x, noise_matrix[i], N_SAA)
        sol = Bayes.solve(f, x0, acquisition='EI',bounds=bounds.T, \
                            print_iteration = True, constraints=2, casadi=True, \
                            maxfun = nbr_feval, ).output_dict
        best.append(sol['f_best_so_far'][-1])
        _, g = Problem_rosenbrockNoise(sol['x_best_so_far'][-1], [0, 0, 0], N_SAA)
        best_constr.append(np.sum(np.maximum(g, 0)))
    RBnoiseSAA_list_Bayes.append(best)
    RBconstraintSAA_list_Bayes.append(best_constr)

with open('BayesRB_listNoiseConvSAA.pickle', 'wb') as handle:
    pickle.dump(RBnoiseSAA_list_Bayes, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('BayesRB_listNoiseConstrSAA.pickle', 'wb') as handle:
    pickle.dump(RBconstraintSAA_list_Bayes, handle, protocol=pickle.HIGHEST_PROTOCOL)
    



