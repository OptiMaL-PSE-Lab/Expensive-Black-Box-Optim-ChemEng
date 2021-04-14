# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 14:54:37 2021

@author: dv516
"""

from algorithms.Bayesian_opt_Pyro.utilities_full import BayesOpt
from test_functions import quadratic_constrained

import numpy as np
import pickle

import pyro
pyro.enable_validation(True)  # can help with debugging

def Problem_quadratic(x):
    f = quadratic_constrained.quadratic_f
    g = quadratic_constrained.quadratic_g

    return f(x), [g(x)]

def Problem_quadraticRand(x):
    f = quadratic_constrained.quadratic_f
    f_noise = np.random.normal(0, 0.05)
    g = quadratic_constrained.quadratic_g
    g_noise = np.random.normal(0, 0.01)

    return f(x) + f_noise, [g(x) + g_noise]

def Problem_quadraticSAA(x):
    N_SAA = 2
    f_SAA = 0
    g_SAA = - np.inf
    for i in range(N_SAA):
        f_sample = quadratic_constrained.quadratic_f(x)
        f_noise = np.random.normal(0, 0.05)
        f_SAA += (f_sample + f_noise)/N_SAA
        g_sample = quadratic_constrained.quadratic_g(x)
        g_noise = np.random.normal(0, 0.01)
        g_SAA = max(g_SAA, g_sample + g_noise)

    return f_SAA, [g_SAA]

def Problem_quadraticNoise(x, noise_std, N_SAA):
    f_SAA = 0 ; g_SAA = - np.inf
    for i in range(N_SAA):
        f_sample = quadratic_constrained.quadratic_f(x)
        f_noise = np.random.normal(0, noise_std[0])
        f_SAA += (f_sample + f_noise)/N_SAA
        g_sample = quadratic_constrained.quadratic_g(x)
        g_noise = np.random.normal(0, noise_std[1])
        g_SAA = max(g_SAA, g_sample + g_noise)

    return f_SAA, [g_SAA]

bounds = np.array([[-1.5,1.5],[-1.5,1.5]])
x0 = np.array([1, 1])

max_it = 100

# max_f_eval = 100
# max_it = 50
nbr_feval = 30

N = 10
quadratic_Bayes_list = []
for i in range(N):
    Bayes = BayesOpt()
    pyro.set_rng_seed(i)
    
    quadratic_Bayes = Bayes.solve(Problem_quadratic, x0, acquisition='EI',bounds=bounds.T, \
                            print_iteration = True, constraints=1, casadi=True, \
                            maxfun = nbr_feval, ).output_dict
    quadratic_Bayes_list.append(quadratic_Bayes)
 
print('10 BayesOpt deterministic iterations completed')

with open('BayesQuadratic_list.pickle', 'wb') as handle:
    pickle.dump(quadratic_Bayes_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

N = 10
quadraticRand_Bayes_list = []
for i in range(N):
    Bayes = BayesOpt()
    pyro.set_rng_seed(i)
    
    quadraticRand_Bayes = Bayes.solve(Problem_quadraticRand, x0, acquisition='EI',bounds=bounds.T, \
                            print_iteration = True, constraints=1, casadi=True, \
                            maxfun = nbr_feval, ).output_dict
    quadraticRand_Bayes_list.append(quadraticRand_Bayes)
 
print('10 BayesOpt random iterations completed')

with open('BayesQuadratic_listRand.pickle', 'wb') as handle:
    pickle.dump(quadraticRand_Bayes_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
N = 10
quadraticSAA_Bayes_list = []
for i in range(N):
    Bayes = BayesOpt()
    pyro.set_rng_seed(i)
    
    quadraticSAA_Bayes = Bayes.solve(Problem_quadraticSAA, x0, acquisition='EI',bounds=bounds.T, \
                            print_iteration = True, constraints=1, casadi=True, \
                            maxfun = nbr_feval, ).output_dict
    quadraticSAA_Bayes_list.append(quadraticSAA_Bayes)
 
print('10 BayesOpt iterations completed')

with open('BayesQuadratic_listRandSAA.pickle', 'wb') as handle:
    pickle.dump(quadraticSAA_Bayes_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

n_noise = 6
noise_matrix = np.zeros((n_noise, 2))
for i in range(n_noise):
    noise_matrix[i] = np.array([0.05/3, 0.01/3])*i

bounds = np.array([[-1.5,1.5],[-1.5,1.5]])
x0 = np.array([-0.5,1.5])
max_f_eval = 50 ; N_SAA = 1

 
N_SAA = 1
N_samples = 20
quadraticNoise_list_Bayes = []
quadraticConstraint_list_Bayes = []
for i in range(n_noise):
    print('Outer Iteration ', i+1, ' out of ', n_noise,' of BayesOpt')
    best = []
    best_constr = []
    for j in range(N_samples):
        f = lambda x: Problem_quadraticNoise(x, noise_matrix[i], N_SAA)
        sol = Bayes.solve(f, x0, acquisition='EI',bounds=bounds.T, \
                            print_iteration = True, constraints=1, casadi=True, \
                            maxfun = nbr_feval, ).output_dict
        best.append(sol['f_best_so_far'][-1])
        _, g = Problem_quadraticNoise(sol['x_best_so_far'][-1], [0, 0], N_SAA)
        best_constr.append(np.sum(np.maximum(g, 0)))
    quadraticNoise_list_Bayes.append(best)
    quadraticConstraint_list_Bayes.append(best_constr)

with open('BayesQuadratic_listNoiseConv.pickle', 'wb') as handle:
    pickle.dump(quadraticNoise_list_Bayes, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('BayesQuadratic_listNoiseConstr.pickle', 'wb') as handle:
    pickle.dump(quadraticConstraint_list_Bayes, handle, protocol=pickle.HIGHEST_PROTOCOL)


N_SAA = 2
nbr_feval = 25
N_samples = 20
quadraticNoiseSAA_list_Bayes = []
quadraticConstraintSAA_list_Bayes = []
for i in range(n_noise):
    print('Outer Iteration ', i+1, ' out of ', n_noise,' of BayesOpt')
    best = []
    best_constr = []
    for j in range(N_samples):
        f = lambda x: Problem_quadraticNoise(x, noise_matrix[i], N_SAA)
        sol = Bayes.solve(f, x0, acquisition='EI',bounds=bounds.T, \
                            print_iteration = True, constraints=2, casadi=True, \
                            maxfun = nbr_feval, ).output_dict
        best.append(sol['f_best_so_far'][-1])
        _, g = Problem_quadraticNoise(sol['x_best_so_far'][-1], [0, 0], N_SAA)
        best_constr.append(np.sum(np.maximum(g, 0)))
    quadraticNoiseSAA_list_Bayes.append(best)
    quadraticConstraintSAA_list_Bayes.append(best_constr)

with open('BayesQuadrati_listNoiseConvSAA.pickle', 'wb') as handle:
    pickle.dump(quadraticNoiseSAA_list_Bayes, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('BayesQuadrati_listNoiseConstrSAA.pickle', 'wb') as handle:
    pickle.dump(quadraticConstraintSAA_list_Bayes, handle, protocol=pickle.HIGHEST_PROTOCOL)
    



