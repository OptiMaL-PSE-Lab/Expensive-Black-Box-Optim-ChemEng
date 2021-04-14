# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 22:35:54 2021

@author: dv516
"""


import numpy as np
import pickle

import pyro
pyro.enable_validation(True)  # can help with debugging
pyro.set_rng_seed(1)

from algorithms.Bayesian_opt_Pyro.utilities_full import BayesOpt

from case_studies.Controller_tuning.Control_system import phi, phi_rand
from case_studies.Controller_tuning.Control_system import reactor_phi_2st, reactor_phi_2stNS

def phi_SAA(x):
    # x = extract_FT(x)
    N_SAA = 5
    
    f = phi_rand
    f_SAA = 0
    
    for i in range(N_SAA):
        f_SAA += f(x)[0]/N_SAA
    
    return f_SAA, [0]

def cost_control_noise(x, bounds_abs, noise, N_SAA, x0 = [.116, 368.489], \
                       N = 200, T = 20, NS = False):
    f_SAA = 0 ; g_SAA = -np.inf
    if not NS:
        f = lambda x: reactor_phi_2st(x, bounds_abs, noise, x0 = x0, N = N, \
                                    T = T, return_sys_resp = False)
    else:
        f = lambda x: reactor_phi_2stNS(x, noise, x0 = x0, N = N, \
                                    T = T, return_sys_resp = False)
    for i in range(N_SAA):
        f_sample = f(x)
        f_SAA += f_sample[0]/N_SAA
        g_SAA = np.maximum(g_SAA, float(f_sample[1][0]))
        
    return f_SAA, [g_SAA]

# x0 = np.array([4, 4, 4, 4])
# bounds = np.array([[0, 8], [0, 8], [0, 8], [0, 8]])

# nbr_feval = 30

# N = 10 
# ContrLin_Bayes_list = []
# for i in range(N):
#     Bayes = BayesOpt()
#     pyro.set_rng_seed(i)
    
#     phi_uncon = lambda x: phi(x)[0]
#     ContrLin_Bayes = Bayes.solve(phi_uncon, x0, acquisition='EI',bounds=bounds.T, \
#                             print_iteration = True, constraints=0, casadi=True, \
#                             maxfun = nbr_feval, ).output_dict
#     ContrLin_Bayes_list.append(ContrLin_Bayes)

# print('10 deterministic BayesOpt iterations completed')

# with open('BayesContrLin_list.pickle', 'wb') as handle:
#     pickle.dump(ContrLin_Bayes_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
# N = 10
# ContrLinRand_Bayes_list = []
# for i in range(N):
#     Bayes = BayesOpt()
#     pyro.set_rng_seed(i)
        
#     phi_unconRand = lambda x: phi_rand(x)[0]
#     ContrLinRand_Bayes = Bayes.solve(phi_unconRand, x0, acquisition='EI',bounds=bounds.T, \
#                             print_iteration = True, constraints=0, casadi=True, \
#                             maxfun = 40, ).output_dict
#     ContrLinRand_Bayes_list.append(ContrLinRand_Bayes)
 
# print('10 random BayesOpt iterations completed')

# with open('BayesContrLinRand_list.pickle', 'wb') as handle:
#     pickle.dump(ContrLinRand_Bayes_list, handle, protocol=pickle.HIGHEST_PROTOCOL)


# N = 10
# ContrLinSAA_Bayes_list = []
# for i in range(N):
#     Bayes = BayesOpt()
#     pyro.set_rng_seed(i)
        
#     phi_unconSAA = lambda x: phi_SAA(x)[0]
#     ContrLinSAA_Bayes = Bayes.solve(phi_unconSAA, x0, acquisition='EI',bounds=bounds.T, \
#                             print_iteration = True, constraints=0, casadi=True, \
#                             maxfun = nbr_feval, ).output_dict
#     ContrLinSAA_Bayes_list.append(ContrLinSAA_Bayes)
 
# print('10 SAA BayesOpt iterations completed')

# with open('BayesContrLinSAA_list.pickle', 'wb') as handle:
#     pickle.dump(ContrLinSAA_Bayes_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

pi = [.8746, .0257, -1.43388, -0.00131, 0.00016, 55.8692, 0.7159, .0188, .00017]
pi_init = [.8746, .0257, -1.43388, -0.00131, 0.00016, 0, 0, 0, 0, 0]

bounds_abs = np.zeros((10, 2))

for i in range(5):
    if pi[i] > 0:
        bounds_abs[i] = [pi[i]/2, pi[i]*2]
        bounds_abs[i+5] = [-pi[i]*10, pi[i]*10]
    else:
        bounds_abs[i] = [pi[i]*2, pi[i]/2]
        bounds_abs[i+5] = [pi[i]*10, -pi[i]*10]
        
noise = [.001, 1]
N_SAA = 1
cost_randNS = lambda x: cost_control_noise(x, bounds_abs, noise, N_SAA, \
                                    x0 = [.116, 368.489], N = 200, T = 20, NS = True)
cost_unconNS = lambda x: cost_randNS(x)[0]  

cost_rand = lambda x: cost_control_noise(x, bounds_abs, noise, N_SAA, \
                                    x0 = [.116, 368.489], N = 200, T = 20)
cost_uncon = lambda x: cost_rand(x)[0]  

N = 10

x0_abs = np.array(pi_init)

bounds = np.array([[0, 1]]*10)
 
x0 = (np.array(pi_init) - bounds_abs[:,0]) / (bounds_abs[:,1]-bounds_abs[:,0])     

# ContrSynRand_Bayes_list = []
# for i in range(N):
#     Bayes = BayesOpt()
#     pyro.set_rng_seed(i)
        
#     ContrSynRand_Bayes = Bayes.solve(cost_uncon, x0, acquisition='EI',bounds=bounds.T, \
#                             print_iteration = True, constraints=0, casadi=True, \
#                             maxfun = nbr_feval, ).output_dict
#     ContrSynRand_Bayes_list.append(ContrSynRand_Bayes)
 
# print('10 SAA BayesOpt iterations completed')

# with open('BayesContrSynRand_list.pickle', 'wb') as handle:
#     pickle.dump(ContrSynRand_Bayes_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

## No Bayes for ContrLin, omly for ContrSyn

n_noise = 6
noise_matrix = np.zeros((n_noise, 2))
for i in range(n_noise):
    noise_matrix[i] = np.array([.001, 1])*i/3

nbr_feval = 30 ; N_SAA = 1
N_samples = 20
ContrSynNoise_list_Bayes = []
for i in range(n_noise):
    Bayes = BayesOpt()
    print('Outer Iteration ', i+1, ' out of ', n_noise,' of BayesOpt')
    best = []
    cost_rand = lambda x: cost_control_noise(x, bounds_abs, noise_matrix[i], N_SAA, \
                                    x0 = [.116, 368.489], N = 200, T = 20)
    cost_uncon = lambda x: cost_rand(x)[0] 
    for j in range(N_samples):
        sol = Bayes.solve(cost_uncon, x0, acquisition='EI',bounds=bounds.T, \
                            print_iteration = True, constraints=0, casadi=True, \
                            maxfun = nbr_feval, ).output_dict
        best.append(sol['f_best_so_far'][-1])
    ContrSynNoise_list_Bayes.append(best)

with open('BayesContrSyn_listNoiseConv.pickle', 'wb') as handle:
    pickle.dump(ContrSynNoise_list_Bayes, handle, protocol=pickle.HIGHEST_PROTOCOL)

   
nbr_feval = 25 ; N_SAA = 2
N_samples = 20
ContrSynNoiseSAA_list_Bayes = []
for i in range(n_noise):
    Bayes = BayesOpt()
    print('Outer Iteration ', i+1, ' out of ', n_noise,' of BayesOpt')
    best = []
    cost_rand = lambda x: cost_control_noise(x, bounds_abs, noise_matrix[i], N_SAA, \
                                    x0 = [.116, 368.489], N = 200, T = 20)
    cost_uncon = lambda x: cost_rand(x)[0] 
    for j in range(N_samples):
        sol = Bayes.solve(cost_uncon, x0, acquisition='EI',bounds=bounds.T, \
                            print_iteration = True, constraints=0, casadi=True, \
                            maxfun = nbr_feval, ).output_dict
        best.append(sol['f_best_so_far'][-1])
    ContrSynNoiseSAA_list_Bayes.append(best)

with open('BayesContrSynSAA_listNoiseConv.pickle', 'wb') as handle:
    pickle.dump(ContrSynNoiseSAA_list_Bayes, handle, protocol=pickle.HIGHEST_PROTOCOL)


