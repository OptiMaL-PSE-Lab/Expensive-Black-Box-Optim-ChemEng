# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 15:49:18 2021

@author: dv516
"""


from algorithms.Bayesian_opt_Pyro.utilities_full import BayesOpt
from case_studies.RTO.systems import *

import numpy as np
import pickle

import pyro
pyro.enable_validation(True)  # can help with debugging

def RTO(x):
    # x = extract_FT(x)
    plant = WO_system()
    f = plant.WO_obj_sys_ca_noise_less
    g1 = plant.WO_con1_sys_ca_noise_less
    g2 = plant.WO_con2_sys_ca_noise_less
    return f(x), [g1(x), g2(x)]

def RTO_rand(x):
    # x = extract_FT(x)
    plant = WO_system()
    f = plant.WO_obj_sys_ca
    g1 = plant.WO_con1_sys_ca
    g2 = plant.WO_con2_sys_ca
    return f(x), [g1(x), g2(x)]

def RTO_SAA(x):
    # x = extract_FT(x)
    N_SAA = 5
    
    plant = WO_system()
    
    f = plant.WO_obj_sys_ca
    g1 = plant.WO_con1_sys_ca
    g2 = plant.WO_con2_sys_ca
    f_SAA = 0
    g1_SAA, g2_SAA = - np.inf, - np.inf
    
    for i in range(N_SAA):
        f_SAA += f(x)/N_SAA
        g1_SAA = max(g1_SAA, g1(x))
        g2_SAA = max(g2_SAA, g2(x))
    
    return f_SAA, [g1_SAA, g2_SAA]

def RTO_Noise(x, noise, N_SAA):
    
    plant = WO_system()
    
    f = plant.WO_obj_sys_ca
    g1 = plant.WO_con1_sys_ca
    g2 = plant.WO_con2_sys_ca
    f_SAA = 0
    g1_SAA, g2_SAA = - np.inf, - np.inf
    
    for i in range(N_SAA):
        f_SAA += (f(x) +  5e-1 * np.random.normal(0., noise))/N_SAA
        g1_SAA = max(g1_SAA, g1(x) +  5e-4 * np.random.normal(0., noise))
        g2_SAA = max(g2_SAA, g2(x) +  5e-4 * np.random.normal(0., noise))
    
    return f_SAA, [g1_SAA, g2_SAA]

x0 = [6.9, 83]
bounds  = np.array([[4., 7.], [70., 100.]])

# max_f_eval = 100
# max_it = 50
nbr_feval = 50

N = 10
RTO_Bayes_list = []
for i in range(N):
    Bayes = BayesOpt()
    pyro.set_rng_seed(i)
    
    RTO_Bayes = Bayes.solve(RTO, x0, acquisition='EI',bounds=bounds.T, \
                            print_iteration = True, constraints=2, casadi=True, \
                            maxfun = nbr_feval, ).output_dict
    RTO_Bayes_list.append(RTO_Bayes)
 
print('10 BayesOpt deterministic iterations completed')

with open('BayesRTO_list.pickle', 'wb') as handle:
    pickle.dump(RTO_Bayes_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

N = 10
RTORand_Bayes_list = []
for i in range(N):
    Bayes = BayesOpt()
    pyro.set_rng_seed(i)
    
    RTORand_Bayes = Bayes.solve(RTO_rand, x0, acquisition='EI',bounds=bounds.T, \
                            print_iteration = True, constraints=2, casadi=True, \
                            maxfun = nbr_feval, ).output_dict
    RTORand_Bayes_list.append(RTORand_Bayes)
 
print('10 BayesOpt random iterations completed')

with open('BayesRTO_listRand.pickle', 'wb') as handle:
    pickle.dump(RTORand_Bayes_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
 
N = 10
RTOSAA_Bayes_list = []
for i in range(N):
    Bayes = BayesOpt()
    pyro.set_rng_seed(i)
    
    RTOSAA_Bayes = Bayes.solve(RTO_SAA, x0, acquisition='EI',bounds=bounds.T, \
                            print_iteration = True, constraints=2, casadi=True, \
                            maxfun = nbr_feval, ).output_dict
    RTOSAA_Bayes_list.append(RTOSAA_Bayes)
 
print('10 BayesOpt SAA iterations completed')

with open('BayesRTO_listRandSAA.pickle', 'wb') as handle:
    pickle.dump(RTOSAA_Bayes_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

n_noise = 6
noise_mat = np.zeros(n_noise)
for i in range(n_noise):
    noise_mat[i] = 1/3*i

bounds = np.array([[-1.5,1.5],[-1.5,1.5]])
x0 = np.array([-0.5,1.5])
max_f_eval = 50 ; N_SAA = 1

 
N_SAA = 1
N_samples = 20
RTOnoise_list_Bayes = []
RTOconstraint_list_Bayes = []
for i in range(n_noise):
    print('Outer Iteration ', i+1, ' out of ', n_noise,' of BayesOpt')
    best = []
    best_constr = []
    for j in range(N_samples):
        f = lambda x: RTO_Noise(x, noise_mat[i], N_SAA)
        sol = Bayes.solve(f, x0, acquisition='EI',bounds=bounds.T, \
                            print_iteration = True, constraints=2, casadi=True, \
                            maxfun = nbr_feval, ).output_dict
        best.append(sol['f_best_so_far'][-1])
        _, g = RTO_Noise(sol['x_best_so_far'][-1], 0, N_SAA)
        best_constr.append(np.sum(np.maximum(g, 0)))
    RTOnoise_list_Bayes.append(best)
    RTOconstraint_list_Bayes.append(best_constr)

with open('BayesRTO_listNoiseConv.pickle', 'wb') as handle:
    pickle.dump(RTOnoise_list_Bayes, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('BayesRTO_listNoiseConstr.pickle', 'wb') as handle:
    pickle.dump(RTOconstraint_list_Bayes, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

N_SAA = 2
N_samples = 20
RTOnoiseSAA_list_Bayes = []
RTOconstraintSAA_list_Bayes = []
for i in range(n_noise):
    print('Outer Iteration ', i+1, ' out of ', n_noise,' of BayesOpt')
    best = []
    best_constr = []
    for j in range(N_samples):
        f = lambda x: RTO_Noise(x, noise_mat[i], N_SAA)
        sol = Bayes.solve(f, x0, acquisition='EI',bounds=bounds.T, \
                            print_iteration = True, constraints=2, casadi=True, \
                            maxfun = nbr_feval, ).output_dict
        best.append(sol['f_best_so_far'][-1])
        _, g = RTO_Noise(sol['x_best_so_far'][-1], 0, N_SAA)
        best_constr.append(np.sum(np.maximum(g, 0)))
    RTOnoiseSAA_list_Bayes.append(best)
    RTOconstraintSAA_list_Bayes.append(best_constr)

with open('BayesRTO_listNoiseConvSAA.pickle', 'wb') as handle:
    pickle.dump(RTOnoiseSAA_list_Bayes, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('BayesRTO_listNoiseConstrSAA.pickle', 'wb') as handle:
    pickle.dump(RTOconstraintSAA_list_Bayes, handle, protocol=pickle.HIGHEST_PROTOCOL)
    



