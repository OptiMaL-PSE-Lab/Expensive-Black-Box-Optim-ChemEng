# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 22:09:36 2021

@author: dv516
"""

import numpy as np
import pickle

import pyro
pyro.enable_validation(True)  # can help with debugging
pyro.set_rng_seed(1)

from algorithms.Bayesian_opt_Pyro.utilities_full import BayesOpt

from case_studies.MBDoE.construct_MBDoE_funs import set_funcs_mbdoe

bounds = np.array([[0,1.],[0,1.],[0,1.],[0,1.]])
x0 = np.array([0.1]*4)

cost_BO = lambda x: set_funcs_mbdoe(x)[0]

nbr_feval = 30

N = 10
MBDoE_Bayes_list = []
for i in range(N):
    Bayes = BayesOpt()
    pyro.set_rng_seed(i)
    print('Iteration ', i+1, ' out of ', N)
    MBDoE_Bayes = Bayes.solve(cost_BO, x0, acquisition='EI',bounds=bounds.T, \
                            print_iteration = True, constraints=0, casadi=True, \
                            maxfun = nbr_feval, ).output_dict
    MBDoE_Bayes_list.append(MBDoE_Bayes)

with open('BayesMBDoE_list.pickle', 'wb') as handle:
    pickle.dump(MBDoE_Bayes_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    