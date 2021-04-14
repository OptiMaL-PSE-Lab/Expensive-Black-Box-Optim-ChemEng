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

### Change this
bounds = np.array([[0,1.],[0,1.],[0,1.],[0,1.]])
x0 = np.array([0.1]*4)
###

### Include function here, remember that it should return f(x), [g_1(x), ...]
f = lambda x: x
###

nbr_feval = 30

### If any of them don't work for now, comment them out for now and let me 
### know which ones aren't working. Will see then if the method is just poor
### or if there's a problem with the code

### I would probably start with a direct method like DIRECT to start with, as
### they are at least not as sensitive to ill-conditioning

### Don't forget to set this to the number of constraints
n_constr = 1
###

N = 10
FS_Bayes_list = []
for i in range(N):
    Bayes = BayesOpt()
    pyro.set_rng_seed(i)
    print('Iteration ', i+1, ' out of ', N)
    FS_Bayes = Bayes.solve(f, x0, acquisition='EI',bounds=bounds.T, \
                            print_iteration = True, constraints= n_constr, casadi=True, \
                            maxfun = nbr_feval, ).output_dict
    FS_Bayes_list.append(FS_Bayes)

with open('BayesFS_list.pickle', 'wb') as handle:
    pickle.dump(FS_Bayes_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    