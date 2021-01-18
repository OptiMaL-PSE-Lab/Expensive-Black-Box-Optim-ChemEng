import numpy as np 
import sys
sys.path.insert(1, 'utilities')
from general_utility_functions import PenaltyFunctions


def nesterov_random(f,x0,bounds,max_iter,constraints):
    f_aug = PenaltyFunctions(f,constraints,type_penalty='l2',mu=100)
    d = len(x0)
    con_d = len(constraints)
    x_store = np.zeros((max_iter,d))
    g_store = np.zeros((max_iter,con_d))
    f_store = np.zeros(max_iter)
    x = x0
    f_eval_count = 0 
    alpha = 0.001
    mu = 0.01
    for i in range(max_iter):

        x_store[i,:] = x
        f_store[i] = f_aug.f(x)
        
        u = np.random.normal(0,1,(1,d))
        B = np.array([[1,0],[0,1]])
        forw = (x + mu * u)[0,:]
        back = (x - mu * u)[0,:]
        
        f_forw = f_aug.aug_obj(forw)
        f_eval_count += 1
            
        f_back = f_aug.aug_obj(back)
        f_eval_count += 1
            
        g = ((f_forw - f_back)/(2*mu))*B@u.T
        x = x - alpha * (g.T)[0,:]
        
    output_dict = {}
    output_dict['g_store'] = g_store
    output_dict['x_store'] = x_store
    output_dict['f_store'] = f_store 
    output_dict['N_evals'] = f_eval_count
    return output_dict
    
