import numpy as np 
# import sys
# sys.path.insert(1, 'utilities')
from utilities.general_utility_functions import PenaltyFunctions


def nesterov_random(f,x0,bounds,max_iter,constraints, alpha = 0.001, mu = 0.001, \
                    max_f_eval = 100, rnd_seed = 0):
    np.random.seed(rnd_seed)
    
    f_aug = PenaltyFunctions(f,type_penalty='l2',mu=1e3)
    
    d = len(x0)
    con_d = constraints
    x_store = np.zeros((max_iter*3,d))
    g_store = np.zeros((max_iter*3,con_d))
    f_store = np.zeros(max_iter*3)
    f_best_so_far = np.zeros(max_iter)      # initialising function store
    x_best_so_far = np.zeros((max_iter,d))
    g_best_so_far = np.zeros((max_iter,con_d))
    nbr_samples = np.zeros(max_iter)
    x = x0
    f_eval_count = 0 
    for i in range(max_iter):

        x_store[i*3,:] = x
        f = f_aug.f(x)
        f_store[i*3] = f[0]
        g_store[i*3] = f[1]
        f_eval_count += 1
        
        u = np.random.normal(0,1,(1,d))
        B = np.eye(d)
        # B = np.array([[1,0],[0,1]])
        forw = (x + mu * u)[0,:]
        back = (x - mu * u)[0,:]
        
        
        f_f = f = f_aug.f(forw)
        x_store[i*3+1] = forw
        f_store[i*3+1] = f_f[0]
        g_store[i*3+1] = f_f[1]
        
        f_b = f_aug.f(back)
        x_store[i*3+2] = back
        f_store[i*3+2] = f_b[0]
        g_store[i*3+2] = f_b[1]
                             
        f_forw = f_aug.aug_obj(forw)
        f_eval_count += 1
            
        f_back = f_aug.aug_obj(back)
        f_eval_count += 1
            
        g = ((f_forw - f_back)/(2*mu))*B@u.T
        x = x - alpha * (g.T)[0,:]
        
        feas = np.product( (np.array(g_store[:(i+1)*3]) <= 0).astype(int), axis = 1)
        # print(i, f_store)
        ind = np.where(f_store == np.min(f_store[:(i+1)*3][feas == 1]))
    
        
        f_best_so_far[i] = f_store[ind]     # initialising function store
        x_best_so_far[i] = x_store[ind]
        g_best_so_far[i] = g_store[ind]
        nbr_samples[i] = 3*(i+1)
        
        if f_eval_count >= max_f_eval:
            break
        
    output_dict = {}
    output_dict['g_store'] = g_store[:(i+1)*3]
    output_dict['x_store'] = x_store[:(i+1)*3]
    output_dict['f_store'] = f_store[:(i+1)*3]
    output_dict['g_best_so_far'] = g_best_so_far[:i+1]
    output_dict['x_best_so_far'] = x_best_so_far[:i+1]
    output_dict['f_best_so_far'] = f_best_so_far[:i+1]
    output_dict['N_evals'] = f_eval_count
    output_dict['samples_at_iteration'] = nbr_samples[:i+1]
    
    return output_dict
    
