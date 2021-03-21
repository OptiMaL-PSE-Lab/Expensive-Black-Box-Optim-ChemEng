import numpy as np 
# import sys
# sys.path.insert(1, 'utilities')
from utilities.general_utility_functions import PenaltyFunctions
    
def simplex_method(f,x0,bounds,max_iter,constraints, max_f_eval = 100, \
                   mu_con = 1e3, rnd_seed = 0):
    '''
    INPUTS
    ------------------------------------
    f:          function to be optimised
    
    x0:         initial guess in form [x1,x2,...,xn]
    
    bounds:     variable upper and lower bounds in form
                [[x1l,x1u],[x2l,x2u],...,[xnl,xnu]]
                
    max_iter:   total optimisation iterations due to 
                no stopping conditions
    
    constraints: the number of constraints
                
    OUTPUTS
    ------------------------------------
    output_dict: 
        - 'N_evals'     : total number of function evaluations
        
        - 'f_store'     : best function value at each iteration
                            
        - 'x_store'     : list of all previous best variables (per iteration)
                            
        - 'g_store'     : list of all previous constraint values (per iteration)
        
        - 'f_best_so_far'     : best function value of all previous iterations
                            
        - 'g_best_so_far'     : constraint violation of previous best f 
                            
        - 'x_best_so_far'     : variables of best function value across previous its
        
    NOTES
    --------------------------------------
     - Only function values that contribute towards the optimisation are counted.
     - Stored function values at each iteration are not the penalised objective
        but the objective function itself.
    '''
    np.random.seed(rnd_seed)
    
    f_aug = PenaltyFunctions(f,type_penalty='l2', mu= mu_con)
    
    bounds = np.array(bounds)   # converting to numpy if not 
    d = len(x0)                 # dimension 
    con_d = constraints
    f_range = (bounds[:,1] - bounds[:,0])*0.1       # range of initial simplex
    x_nodes = np.random.normal(x0,f_range,(d+1,d))  # creating nodes
    f_nodes = np.zeros((len(x_nodes[:,0]),1))       # function value at each node
    f_eval_count = 0            # initialising total function evaluation counter
    f_store = np.zeros(max_iter)            # initialising function store
    g_store = np.zeros((max_iter,con_d))
    x_store = np.zeros((max_iter,d))        # initialising x_store
    f_best_so_far = np.zeros(max_iter)      # initialising function store
    x_best_so_far = np.zeros((max_iter,d))
    g_best_so_far = np.zeros((max_iter,con_d))
    nbr_samples = np.zeros(max_iter)


    # evaluating function 
    for i in range(d+1):
        f_nodes[i,:] = f_aug.aug_obj(x_nodes[i,:])  
        f_eval_count += 1 
        
    
    for its in range(max_iter):
        
        
        
        sorted_nodes = np.argsort(f_nodes[:,0])
        best_nodes = x_nodes[sorted_nodes[:-1]]
        
        # storing important quantities
        best_node = x_nodes[sorted_nodes[0]]
        x_store[its,:] = best_node
        f_evalled = f_aug.f(best_node)
        f_store[its] = f_evalled[0]
        g_store[its,:] = f_evalled[1]
        
        if its == 0:
            f_best_so_far[its] = f_store[its] 
            x_best_so_far[its] = x_store[its]
            g_best_so_far[its] = g_store[its]
        else:
            f_best_so_far[its] = f_store[its] 
            x_best_so_far[its] = best_node
            g_best_so_far[its] = g_store[its]
            if f_best_so_far[its] > f_best_so_far[its-1]:
                f_best_so_far[its] = f_best_so_far[its-1]
                x_best_so_far[its] = x_best_so_far[its-1]
                g_best_so_far[its] = g_best_so_far[its-1]

        # centroid of all bar worst nodes
        centroid = np.mean(best_nodes,axis=0)
        # reflection of worst node
        x_reflected = centroid + (centroid - x_nodes[sorted_nodes[-1],:])
        f_reflected =  f_aug.aug_obj(x_reflected) 
        f_eval_count += 1 
        # accept reflection? 
        if f_reflected < f_nodes[sorted_nodes[-2]] and \
            f_reflected > f_nodes[sorted_nodes[0]]:
                x_nodes[sorted_nodes[-1],:] = x_reflected
                f_nodes[sorted_nodes[-1],:] = f_reflected
        # try expansion of reflected then accept? 
        elif f_reflected < f_nodes[sorted_nodes[0]]:
                x_expanded = centroid + 2*(x_reflected-centroid)
                f_expanded = f_aug.aug_obj(x_expanded)
                f_eval_count += 1 
                if f_expanded < f_reflected:
                    x_nodes[sorted_nodes[-1],:] = x_expanded
                    f_nodes[sorted_nodes[-1],:] = f_expanded
                else: # ...expansion worse so accept reflection 
                    x_nodes[sorted_nodes[-1],:] = x_reflected
                    f_nodes[sorted_nodes[-1],:] = f_reflected
        else: # all else fails, contraction of worst internal of simplex
            x_contracted = centroid + 0.5*(x_nodes[sorted_nodes[-1],:]-centroid)
            f_contracted = f_aug.aug_obj(x_contracted)
            f_eval_count += 1
            if f_contracted < f_nodes[sorted_nodes[-1]]:
                x_nodes[sorted_nodes[-1],:] = x_contracted
                f_nodes[sorted_nodes[-1],:] = f_contracted
             
        nbr_samples[its] = f_eval_count
        if f_eval_count >= max_f_eval:
            break
        
   # computing final constraint violation 
    output_dict = {}
    output_dict['g_store'] = g_store[:its+1]
    output_dict['x_store'] = x_store[:its+1]
    output_dict['f_store'] = f_store[:its+1]
    output_dict['g_best_so_far'] = g_best_so_far[:its+1]
    output_dict['x_best_so_far'] = x_best_so_far[:its+1]
    output_dict['f_best_so_far'] = f_best_so_far[:its+1]
    output_dict['N_evals'] = f_eval_count
    output_dict['samples_at_iteration'] = nbr_samples[:its+1]
    
    return output_dict
