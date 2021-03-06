import numpy as np 
# import sys
# sys.path.insert(1, 'utilities')
from utilities.general_utility_functions import PenaltyFunctions

def project_to_bounds(x, bounds):
    node = x
    for j in range(len(x)):
        if node[j] < bounds[j,0]:
            node[j] = bounds[j,0]
        if node[j] > bounds[j,1]:
            node[j] = bounds[j,1]
    return node

def extract_best(x_store, f_store, g_store, samples_at_iteration):
    N_f = len(f_store)
    _, N_x = x_store.shape
    _, N_g = g_store.shape
    N_it = len(samples_at_iteration)
    x_best = np.zeros((N_it, N_x))
    f_best = np.zeros(N_it)
    g_best = np.zeros((N_it, N_g))
    for i in range(N_it):
        nbr_samples = samples_at_iteration[i]
        feas = np.product((np.array(g_store[:int(nbr_samples)]) <= 0).astype(int), axis = 1)
        idx = np.where(f_store == np.min(f_store[:int(nbr_samples)][feas == 1]))
        # print(idx)
        f_best[i] = f_store[idx[0][0]]     # initialising function store
        x_best[i] = x_store[idx[0][0]]
        g_best[i] = g_store[idx[0][0]]
    return x_best, f_best, g_best
    


def simplex_method(f,x0,bounds,max_iter,constraints, max_f_eval = 100, \
                   mu_con = 1e3, rnd_seed = 0, initialisation = 0.1):
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
    f_range = (bounds[:,1] - bounds[:,0])*initialisation # range of initial simplex
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

    for i in range(len(x_nodes)):
        node = x_nodes[i,:]
        x_nodes[i,:] = project_to_bounds(node, bounds) 
            

    # evaluating function 
    for i in range(d+1):
        f_nodes[i,:] = f_aug.aug_obj(x_nodes[i,:])  
        f_eval_count += 1 
        
    
    for its in range(max_iter):
        

        
        sorted_nodes = np.argsort(f_nodes[:,0])
        best_nodes = x_nodes[sorted_nodes[:-1]]
        
        best_node = x_nodes[sorted_nodes[0]]
        f_evalled = f_aug.f(best_node)
        f_eval_count += 1
        # storing important quantities
        if its == 0:
            x_store = [best_node]
            f_store = [f_evalled[0]]
            g_store = [f_evalled[1]]
        else:
            x_store = np.append(x_store,[best_node],axis=0)
            f_store = np.append(f_store,f_evalled[0])
            g_store = np.append(g_store,[f_evalled[1]],axis=0)

        
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
        x_provis = centroid + (centroid - x_nodes[sorted_nodes[-1],:])
        x_reflected = project_to_bounds(x_provis, bounds) 
        f_reflected =  f_aug.aug_obj(x_reflected) 
        # adding hypothesised function eval (have to re-eval to get f,g separate?)
        f_total_func = f_aug.f(x_reflected)
        x_store = np.append(x_store,[x_reflected],axis=0)
        f_store = np.append(f_store,f_total_func[0])
        g_store = np.append(g_store,[f_total_func[1]],axis=0)

        f_eval_count += 1 
        # accept reflection? 
        if f_reflected < f_nodes[sorted_nodes[-2]] and \
            f_reflected > f_nodes[sorted_nodes[0]]:
                x_nodes[sorted_nodes[-1],:] = x_reflected
                f_nodes[sorted_nodes[-1],:] = f_reflected
        # try expansion of reflected then accept? 
        elif f_reflected < f_nodes[sorted_nodes[0]]:
                x_provis =   centroid + 2*(x_reflected-centroid)
                x_expanded = project_to_bounds(x_provis, bounds) 
                f_expanded = f_aug.aug_obj(x_expanded)

                # adding hypothesised function eval (have to re-eval to get f,g separate?)
                f_total_func = f_aug.f(x_expanded)
                x_store = np.append(x_store,[x_expanded],axis=0)
                f_store = np.append(f_store,f_total_func[0])
                g_store = np.append(g_store,[f_total_func[1]],axis=0)

                f_eval_count += 1 
                if f_expanded < f_reflected:
                    x_nodes[sorted_nodes[-1],:] = x_expanded
                    f_nodes[sorted_nodes[-1],:] = f_expanded
                else: # ...expansion worse so accept reflection 
                    x_nodes[sorted_nodes[-1],:] = x_reflected
                    f_nodes[sorted_nodes[-1],:] = f_reflected
        else: # all else fails, contraction of worst internal of simplex
            x_provis = centroid + 0.5*(x_nodes[sorted_nodes[-1],:]-centroid) 
            x_contracted = project_to_bounds(x_provis, bounds) 
            f_contracted = f_aug.aug_obj(x_contracted)

            # adding hypothesised function eval (have to re-eval to get f,g separate?)
            f_total_func = f_aug.f(x_contracted)
            x_store = np.append(x_store,[x_contracted],axis=0)
            f_store = np.append(f_store,f_total_func[0])
            g_store = np.append(g_store,[f_total_func[1]],axis=0)

            f_eval_count += 1
            if f_contracted < f_nodes[sorted_nodes[-1]]:
                x_nodes[sorted_nodes[-1],:] = x_contracted
                f_nodes[sorted_nodes[-1],:] = f_contracted
             
        nbr_samples[its] = f_eval_count
        # for i in range(len(x_nodes)):
        #     node = x_nodes[i,:]
        #     for j in range(len(node)):
        #         if node[j] < bounds[j,0]:
        #             node[j] = bounds[j,0]
        #         if node[j] > bounds[j,1]:
        #             node[j] = bounds[j,1]
        #     x_nodes[i,:] = node 
        if f_eval_count >= max_f_eval:
            break
        
   # computing final constraint violation 
    output_dict = {}
    output_dict['g_store'] = g_store
    output_dict['x_store'] = x_store
    output_dict['f_store'] = f_store
    output_dict['N_evals'] = f_eval_count
    output_dict['samples_at_iteration'] = nbr_samples[:its+1]
    x_best, f_best, g_best = extract_best(x_store, f_store, g_store, nbr_samples[:its+1])
    output_dict['g_best_so_far'] = g_best
    output_dict['x_best_so_far'] = x_best
    output_dict['f_best_so_far'] = f_best
    
    return output_dict
