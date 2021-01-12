import numpy as np 
    
def simplex_method(f,x0,bounds,max_iter,constraints):
    '''
    INPUTS
    ------------------------------------
    f:          function to be optimised
    
    x0:         initial guess in form [x1,x2,...,xn]
    
    bounds:     variable upper and lower bounds in form
                [[x1l,x1u],[x2l,x2u],...,[xnl,xnu]]
                
    max_iter:   total optimisation iterations due to 
                no stopping conditions
    
    constraints: constraint functions in form [g1,g2,...,gm]
                all have form g(x) <= 0 and return g(x)
                
    OUTPUTS
    ------------------------------------
    output_dict: 
        - 'x'           : final input variable
        
        - 'f'           : final function value
        
        - 'g'           : value of each constraint function at optimal solution
        
        - 'f_evals'     : total number of function evaluations
        
        - 'f_store'     : best function value at each iteration
                            
        - 'x_store'     : list of all previous best variables (per iteration)
                            
        - 'g_store'     : list of all previous constraint values (per iteration)
        
        - 'g_viol'      : total constraint violation (sum over constraints)
    
    NOTES
    --------------------------------------
     - Only function values that contribute towards the optimisation are counted.
     - Stored function values at each iteration are not the penalised objective
        but the objective function itself.
    '''
    bounds = np.array(bounds)   # converting to numpy if not 
    d = len(x0)                 # dimension 
    con_d = len(constraints)
    f_range = (bounds[:,1] - bounds[:,0])*0.1       # range of initial simplex
    x_nodes = np.random.normal(x0,f_range,(d+1,d))  # creating nodes
    f_nodes = np.zeros((len(x_nodes[:,0]),1))       # function value at each node
    f_eval_count = 0            # initialising total function evaluation counter
    con_weight = 10000           # setting constraint penalty
    f_store = np.zeros(max_iter)      # initialising function store
    g_store = np.zeros((max_iter,con_d))
    x_store = np.zeros((max_iter,d)) # initialising x_store
    # evaluating function 
    for i in range(d+1):
        f_nodes[i,:] = f(x_nodes[i,:]) 
        # evaulating constraints as penalty
        for ii in range(len(constraints)):
            f_nodes[i,:] += con_weight * max(0,constraints[ii](x_nodes[i,:]))
        f_eval_count += 1 
    for its in range(max_iter):
        sorted_nodes = np.argsort(f_nodes[:,0])
        best_nodes = x_nodes[sorted_nodes[:-1]]
        
        # storing important quantities
        best_node = x_nodes[sorted_nodes[0]]
        x_store[its,:] = best_node
        f_store[its] = f(best_node)
        con_it = [0 for i in range(len(constraints))]
        for i in range(len(con_it)):
            con_it[i] = constraints[i](best_node)
        g_store[its,:] = con_it 

        # centroid of all bar worst nodes
        centroid = np.mean(best_nodes,axis=0)
        # reflection of worst node
        x_reflected = centroid + (centroid - x_nodes[sorted_nodes[-1],:])
        f_reflected =  f(x_reflected) 

        for ii in range(len(constraints)):
            f_reflected += con_weight * max(0,constraints[ii](x_reflected))
        f_eval_count += 1 
        # accept reflection? 
        if f_reflected < f_nodes[sorted_nodes[-2]] and \
            f_reflected > f_nodes[sorted_nodes[0]]:
                x_nodes[sorted_nodes[-1],:] = x_reflected
                f_nodes[sorted_nodes[-1],:] = f_reflected
        # try expansion of reflected then accept? 
        elif f_reflected < f_nodes[sorted_nodes[0]]:
                x_expanded = centroid + 2*(x_reflected-centroid)
                f_expanded = f(x_expanded)
                for ii in range(len(constraints)):
                    f_expanded += con_weight * max(0,constraints[ii](x_expanded))
                f_eval_count += 1 
                if f_expanded < f_reflected:
                    x_nodes[sorted_nodes[-1],:] = x_expanded
                    f_nodes[sorted_nodes[-1],:] = f_expanded
                else: # ...expansion worse so accept reflection 
                    x_nodes[sorted_nodes[-1],:] = x_reflected
                    f_nodes[sorted_nodes[-1],:] = f_reflected
        else: # all else fails, contraction of worst internal of simplex
            x_contracted = centroid + 0.5*(x_nodes[sorted_nodes[-1],:]-centroid)
            f_contracted = f(x_contracted)
            for ii in range(len(constraints)):
                f_contracted += con_weight * max(0,constraints[ii](x_contracted))
            f_eval_count += 1
            if f_contracted < f_nodes[sorted_nodes[-1]]:
                x_nodes[sorted_nodes[-1],:] = x_contracted
                f_nodes[sorted_nodes[-1],:] = f_contracted
   # computing final constraint violation 
    con_viol = [0 for i in range(len(constraints))]
    act_cons = [0 for i in range(len(constraints))]
    for i in range(len(constraints)):
        con_val = constraints[i](x_nodes[sorted_nodes[0]])
        con_viol[i] = con_val 
    con_viol_total = sum(max(0,con_viol[i]) for i in range(len(con_viol)))         
    output_dict = {}
    output_dict['x'] = x_nodes[sorted_nodes[0]]
    output_dict['f'] = f(x_nodes[sorted_nodes[0]])
    output_dict['g'] = con_viol 
    output_dict['g_viol'] =  con_viol_total
    output_dict['g_store'] = g_store
    output_dict['x_store'] = x_store
    output_dict['f_store'] = f_store 
    output_dict['f_evals'] = f_eval_count
    
    
    return output_dict

