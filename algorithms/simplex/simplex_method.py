import numpy as np 
import matplotlib.pyplot as plt 
from itertools import combinations,product
import sys
sys.path.insert(1, 'test_functions')
from quadratic_constrained import * 
from rosenbrock_constrained import * 

    
def simplex_method(f,x0,bounds,max_iter,constraints):
    iterations = max_iter
    d = len(x0) # dimension 
    f_range = (bounds[:,1] - bounds[:,0])*0.1 # range of initial simplex
    x_nodes = np.random.normal(x0,f_range,(d+1,d)) # creating nodes
    f_nodes = np.zeros((len(x_nodes[:,0]),1)) # function value at each node
    f_eval_count = 0 
    f_store = np.array([])
    x_store = np.array([x_nodes[0,:]])
    # evaluating function 
    for i in range(d+1):
        f_nodes[i,:] = f(x_nodes[i,:]) 
        f_eval_count += 1 
    for its in range(1,iterations+1):
        sorted_nodes = np.argsort(f_nodes[:,0])
        best_nodes = x_nodes[sorted_nodes[:-1]]
        x_store = np.append(x_store,[x_nodes[sorted_nodes[0]]],axis=0)
        f_store = np.append(f_store,min(f_nodes))

        # centroid of all bar worst nodes
        centroid = np.mean(best_nodes,axis=0)
        # reflection of worst node
        x_reflected = centroid + (centroid - x_nodes[sorted_nodes[-1],:])
        f_reflected =  f(x_reflected) 
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
            f_eval_count += 1
            if f_contracted < f_nodes[sorted_nodes[-1]]:
                x_nodes[sorted_nodes[-1],:] = x_contracted
                f_nodes[sorted_nodes[-1],:] = f_contracted
    
    output_dict = {}
    output_dict['x'] = x_nodes[sorted_nodes[0]]
    output_dict['x_store'] = x_store
    output_dict['f_store'] = f_store 
    output_dict['f_evals'] = f_eval_count
    
    return output_dict

