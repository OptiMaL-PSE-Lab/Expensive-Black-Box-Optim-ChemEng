# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 21:32:39 2021

@author: dv516
"""

import numpy as np
import matplotlib.pyplot as plt
# from scipy.optimize import minimize

def evaluate_bounds(x, b_constr, con_weight = 100):
    bounds_list = [[g_(x) for g_ in b_constr]]
    return con_weight*np.sum(np.maximum(0, bounds_list))**2

def add_penalty(sim_out, b_list = [], con_weight = 100):
    f_val, g_list = sim_out
    f_val += con_weight*np.sum(np.maximum(0, g_list)**2)
    if b_list != []:
        f_val += con_weight*np.sum(np.maximum(0, b_list)**2)
    return f_val

def is_in_bounds(x, bounds):
    if bounds is not None:
        for i in range(len(bounds)):
            if x[i] < bounds[i][0] or x[i] > bounds[i][1]:
                return False
        return True
    else:
        return True

def find_t(t, x0, delta_x, bounds, beta):
    x = x0 + t*delta_x
    i = 0
    while (not is_in_bounds(x, bounds)) and i < 10:
        t *= beta
        x = x0 + t*delta_x
        i += 1
    return x, t

def backtracking(sim, x, delta_x, x_list, f_list, g_list, current_f, gradient, \
                 constraints, bounds, alpha = 0.1, beta = 0.25, con_weight = 100):
    t = 1
    # x_new, t = find_t(1, x, delta_x, bounds, beta)
    x_new = x + t*delta_x
    # print('t: ', t)
    f_lhs, g_lhs = sim(x_new)
    x_list += [x_new.tolist()]
    f_list += [f_lhs]
    g_list += [g_lhs]
    
    if constraints != []:
        b = evaluate_bounds(x, constraints, con_weight = con_weight)
    else:
        b = []
    
    lhs = add_penalty((f_lhs, g_lhs), b_list = b, con_weight = con_weight)
    rhs = current_f + alpha*t*float(gradient @ delta_x.reshape(-1,1))
    N = 0
    
    ### Help
    
    while (N<5) and not ((lhs < rhs) and (np.sum(np.maximum(0, g_list[-1])**2) < 1e-18)):
        t *= beta
        x_new = x + t*delta_x
        f_lhs, g_lhs = sim(x_new)
        x_list += [x_new.tolist()]
        f_list += [f_lhs]
        g_list += [g_lhs]
        
        rhs = current_f + alpha*t*float(gradient @ delta_x.reshape(-1,1))
        lhs = add_penalty((f_lhs, g_lhs), b_list = b, con_weight = con_weight)
        N += 1
        
    # print(N) 
    
    return x_list, f_list, g_list

def update_best_lists(X_list, f_list, g_list, X_best, f_best, g_best, \
                      con_weight = 100):
    
    X_arr = np.array(X_list)
    f_arr = np.array(f_list)
    g_arr = np.array(g_list)
    
    f_ = f_arr + \
        con_weight*np.sum(np.maximum(0, g_arr)**2, axis = 1).reshape(f_arr.shape)
    ind_feas = np.where(np.sum(np.maximum(0, g_arr)**2, axis = 1) == 0)

    

    ind = np.where(f_[ind_feas] == np.min(f_[ind_feas]))
    # ind = np.where(f_[ind_feas] == np.min(f_arr[ind_feas]))
    X_best += X_arr[ind_feas][ind].tolist()
    f_best += f_arr[ind_feas][ind].tolist()
    g_best += g_arr[ind_feas][ind].tolist()
    
    return X_best, f_best, g_best

def finite_differences(sim, x, x_list, f_list, g_list, constraints, h = 0.001,  \
                       current_f = None, method = 'central', Hessian = True, \
                       con_weight = 100):
    
    n_dim = len(x)
    forward_eval = np.zeros(n_dim) ; backward_eval = np.zeros(n_dim)
    for i in range(n_dim):
        if (method == 'forward') or (method =='central'):
            x_forward = x.copy()
        if (method == 'backward') or (method =='central'):   
            x_backward = x.copy()
            
        if (method == 'forward') or (method =='central'):
            x_forward[i] += h 
            x_list += [list(x_forward)]
            f_eval, g_eval = sim(x_forward)
            # f_eval = f(x_forward)
            f_list += [f_eval]
            g_list += [g_eval]
            
            if constraints == []:
                b_l = []
            else:
                b_l = [[g(x_forward) for g in constraints]]
            
            forward_eval[i] = add_penalty((f_eval, g_eval), b_list = b_l, con_weight = con_weight)
                
        if (method == 'backward') or (method =='central'):    
            x_backward[i] -= h
            x_list += [list(x_backward)]
            f_eval, g_eval = sim(x_backward)
            # f_eval = f(x_forward)
            f_list += [f_eval]
            g_list += [g_eval]
            
            if constraints == []:
                b_l = []
            else:
                b_l = [[g(x_backward) for g in constraints]]
            
            backward_eval[i] = add_penalty((f_eval, g_eval), b_list = b_l, con_weight = con_weight)
 
    if (method != 'central' or Hessian) and (current_f is None):
        f_eval, g_eval = sim(x)
            # f_eval = f(x_forward)
        f_list += [f_eval]
        g_list += [g_eval]  
        
        if constraints == []:
            b_l = []
        else:
            b_l = [[g(x) for g in constraints]]
        
        current_f = add_penalty((f_eval, g_eval), b_list = b_l, con_weight = con_weight)
    
    if method == 'forward':
        gradient = (forward_eval - current_f) / (h)
    elif method == 'backward':
        gradient = (current_f - backward_eval) / (h)
    else:
        gradient = (forward_eval - backward_eval) / (2*h)
    
    if Hessian:
        H = np.zeros((n_dim, n_dim))
        for i in range(n_dim):
            for j in range(n_dim):
                
                if i == j:
                    H[i][i] = (forward_eval[i] + backward_eval[i] - 2*current_f) / h**2
                
                elif i < j:
                    xf = x.copy() ; xf[i] += h ; xf[j] += h
                    xb = x.copy() ; xb[i] -= h ; xb[j] -= h
                    
                    x_list += [list(xf), list(xb)]
                    f_f, g_f = sim(xf) ; f_b, g_b = sim(xb)
                    f_list += [f_f, f_b]
                    g_list += [g_f] + [g_b]
                    
                    if constraints == []:
                        b_f = [] ; b_b = []
                    else:
                        b_f = [[g(xf) for g in constraints]]
                        b_b = [[g(xb) for g in constraints]]
                    
                    f_f = add_penalty((f_f, g_f), b_list = b_f, con_weight = con_weight)
                    f_b = add_penalty((f_b, g_b), b_list = b_b, con_weight = con_weight)
                    
                    H[i][j] = (f_f + f_b + 2*current_f - \
                               backward_eval[i] - backward_eval[j] - \
                               forward_eval[i] - forward_eval[j]) / (2*h**2)
                    
                else:
                    H[i][j] = H[j][i]

        return gradient, H, x_list, f_list, g_list
    
    else:
        return gradient, x_list, f_list, g_list
        

def finite_Diff_Newton(sim, x0, step_size = 1, bounds = None, \
                      max_f_eval = 100, max_iter = 100, tolerance = 1e-8, \
                      con_weight = 100):
    
    constraints = []
    
    # if bounds is not None:
        
    #     bound_g = lambda x: np.sum(np.maximum(0, np.array(bounds)[:,0] - np.array(x))) + \
    #               np.sum(np.maximum(0, np.array(x) - np.array(bounds)[:,1]))
    #     constraints += [bound_g]
    
    # if constraints == []:
    #     constr = lambda x: 0
    #     constraints += [constr]
    
    x = list(x0)
    
    f_eval, g_eval = sim(x)
    
    x_list = [x]
    f_list = [f_eval]
    g_list = [g_eval]

    
    best_x = x_list.copy()
    best_f = f_list.copy()
    best_g = g_list.copy()
    
    if constraints != []:
        b = evaluate_bounds(x, constraints, con_weight = con_weight)
    else:
        b = []
    
    curr_f = add_penalty((f_eval, g_eval), b_list = b, con_weight = con_weight)
    gradient, H, x_list, f_list, g_list = finite_differences(sim, x, x_list, f_list, g_list, \
                                                     constraints, \
                                                     con_weight = con_weight, \
                                                     current_f = curr_f)
    
    # print(gradient, H)    
    
    x = np.array(x).reshape(len(x0),)
    update = - (np.linalg.inv(H) @ gradient.reshape(-1, 1)).reshape(len(x0),)
    
    x_list, f_list, g_list = backtracking(sim, x, update, x_list, f_list, g_list, \
                                          curr_f, gradient, constraints, bounds, \
                                          con_weight = con_weight)
    
    # print(x_list[-1], f_list[-1], g_list[-1])
    
    best_x, best_f, best_g = update_best_lists(x_list, f_list, g_list, \
                                               best_x, best_f, best_g)
    
    
    while (len(f_list) < max_f_eval) and (np.linalg.norm(gradient) > tolerance):
       x = np.array(x_list[-1]).reshape(len(x0),)
       
       if constraints != []:
           b = evaluate_bounds(x, constraints, con_weight = con_weight)
       else:
           b = []
    
       curr_f = add_penalty((f_list[-1], g_list[-1]), b_list = b, con_weight = con_weight)
       gradient, H, x_list, f_list, g_list = finite_differences(sim, x, x_list, \
                                                                f_list, g_list, \
                                                                constraints, \
                                                                con_weight = con_weight, \
                                                                current_f = curr_f)

       update = - (np.linalg.inv(H) @ gradient.reshape(-1, 1)).reshape(len(x0),)
        
       # print(update)
        
       x_list, f_list, g_list = backtracking(sim, x, update, x_list, f_list, \
                                             g_list, curr_f, gradient, \
                                             constraints, bounds, \
                                             con_weight = con_weight)
       
        # print(x_list[-1], f_list[-1], g_list[-1])
    
           
       # x_list += [x_list[-1]]
       # f_list += [f_list[-1]]
       
       best_x, best_f, best_g = update_best_lists(x_list, f_list, g_list, \
                                                  best_x, best_f, best_g)
       
    solution_output = {}
    solution_output['x_store'] = x_list
    solution_output['f_store'] = f_list
    solution_output['x_best_so_far'] = best_x
    solution_output['f_best_so_far'] = best_f
    solution_output['g_store'] = g_list
    solution_output['g_best_so_far'] = best_g
    solution_output['N_eval'] = len(f_list)
    solution_output['TR'] = None

    return solution_output
 
def Adam_optimizer(sim, x0, bounds = None, method = 'forward', \
                   alpha = 1e-3, beta1 = 0.9, beta2 = 0.99, epsilon = 1e-8, \
                   max_f_eval = 100, max_iter = 100, tolerance = 1e-8, \
                   con_weight = 100, print_status = True):
    
    constraints = []
    
    # if bounds is not None:
    #     bound_g = lambda x: np.sum(np.maximum(0, np.array(bounds)[:,0] - np.array(x))) + \
    #               np.sum(np.maximum(0, np.array(x) - np.array(bounds)[:,1]))
    #     constraints += [bound_g]
    
    # if constraints == []:
    #     constr = lambda x: 0
    #     constraints += [constr]
    
    x = list(x0)
    
    f_eval, g_eval = sim(x)
    
    x_list = [x]
    f_list = [f_eval]
    g_list = [g_eval]

    
    best_x = x_list.copy()
    best_f = f_list.copy()
    best_g = g_list.copy()
    
    if constraints != []:
        b = evaluate_bounds(x, constraints, con_weight = con_weight)
    else:
        b = []
    
    curr_f = add_penalty((f_list[-1], g_list[-1]), b_list = b, con_weight = con_weight)
    gradient, x_list, f_list, g_list = finite_differences(sim, x, x_list, f_list, \
                                                          g_list, constraints, \
                                                          current_f = curr_f, \
                                                          method = method, \
                                                          Hessian = False, \
                                                          con_weight = con_weight)   
    
    # print(gradient)    
    
    t = 0    
    x = np.array(x)
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    
    while (len(f_list) < max_f_eval) and (np.linalg.norm(gradient) > tolerance):
       
       t += 1
       
       m = beta1*m + (1 - beta1)*gradient
       v = beta2*v + (1- beta2)*gradient**2
       m_hat = m / (1 - beta1**t)
       v_hat = v / (1-beta2**t)
       
       update = - alpha*m_hat / (np.sqrt(v_hat) + epsilon)
       x_list, f_list, g_list = backtracking(sim, x, update, x_list, f_list, \
                                             g_list, curr_f, gradient, \
                                             constraints, bounds, \
                                             con_weight = con_weight)
           
       # if t == 1:
       #     print(x_list[-1], f_list[-1], g_list[-1])
           
       x = np.array(x_list[-1])
       
       # x_list += [list(x)]
       # f_list += [f(x)]
       # g_list += [[g_(x) for g_ in constraints]]
 
       best_x, best_f, best_g = update_best_lists(x_list, f_list, g_list, \
                                                  best_x, best_f, best_g)
       if constraints != []:
           b = evaluate_bounds(x, constraints, con_weight = con_weight)
       else:
           b = []
       curr_f = add_penalty((f_list[-1], g_list[-1]), b_list = b, con_weight = con_weight)
       
       gradient, x_list, f_list, g_list = finite_differences(sim, x, x_list, f_list, \
                                                             g_list, constraints, \
                                                             method = method, \
                                                             Hessian = False, \
                                                             con_weight = con_weight, \
                                                             current_f = curr_f)
    solution_output = {}
    solution_output['x_store'] = x_list
    solution_output['f_store'] = f_list
    solution_output['x_best_so_far'] = best_x
    solution_output['f_best_so_far'] = best_f
    solution_output['g_store'] = g_list
    solution_output['g_best_so_far'] = best_g
    solution_output['N_evals'] = len(f_list)
    solution_output['TR'] = None
    
    return solution_output   

def BFGS_optimizer(sim, x0, bounds = None, method = 'forward', \
                   max_f_eval = 100, max_iter = 100, tolerance = 1e-8, \
                   con_weight = 100, stepsize = 1, print_status = True):
   
    constraints = []
    
    # if bounds is not None:
    #     bound_g = lambda x: np.sum(np.maximum(0, np.array(bounds)[:,0] - np.array(x))) + \
    #               np.sum(np.maximum(0, np.array(x) - np.array(bounds)[:,1]))
    #     constraints += [bound_g]
    
    # if constraints == []:
    #     constr = lambda x: 0
    #     constraints += [constr]
    
    x = list(x0)
    
    f_eval, g_eval = sim(x)
    
    x_list = [x]
    f_list = [f_eval]
    g_list = [g_eval]

    
    best_x = x_list.copy()
    best_f = f_list.copy()
    best_g = g_list.copy()
    
    if constraints != []:
        b = evaluate_bounds(x, constraints, con_weight = con_weight)
    else:
        b = []
    
    curr_f = add_penalty((f_list[-1], g_list[-1]), b_list = b, con_weight = con_weight)
    
    gradient, H, x_list, f_list, g_list = finite_differences(sim, x, x_list, f_list, \
                                                          g_list, constraints, \
                                                          current_f = curr_f, \
                                                          con_weight = con_weight)
    # print(gradient)
    old_gradient = gradient
    old_H = np.linalg.inv(H)
    s = - (old_H @ gradient.reshape(-1, 1))
    
    # x = x + stepsize *s.reshape(len(x0),) 
    
    x_list, f_list, g_list = backtracking(sim, x, stepsize*s.reshape(len(x0),) , \
                                          x_list, f_list, g_list, \
                                          curr_f, gradient, constraints, bounds)
    # print(x_list[-1], f_list[-1], g_list[-1])
    # x_list += [x.tolist()]
    # f_list += [f(x)]
    # g_list += [[g_(x) for g_ in constraints]]
    
    best_x, best_f, best_g = update_best_lists(x_list, f_list, g_list, \
                                                  best_x, best_f, best_g)
        
    x = np.array(x_list[-1])
    
    while (len(f_list) < max_f_eval) and (np.linalg.norm(gradient) > tolerance):
      
       if constraints != []:
           b = evaluate_bounds(x, constraints, con_weight = con_weight)
       else:
           b = []  
      
       curr_f = add_penalty((f_list[-1], g_list[-1]), b_list = b, con_weight = con_weight)
           
       gradient, x_list, f_list, g_list = finite_differences(sim, x, x_list, f_list, \
                                                             g_list, constraints, \
                                                             current_f = curr_f, \
                                                             method = method, \
                                                             Hessian = False)
       
       y = (gradient - old_gradient).reshape(-1,1)
       s = s.reshape(len(x0),1)
       ro = 1 / (y.T @ s)
       I = np.eye(len(x0))
       
       next_H_inv = (I - ro * s @ y.T) @ old_H @ \
                    (I - ro * y @ s.T) + ro * s @ s.T
       s = - (next_H_inv @ gradient.reshape(-1, 1))
       old_gradient = gradient
       old_H = next_H_inv
       
       # print(s)
       
    
        # x = x + stepsize *s.reshape(len(x0),) 
    
       x_list, f_list, g_list = backtracking(sim, x, stepsize*s.reshape(len(x0),), \
                                             x_list, f_list, g_list, \
                                             curr_f, gradient, constraints, bounds)
       # print(x_list[-1], f_list[-1], g_list[-1])
        # x_list += [x.tolist()]
        # f_list += [f(x)]
        # g_list += [[g_(x) for g_ in constraints]]
    
       best_x, best_f, best_g = update_best_lists(x_list, f_list, g_list, \
                                                  best_x, best_f, best_g)
       
       x = np.array(x_list[-1])
    
        
    solution_output = {}
    solution_output['x_store'] = x_list
    solution_output['f_store'] = f_list
    solution_output['x_best_so_far'] = best_x
    solution_output['f_best_so_far'] = best_f
    solution_output['g_store'] = g_list
    solution_output['g_best_so_far'] = best_g
    solution_output['N_evals'] = len(f_list)
    solution_output['TR'] = None
    
    return solution_output   


# def Broyden_optimizer(f, x0, constraints = [],  bounds = None, method = 'central', \
#                    max_f_eval = 100, max_iter = 100, tolerance = 1e-8, \
#                    stepsize = 1, print_status = True):
    
#     x = list(x0)
    
#     x_list = [x]
#     f_list = [f(x)]
#     # g_eval_list = oracle.sample_g(center)
    
#     best_x = x_list.copy()
#     best_f = f_list.copy()
#     # best_g = g_eval_list.copy()
    
#     step = 1
    
#     gradient, x_list, f_list = finite_differences(f, x, x_list, f_list, \
#                                                      current_f = best_f[-1], \
#                                                      Hessian = False, \
#                                                      method = method)
    
#     update = - step*gradient
#     x_list, f_list = backtracking(f_, x, update, x_list, f_list, \
#                                       best_f[0] , gradient)
    
#     old_x = x
#     old_y = best_f[0]    
    
#     x = x_list[-1] 
    
#     y = f_list[-1]
    
#     best_x, best_f = update_best_lists(x_list, f_list, best_x, best_f)
    
#     while (len(f_list) < max_f_eval) and (np.linalg.norm(gradient) > tolerance):
      
#        dY = y - old_y ; dX = (np.array(x) - np.array(old_x)).reshape(-1,1)
       
#        gradient = gradient + (dY - float(gradient.reshape(-1,1).T @ dX)) / \
#                               (float(dX.T @ dX))*(np.array(x) - np.array(old_x))
       
#        old_x = x
#        old_y = y
       
#        update = - step*gradient
       
#        x_list, f_list = backtracking(f_, x, update, x_list, f_list, \
#                                       old_y, gradient)
       
      
#        x = x_list[-1] 
    
#        y = f_list[-1]
        
#        best_x, best_f = update_best_lists(x_list, f_list, best_x, best_f)
        
#     solution_output = {}
#     solution_output['x_store'] = x_list
#     solution_output['f_store'] = f_list
#     solution_output['x_best_so_far'] = best_x
#     solution_output['f_best_so_far'] = best_f
    
#     return solution_output 



# f = lambda x: np.cos(x[0] - 0.1*x[1])
# df = lambda x: np.array([- np.sin(x[0] - 0.1*x[1]), \
#                          + 0.1*np.sin(x[0] - 0.1*x[1])])
# H = lambda x: np.array([[- np.cos(x[0] - 0.1*x[1]), + 0.1*np.cos(x[0] - 0.1*x[1])], \
#                         [+ 0.1*np.cos(x[0] - 0.1*x[1]), - 0.01*np.cos(x[0] - 0.1*x[1])]])




# def trust_fig(oracle, bounds):
#     N = 200
#     lim = 2
#     x = np.linspace(-lim, lim, N)
#     y = np.linspace(-lim, lim, N)
#     X,Y = np.meshgrid(x, y)
#     Z = oracle.sample_obj(X,Y)
#     constr = oracle.sample_constr(X,Y)

#     level_list = np.logspace(-0.5, 4, 10)

#     fig = plt.figure(figsize = (6,4))
#     ax = fig.add_subplot()
    
#     ax.contour(X,Y,Z*constr, levels = level_list)
#     ax.plot([bounds[0,0], bounds[0, 1]], [bounds[1,0], bounds[1, 0]], c = 'k')
#     ax.plot([bounds[0,0], bounds[0, 1]], [bounds[1,1], bounds[1, 1]], c = 'k')
#     ax.plot([bounds[0,0], bounds[0, 0]], [bounds[1,0], bounds[1, 1]], c = 'k')
#     ax.plot([bounds[0,1], bounds[0, 1]], [bounds[1,0], bounds[1, 1]], c = 'k')
    
#     return ax


# class RB:
#     def __init__(self, objective, ineq = []):
#         self.obj = objective ; self.ieq = ineq
#     def sample_obj(self, x, y):
#         return self.obj(x, y)
#     def sample_constr(self, x, y):
#         if self.ieq == []:
#             if (type(x) == float) or (type(x) == int):
#                 return 1
#             else:
#                 return np.ones(len(x))
#         elif (type(x) == float) or (type(x) == int):
#             temporary = [int(g(x, y)) for g in self.ieq]
#             return np.product(np.array(temporary))
#         else:
#             temporary = [g(x, y).astype(int) for g in self.ieq]
#             return np.product(np.array(temporary), axis = 0)
    
# def sim_RB(x):
#     f = lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

#     g1 = lambda x: (x[0]-1)**3 - x[1] + 1
#     # g2 = lambda x, y: x + y - 2 <= 0
#     g2 = lambda x: x[0] + x[1] - 1.8
    
#     return f(x), [g1(x), g2(x)]

# # f_Rosenbruck = lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

# # g1 = lambda x: (x[0]-1)**3 - x[1] + 1
# # # g2 = lambda x, y: x + y - 2 <= 0
# # g2 = lambda x: x[0] + x[1] - 1.8

# # quadratic_f = lambda x: x[0]**2 + 10*x[1]**2 + x[0]*x[1]
# # quadratic_g = lambda x: 1 - x[0] - x[1]

# x0 = [-0.5, 1.5]

# bounds = np.array([[-1.5,1.5],[-1.5,1.5]])

# solution_output_approxHessian2 = finite_Diff_Newton(sim_RB, x0, bounds = bounds, con_weight = 100)

# solution_output_Adam = Adam_optimizer(sim_RB, x0, method = 'forward', \
#                                       bounds = bounds, alpha = 0.4, \
#                                       beta1 = 0.2, beta2  = 0.1, \
#                                       max_f_eval = 100, con_weight = 100)


# # # beta1 = 0.2, beta2 = 0.1

# solution_output_BFGS = BFGS_optimizer(sim_RB, x0, bounds = bounds, con_weight = 100)
    
# f_RB = lambda x, y: (1 - x)**2 + 100*(y - x**2)**2

# # oracle = RB(f)

# g1_RB = lambda x, y: (x-1)**3 - y + 1 <= 0
# # g2 = lambda x, y: x + y - 2 <= 0
# g2_RB = lambda x, y: x + y - 1.8 <= 0
# g_RB = [g1_RB, g2_RB]


# oracle = RB(f_RB, ineq = [g1_RB, g2_RB])




# ax = trust_fig(oracle, bounds)

# # x_all_Approx = solution_output_approxHessian['x_store']
# # x_best_Approx = solution_output_approxHessian['x_best_so_far']

# x_all_Approx = solution_output_approxHessian2['x_store']
# x_best_Approx = solution_output_approxHessian2['x_best_so_far']


# ax.scatter(np.array(x_all_Approx)[:,0], np.array(x_all_Approx)[:,1], \
#             s = 10, c = 'green')
    
# # ax.plot(np.array(x_all_Approx)[:,0], np.array(x_all_Approx)[:,1], '--k', \
# #             s = 10)
# ax.plot(np.array(x_best_Approx)[:,0], np.array(x_best_Approx)[:,1], '--r')
# ax.set_title('Approximate Newton method')

# ax = trust_fig(oracle, bounds)

# x_all_BFGS = solution_output_BFGS['x_store']
# x_best_BFGS = solution_output_BFGS['x_best_so_far']

# ax.scatter(np.array(x_all_BFGS)[:,0], np.array(x_all_BFGS)[:,1], \
#             s = 10, c = 'green')
# ax.plot(np.array(x_best_BFGS)[:,0], np.array(x_best_BFGS)[:,1], '--r')
# ax.set_title('BFGS')
# ax.set_xlim([-2, 2])
# ax.set_ylim([-2, 2])
# # Warm_start Hessian

# ax = trust_fig(oracle, bounds)

# x_all_Adam = solution_output_Adam['x_store']
# x_best_Adam = solution_output_Adam['x_best_so_far']

# ax.scatter(np.array(x_all_Adam)[:,0], np.array(x_all_Adam)[:,1], \
#             s = 10, c = 'green')
# ax.plot(np.array(x_best_Adam)[:,0], np.array(x_best_Adam)[:,1], '--r')
# ax.set_title('Adam')

# fig = plt.figure()
# ax = fig.add_subplot()


# y_Adam = solution_output_Adam['f_best_so_far']
# y_BFGS = solution_output_BFGS['f_best_so_far']
# y_Hess = solution_output_approxHessian2['f_best_so_far']
# # y = solution_output['f_best_so_far']

# ax.plot(np.arange(len(y_Adam)), y_Adam, 'b', label = 'Adam, #f_eval = ' + \
#         str(len(solution_output_Adam['f_store'])))
# ax.plot(np.arange(len(y_BFGS)), y_BFGS, 'g', label = 'BFGS, #f_eval = ' + \
#         str(len(solution_output_BFGS['f_store'])))
# ax.plot(np.arange(len(y_Hess)), y_Hess, 'orange', label = 'Approx. Newton, #f_eval = ' + \
#         str(len(solution_output_approxHessian2['f_store'])))
# ax.legend()
# # ax.plot(np.arange(len(y)), y, 'k')

# ax.set_yscale('log')