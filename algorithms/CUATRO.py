# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 21:12:38 2021

@author: dv516
"""

import cvxpy as cp
import numpy as np
import scipy.linalg as LA
import matplotlib.pyplot as plt

def quadratic_LA(X, Y, P, q, r):
    N = len(X)
    Z = np.zeros(X.shape)
    for i in range(N):
        for j in range(N):
            X_ = np.array([X[i,j], Y[i,j]]).reshape(-1,1)
            Z[i,j] = float(X_.T @ P @ X_ + q.T @ X_ + r)
    return Z

def make_PSD(P):
    eig_val, eig_vec = LA.eigh(P)
    # print(eig_val)
    eig_val = np.array([max(val, 1e-8) for val in eig_val])
    # print(eig_val)
    P = np.dot(eig_vec, eig_val[:, np.newaxis]*eig_vec.T)
    return P

def LHS(bounds, N, rnd_seed = 1):
    np.random.seed(rnd_seed)
    matrix = np.zeros((len(bounds), N))
    for i in range(len(bounds)):
        l, u = bounds[i]
        rnd_ind = np.arange(N)
        np.random.shuffle(rnd_ind)
        # print(rnd_ind)
        rnd_array = l + (np.random.rand(N)+ rnd_ind)*(u-l)/N
        matrix[i] = rnd_array
    return matrix

def sample_LHS(sim, bounds, N, rnd_seed = 1):
    data_points = LHS(bounds, N, rnd_seed = rnd_seed).T
    func_eval, g_eval, feas = sample_simulation(data_points.tolist(), sim)
    
    return data_points, func_eval, g_eval, feas

def sample_points(center, radius, f, bounds, N = 10):
    
    if bounds is None:
        data_points = np.array(center*N).reshape(N, len(center)) + \
                      np.random.uniform(-radius, radius, (N, len(center)))
    else:
        uniform_sampling = np.zeros((N, len(center)))
        for i in range(len(center)):
            lower_bound = - radius ; upper_bound = radius
            if center[i] - radius < bounds[i,0]:
                lower_bound = bounds[i,0] - center[i]
            if center[i] + radius > bounds[i,1]:
                upper_bound = bounds[i,1] - center[i]
            uniform_sampling[:,i] = np.random.uniform(lower_bound, upper_bound, N)
            
        data_points = np.array(center*N).reshape(N, len(center)) + \
                        uniform_sampling
    func_eval, g_eval, feas = sample_simulation(data_points.tolist(), f)
    
    return data_points, func_eval, g_eval, feas


def update_best_lists(X_list, f_list, g_list, X_best, f_best, g_best):
    g_feas = constr_creation(X_list, g_list)
    f = np.array(f_list)
    ind = np.where(f == np.min(f[g_feas == 1]))
    X_best += np.array(X_list)[ind].tolist()
    f_best += f[ind].tolist()
    g_best += np.array(g_list)[ind].tolist()
    
    return X_best, f_best, g_best

def samples_in_trust(center, radius, \
                     X_samples_list, y_samples_list, g_list):
    X = np.array(X_samples_list) 
    y = np.array(y_samples_list) 
    g = np.array(g_list)
    ind = np.where(np.linalg.norm(X - np.array(center), axis = 1,\
                                  keepdims = True) < radius)[0]
    X_in_trust = X[ind] ; y_in_trust = y[ind] ; g_in_trust = g[ind]
    feas_in_trust = constr_creation(X_in_trust, g_in_trust.tolist())
    
    return X_in_trust, y_in_trust, g_in_trust, feas_in_trust

def quadratic_fitting(X_mat, y_mat, discr = False):
    N, M = X_mat.shape[0], X_mat.shape[1]
    P = cp.Variable((M, M), PSD = True)
    q = cp.Variable((M, 1))
    r = cp.Variable()
    X = cp.Parameter(X_mat.shape)
    y = cp.Parameter(y_mat.shape)
    X.value = X_mat
    y.value = y_mat
    quadratic = cp.bmat([cp.quad_form(X.value[i].reshape(-1,1), P) + \
                        q.T @ X.value[i].reshape(-1,1) + r - y.value[i] for i in range(N)])
    # quadratic = cp.quad_form(X, P) + q.T @ X 
    # quadratic = cp.quad_form(X, P) + q.T @ X + r
    obj = cp.Minimize(cp.norm(quadratic))
    if not discr:
        prob = cp.Problem(obj)
    else:
        const_P = [P >> np.eye(M)*1e-9]
        prob = cp.Problem(obj, constraints = const_P)
    if not prob.is_dcp():
        print("Problem is not disciplined convex. No global certificate")
    prob.solve()
    if prob.status not in ['unbounded', 'infeasible']:
        return P.value, q.value, r.value
    else:
        print(prob.status, ' CVX objective fitting call at: ')
        print('X matrix', X_mat)
        print('y array', y_mat)
        raise ValueError

def quadratic_discrimination(x_inside, y_outside):
    N, M, D = x_inside.shape[0], y_outside.shape[0], x_inside.shape[1]
    u = cp.Variable(N, pos = True)
    v = cp.Variable(M, pos = True)
    P = cp.Variable((D,D), PSD = True)
    q = cp.Variable((D, 1))
    r = cp.Variable()
    X = cp.Parameter(x_inside.shape, value = x_inside)
    Y = cp.Parameter(y_outside.shape)
    X.value = x_inside ; Y.value = y_outside
    const_u = [cp.quad_form(X.value[i].reshape(-1,1), P) + \
                        q.T @ X.value[i].reshape(-1,1) + r <= -(1 - u[i]) for i in range(N)]
    const_v = [cp.quad_form(Y.value[i].reshape(-1,1), P) + \
                        q.T @ Y.value[i].reshape(-1,1) + r >= (1 - v[i]) for i in range(M)]
    const_P = [P >> np.eye(D)*1e-9]
    # const_P = [P >> np.eye(D)*1]
    # const_P = [P >> 0]
    prob = cp.Problem(cp.Minimize(cp.sum(u) + cp.sum(v)), \
                      constraints = const_u + const_v + const_P)
    if not prob.is_dcp():
        print("Problem is not disciplined convex. No global certificate")
    prob.solve()
    if prob.status not in ['unbounded', 'infeasible']:
        return P.value, q.value, r.value
    else:
        print(prob.status, ' CVX ineq. classification call at: ')
        print('x_inside', x_inside)
        print('x_outside', y_outside)
        raise ValueError


def quadratic_min(P_, q_, r_, center, radius, bounds, ineq = None):
    X = cp.Variable((len(center), 1))
    # P = cp.Parameter(P_.shape, value = P_, PSD = True)
    try:
        P = cp.Parameter(P_.shape, value = P_, PSD = True)
    except:
        P_ = make_PSD(P_)
        if (P_ == 0).all():
            P_ = np.eye(len(P_))*1e-8
            P = cp.Parameter(P_.shape, value = P_, PSD = True)
        try:
            P = cp.Parameter(P_.shape, value = P_, PSD = True)
        except:
            P_ = np.eye(len(P))*1e-8
            P = cp.Parameter(P_.shape, value = P_, PSD = True)
    q = cp.Parameter(q_.shape, value = q_)
    r = cp.Parameter(r_.shape, value = r_)
    objective = cp.Minimize(cp.quad_form(X, P) + q.T @ X + r)
    trust_center = np.array(center).reshape((P_.shape[0], 1))
    constraints = []
    if ineq != None:
        for coeff in ineq:
            P_ineq, q_ineq, r_ineq = coeff
            if not ((P_ineq is None) or (q_ineq is None) or (r_ineq is None)):
                P_iq = cp.Parameter(P_ineq.shape, value = P_ineq, PSD = True)
                q_iq = cp.Parameter(q_ineq.shape, value = q_ineq)
                r_iq = cp.Parameter(r_ineq.shape, value = r_ineq)
                constraints += [cp.norm(X - trust_center) <= radius,
                           cp.quad_form(X, P_iq) + q_iq.T @ X + r_iq <= 0]

    else:
        constraints = [cp.norm(X - trust_center) <= radius]
    if bounds is not None:
        constraints += [bounds[i,0] <=  X[i] for i in range(P_.shape[0])]
        constraints += [X[i] <= bounds[i,1] for i in range(P_.shape[0])]
    prob = cp.Problem(objective, constraints)
    if not prob.is_dcp():
        print("Problem is not disciplined convex. No global certificate")
    prob.solve()
    if prob.status not in ['unbounded', 'infeasible']:
        return X.value.reshape(P_.shape[0])
    else:
        print(prob.status, ' CVX min. call at: ')
        print('Center', center)
        print('Radius', radius)
        print('P_', P_)
        print('q_', q_)
        print('r_', r_)
        print('Ineq', ineq)
        raise ValueError
        

def minimise(X_samples, feas_X, infeas_X, g_array, P, q, r, bounds, center, radius, method):
    if (P is None) or (q is None) or (r is None):
        print("P is of type None. Jump step..")
    elif len(feas_X) != len(X_samples) :
        
        all_feas = True
        if method == 'Discrimination':
            try:
                P_ineq, q_ineq, r_ineq = quadratic_discrimination(feas_X, infeas_X)
            except:
                P_ineq, q_ineq, r_ineq = None, None, None
                all_feas = False
            ineq_list = [(P_ineq, q_ineq, r_ineq)]
            # print('Discrimination constants: ', P_ineq, q_ineq, r_ineq)
        
        else:
            ineq_list = []
            n_ineq = g_array.shape[1]
            # print(g_array)
            for i in range(n_ineq):
                g_pred = g_array[:,i]
                try:
                    fitting_out = quadratic_fitting(X_samples, g_pred, discr = True)
                # print(i, fitting_out)
                # print(g_pred)
                    ineq_list += [fitting_out]
                    # print('Yes')
                except:
                    ineq_list += [(None, None, None)]
                    # print('No')
                #     print('Fitting constants: ', P_ineq, q_ineq, r_ineq)
                #     print('Yes')
                # except:
                #     all_feas = False
                #     print('Nope')
            
        
        # g_predicted = np.max(g_list, axis = 1)
        # try:
        #     P_ineq, q_ineq, r_ineq = quadratic_fitting(X_samples, g_predicted)
        # except:
        #     P_ineq, q_ineq, r_ineq = None, None, None
        #     print('Inequality constraint fitting failed')
        # print('Fitting constants: ', P_ineq, q_ineq, r_ineq)
        
        if all_feas:
            try:
                center_ = list(quadratic_min(P, q, r, center, radius, bounds, \
                                             ineq = ineq_list))
            except:
                P = make_PSD(P)
                # print(P)
                try:
                    center_ = list(quadratic_min(P, q, r, center, radius, bounds, \
                               ineq = ineq_list))
                except:
                    center_ = center
        else:
            try:
                center_ = list(quadratic_min(P, q, r, center, radius, bounds))
            except:
                P = make_PSD(P)
                # print(P)
                try:
                    center_ = list(quadratic_min(P, q, r, center, radius, bounds))
                except:
                    center_ = center
    else:
        try:
            center_ = list(quadratic_min(P, q, r, center, radius, bounds))
        except:
            P = make_PSD(P)
            # print(P)
            try:
                center_ = list(quadratic_min(P, q, r, center, radius, bounds))
            except:
                center_ = center
    return center_

def constr_creation(x, g):
    if g is None:
        if any(isinstance(item, float) for item in x) or any(isinstance(item, int) for item in x):
            feas = 1
        else:
            feas = np.ones(len(np.array(x)))
    elif any(isinstance(item, float) for item in x) or any(isinstance(item, int) for item in x):
        feas = np.product((np.array(g) <= 0).astype(int))
    else:
        feas = np.product( (np.array(g) <= 0).astype(int), axis = 1)
    return feas

def sample_oracle(x, f, ineq = []):
    if any(isinstance(item, float) for item in x) or any(isinstance(item, int) for item in x):
        y = [f(x)]
        if ineq == []:
            g_list = None
        else:
            g_list = [[g_(x) for g_ in ineq]]
    else:
        y = []
        g_list = []
        for x_ in x:
            y += [f(x_)]
            if ineq != []:
                g_list += [[g_(x_) for g_ in ineq]]
    if g_list == []:
        g_list = None
    
    feas = constr_creation(x, g_list)
    
    return y, g_list, feas


def sample_simulation(x, sim):
    f_list = [] ; g_list = []
    if any(isinstance(item, float) for item in x) or any(isinstance(item, int) for item in x):
        obj, constr_vec = sim(x)
        f_list += [obj]
        if constr_vec is not None:
            g_list = [constr_vec]
    else:
        for x_ in x:
            obj, constr_vec = sim(x_)
            f_list += [obj]
            if constr_vec is not None:
                g_list += [constr_vec]
    if constr_vec is None:
        g_list = None
    
    feas = constr_creation(x, g_list)
    
    return f_list, g_list, feas



def CUATRO(sim, x0, init_radius, constraints = [], bounds = None, \
           max_f_eval = 100, max_iter = 100, tolerance = 1e-8, beta_inc = 1.2, \
           beta_red = 0.8, eta1 = 0.2, eta2 = 0.8, method = 'local', \
           N_min_samples = 6, rnd = 1, print_status = False, constr_handling = 'Discrimination'):
    '''
    INPUTS
    ------------------------------------
    f:          function to be optimised
    
    x0:         initial guess in form [x1,x2,...,xn]
    
    init_radius: initial trust region radius
                
    max_iter:   total optimisation iterations due to 
                no stopping conditions
    
    constraints: constraint functions in form [g1,g2,...,gm]
                all have form g(x) <= 0 and return g(x)
                
    OUTPUTS
    ------------------------------------
    output_dict: 
        - 'x'           : final input variable
        
        - 'f'           : final function value
        
        - 'f_evals'     : total number of function evaluations
        
        - 'f_store'     : best function value at each iteration
                            
        - 'x_store'     : list of all previous best variables (per iteration)
                            
        - 'g_store'     : list of all previous constraint values (per iteration)
        
        - 'g_viol'      : total constraint violation (sum over constraints)
    
    NOTES
    --------------------------------------
     - 
    '''
    
    # class oracle_sample:
    #     def __init__(self, objective, ineq = []):
    #         self.obj = objective ; self.ieq = ineq
    #     def sample_obj(self, x):
    #         if any(isinstance(item, float) for item in x) or any(isinstance(item, int) for item in x):
    #             return [self.obj(x)]
    #         else:
    #             obj_list = []
    #             for x_ in x:
    #                 obj_list += [self.obj(x_)]
    #             return obj_list
    #     def sample_g(self, x):
    #         if self.ieq == []:
    #             return [None]
    #         elif any(isinstance(item, float) for item in x) or any(isinstance(item, int) for item in x):
    #             return [[g(x) for g in self.ieq]]
    #         else:
    #             temporary = []
    #             for x_row in x:
    #                 temporary += [[g(x_row) for g in self.ieq]]
    #             return temporary
    #     def sample_constr(self, x, g_list):
    #         if g_list is None:
    #             if any(isinstance(item, float) for item in x) or any(isinstance(item, int) for item in x):
    #                 return 1
    #             else:
    #                 return np.ones(len(np.array(x)))
    #         elif any(isinstance(item, float) for item in x) or any(isinstance(item, int) for item in x):
    #             return np.product((np.array(g_list) <= 0).astype(int))
    #         else:
    #             return np.product( (np.array(g_list) <= 0).astype(int), axis = 1)
            
    
    # if constraints == []:
    #     g = lambda x: 0
    #     constraints = [g]
    
    # oracle = oracle_sample(f, ineq = constraints)
    
    center = list(x0) ; radius = init_radius
    
    f_eval_list, g_eval_list, feas = sample_simulation(center, sim)
    
    if feas == 0:
        raise ValueError("Please enter feasible starting point")
    
    X_samples_list = [center]
    # f_eval_list = oracle.sample_obj(center)
    # g_eval_list = oracle.sample_g(center)
    
    best_x = X_samples_list.copy()
    best_f = f_eval_list.copy()
    best_g = g_eval_list.copy()
        
    radius_list = [init_radius]
    
    if method == 'local':
        X_samples, y_samples, g_eval, feas =  sample_points(center, radius, sim, \
                                                            bounds, N = N_min_samples)
    elif method == 'global':
        X_samples, y_samples, g_eval, feas = sample_LHS(sim, bounds, \
                                                        N_min_samples, rnd_seed = rnd)
    else:
        raise ValueError('Invalid input for method')
    
    # feas_samples = oracle.sample_constr(X_samples, g_list = g_eval)
    X_samples_list += X_samples.tolist()
    f_eval_list += y_samples
    g_eval_list += g_eval
    
    old_trust = center
    old_f = best_f[0]

    P, q, r = quadratic_fitting(X_samples, np.array(y_samples))
    feas_X = X_samples.copy()[feas == 1]
    infeas_X = X_samples.copy()[feas != 1]

    if not ((P is None) or (q is None) or (r is None)):
        center_ = minimise(X_samples, feas_X, infeas_X, np.array(g_eval), P, q, \
                           r, bounds, center, radius, constr_handling)
    else:
        print('P is None in first iteration')
        center_ = list(x0)
    
    center = [float(c) for c in center_]
    
    f_eval, g_eval, new_feas = sample_simulation(center, sim)
    
    new_f = f_eval[0]
    # g_eval = oracle.sample_g(center)
    # print(center)
    # print(g_eval)
    # new_feas = oracle.sample_constr(center, g_list = g_eval) 
    # print(new_feas)
    X_samples_list += [center]
    f_eval_list += [new_f]
    g_eval_list += g_eval
    
    best_x, best_f, best_g = update_best_lists(X_samples_list, \
                                               f_eval_list, g_eval_list,  \
                                               best_x, best_f, best_g)
    
    X = np.array(center).reshape(-1,1)
    new_pred_f = X.T @ P @ X + q.T @ X + r
    X_old = np.array(old_trust).reshape(-1,1)
    old_pred_f = X_old.T @ P @ X_old + q.T @ X_old + r
    
    pred_dec = old_pred_f - new_pred_f ; dec = old_f - new_f
    
    N = 1
    
    while (len(f_eval_list) < max_f_eval - 1) and (N <= max_iter) and (radius > tolerance):
        if method == 'local':
            if (new_feas == 0) or (new_f - old_f > 0):
                radius *= beta_red
                center = old_trust
            else:
                if (dec >= eta2*pred_dec) and (abs(np.linalg.norm(np.array(old_trust) - np.array(center)) - radius) < 1e-8).any():
                    radius *= beta_inc
                    old_trust = center
                    old_f = new_f

                elif dec <= eta1*pred_dec:
                    radius *= beta_red
                    center = old_trust
                else:
                    old_trust = center
                    old_f = new_f
        else:
            radius *= beta_red
            if (new_feas == 0) or (new_f - old_f > 0):
                center = old_trust
            else:
                old_trust = center
                old_f = new_f
            
        radius_list += [radius]

        if P is not None:
            X = np.array(old_trust).reshape(-1,1)
            old_pred_f = X.T @ P @ X + q.T @ X + r
        
        X_in_trust, y_in_trust, g_in_trust, feas_in_trust = samples_in_trust(center, radius, \
                                                                X_samples_list, f_eval_list, g_eval_list)
        N_samples, N_x = X_in_trust.shape
        if N_samples >= N_min_samples:
            N_s = 1
        else:
            N_s = N_min_samples - N_samples
        if (len(f_eval_list) + N_s) > max_f_eval - 1:
            N_s = max(max_f_eval - 1 - len(f_eval_list), 1)
        
        X_samples, y_samples, g_eval, feas_samples =  sample_points(center, radius, sim, \
                                                                    bounds, N = N_s)
        
        X_samples_list += X_samples.tolist()
        f_eval_list += y_samples
        g_eval_list += g_eval
        
        X_samples = np.array(X_in_trust.tolist() + X_samples.tolist())
        y_samples = np.array(y_in_trust.tolist() + y_samples)
        g_samples = np.array(g_in_trust.tolist() + g_eval)
        feas_samples = np.array(feas_in_trust.tolist() + feas_samples.tolist())
        
        try:
            P, q, r = quadratic_fitting(X_samples, y_samples)
        except:
            print('Mosek failed to find convex quadratic fit')
            
        feas_X = X_samples.copy()[feas_samples == 1]
        infeas_X = X_samples.copy()[feas_samples != 1]
    
        # while len(infeas_X) == len(X_samples):
        #     print("No feasible points sampled. Reducing radius and resampling..")
        #     print('Current center: ', center)
        #     print('Iteration: ', N)
        #     X_samples, y_samples, g_eval, feas_samples =  sample_points(center, radius, \
        #                                                   bounds, N = 9)
        #     X_samples_list += X_samples.tolist()
        #     f_eval_list += y_samples
        #     g_eval_list += g_eval

            
        
        if not ((P is None) or (q is None) or (r is None)):
            
            center_ = minimise(X_samples, feas_X, infeas_X, g_samples, P, q, r, bounds, \
                           center, radius, constr_handling)
            
            center = [float(c) for c in center_]
        
            f_eval, g_eval, new_feas = sample_simulation(center, sim)
            new_f = f_eval[0]
        
            X_samples_list += [center]
            f_eval_list += [new_f]
            g_eval_list += g_eval
            X = np.array(center).reshape(-1,1)
            new_pred_f = X.T @ P @ X + q.T @ X + r
    
            pred_dec = old_pred_f - new_pred_f ; dec = old_f - new_f
        
        best_x, best_f, best_g = update_best_lists(X_samples_list, \
                                               f_eval_list, g_eval_list,  \
                                               best_x, best_f, best_g)
            
        N += 1
    
    N_evals = len(f_eval_list)
    radius_list += [radius] 
   
    if N > max_iter:
        status = "Max # of iterations reached"
    elif radius < tolerance:
        status = "Radius below threshold"
    else:
        status = "Max # of function evaluations"
    
    if print_status:
        print('Minimisation terminated: ', status)
    
    output = {'x_best_so_far': best_x, 'f_best_so_far': best_f, \
              'g_best_so_far': best_g, 'x_store': X_samples_list, \
              'f_store': f_eval_list, 'g_store': g_eval_list, \
              'N_eval': N_evals, 'N_iter': N, 'TR': radius_list}
    
    return output



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



# f = lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

# g1 = lambda x: (x[0]-1)**3 - x[1] + 1
# # g2 = lambda x, y: x + y - 2 <= 0
# g2 = lambda x: x[0] + x[1] - 1.8

# quadratic_f = lambda x: x[0]**2 + 10*x[1]**2 + x[0]*x[1]
# quadratic_g = lambda x: 1 - x[0] - x[1]

# def sim_RB(x):
#     f = lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

#     g1 = lambda x: (x[0]-1)**3 - x[1] + 1
# # g2 = lambda x, y: x + y - 2 <= 0
#     g2 = lambda x: x[0] + x[1] - 1.8
    
#     return f(x), [g1(x), g2(x)]

# def sim_RB_test(x):
#     f = lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

#     # g1 = lambda x: (x[0]-1)**3 - x[1] + 1
# # g2 = lambda x, y: x + y - 2 <= 0
#     # g2 = lambda x: x[0] + x[1] - 1.8
    
#     return f(x), []

# x0 = [-0.5, 1.5]


# bounds = np.array([[-1.5,1.5],[-1.5,1.5]])

# # bounds = np.array([[-1000,1000],[-1000,1000]])

# # bounds = np.array([[-0.6,-0.4],[1,2]])

# bounds_quadratic = np.array([[-1.5,1.5],[-0.5,0.5]])

# # solution_output = cvx_quad_surr_opt(f, x0, init_radius, bounds = bounds, \
# #                                     beta_red = 0.5, constraints = [g1, g2])

# method = 'Fitting'
# # method = 'Discrimination'
# rnd_seed = 10

# N_min_s = 6
# init_radius = .1
# solution_output = CUATRO(sim_RB, x0, init_radius, bounds = bounds, \
#                           N_min_samples = N_min_s, tolerance = 1e-10, \
#                           beta_red = 0.5, rnd = 1, method = 'local', \
#                           constr_handling = 'Fitting')


# N_min_s = 20
# init_radius = 2
# solution_output = CUATRO(sim_RB, x0, init_radius, bounds = bounds, \
#                           N_min_samples = N_min_s, tolerance = 1e-10,\
#                           beta_red = 0.9, rnd = rnd_seed, method = 'global', \
#                           constr_handling = method)



# x_all = solution_output['x_store']
# x_best = solution_output['x_best_so_far']

# f_RB = lambda x, y: (1 - x)**2 + 100*(y - x**2)**2

# # oracle = RB(f)

# g1_RB = lambda x, y: (x-1)**3 - y + 1 <= 0
# # g2 = lambda x, y: x + y - 2 <= 0
# g2_RB = lambda x, y: x + y - 1.8 <= 0
# g_RB = [g1_RB, g2_RB]


# oracle = RB(f_RB, ineq = [g1_RB, g2_RB])

# ax = trust_fig(oracle, bounds)

# ax.scatter(np.array(x_all)[:N_min_s+1,0], np.array(x_all)[:N_min_s+1,1], s = 10, c = 'blue')
# ax.scatter(np.array(x_all)[N_min_s+1:,0], np.array(x_all)[N_min_s+1:,1], s = 10, c = 'green')
# ax.plot(np.array(x_best)[:,0], np.array(x_best)[:,1], '--r')
# ax.set_title('CQSO: ' + str(method) + ' , rnd seed: ' + str(rnd_seed))


# solution_list = []

# N_fail = 0
# N = 100
# for i in range(N):
#     print('Iteration ', i+1)
#     sol = cvx_quad_surr_opt(f, x0, init_radius, bounds = bounds, \
#                                     beta_red = 0.5, constraints = [g1, g2])
#     solution_list += [sol]
    
# fig = plt.figure(figsize = (6,8))
# ax = fig.add_subplot(211)

# nbr_eval = np.zeros(N)

# for i in range(N):
#     y = np.array(solution_list[i]['f_best_so_far'])
#     nbr_eval[i] = len(solution_list[i]['f_store'])
#     ax.plot(np.arange(len(y)),y, '--k')
    
# ax.set_yscale('log')

# ax.set_xlim([0, 40])

# ax2 = fig.add_subplot(212)

# ax2.hist(nbr_eval)

