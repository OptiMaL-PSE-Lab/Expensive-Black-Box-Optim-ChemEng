# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 03:05:08 2021

@author: dv516
"""

import nlopt
import numpy as np
import functools
import warnings

# from test_functions import rosenbrock_constrained

# def Problem_rosenbrock_test(x, grad):
#     f1 = rosenbrock_constrained.rosenbrock_f
#     g1 = rosenbrock_constrained.rosenbrock_g1
#     g2 = rosenbrock_constrained.rosenbrock_g2

#     return f1(x), [g1(x), g2(x)]

def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + '\n'

class PenaltyFunctions_NLopt:
    """
    BE CAREFULL TO INITIALIZE THIS FUNCTION BEFORE USE!!
    IT TAKES HISTORICAL DATA WITH IT
    """
    def __init__(self, f_total, type_penalty='l2', mu=100):
        self.f = f_total
        #self.g     = g

        self.f_his = []
        self.g_his = []
        self.x_his = []
        
        self.type_p = type_penalty
        self.aug_obj = self.augmented_objective(mu)

    def create_quadratic_penalized_objective(self, mu, order, x, grad):
        
        funcs = self.f(x, grad)
        obj   = funcs[0]
        card_of_funcs = len(funcs[1])+1
        if type(obj) == float:
            self.f_his += [obj]
        else:
            self.f_his += [obj.copy()]
        n_con = card_of_funcs-1
        g_tot = np.zeros(n_con)
        for i in range(n_con):
            g_tot[i] = funcs[1][i]
            obj += mu * max(g_tot[i], 0) ** order
        self.g_his += [g_tot]
        self.x_his += [x.tolist().copy()]
        return obj
    
    def augmented_objective(self, mu):
        """

        :param mu: The penalized parameter
        :type mu: float
        :return:  obj_aug
        :rtype:   function
        """
        if self.type_p == 'l2':
            warnings.formatwarning = custom_formatwarning

            warnings.warn(
                'L2 penalty is used with parameter ' + str(mu))

            obj_aug = functools.partial(self.create_quadratic_penalized_objective,
                                        mu, 2)
        elif self.type_p == 'l1':
            warnings.formatwarning = custom_formatwarning

            warnings.warn(
                'L1 penalty is used with parameter ' + str(mu))

            obj_aug = functools.partial(self.create_quadratic_penalized_objective,
                                        mu, 1)
        else:
            mu_new = 100
            warnings.formatwarning = custom_formatwarning

            warnings.warn(
                'WARNING: Penalty type is not supported. L2 penalty is used instead with parameter ' + str(mu_new))

            obj_aug = functools.partial(self.create_quadratic_penalized_objective,
                                        mu_new, 2)
        return obj_aug
    
    
class DIRECTWrapper:
    __metaclass__ = nlopt.opt
    def __init__(self):
        self.opt = nlopt.opt

    def solve(self, objfun, x0, bounds, maxfun=100, 
              constraints =0, penalty_con='l2', mu_con=1e3):
        self.opt = self.opt(nlopt.GN_DIRECT_L_RAND, len(x0))
        self.maxfun = maxfun
        if (constraints)==0:#constraints==None:
            self.card_of_funcs = 1
            self.opt.set_min_objective(objfun)
            self.opt.set_lower_bounds(bounds[:,0])
            self.opt.set_upper_bounds(bounds[:,1])
            self.opt.set_maxeval(maxfun)
            self.x_opt = self.opt.optimize(x0.tolist())
            self.f_opt = self.opt.last_optimum_value()
            
        else:
            #self.constraints = constraints
            self.card_of_funcs = 1 +constraints
            self.Penaly_fun = PenaltyFunctions_NLopt(objfun, type_penalty=penalty_con, \
                                               mu = mu_con)
            f_pen = self.Penaly_fun.aug_obj  # functools.partial(penalized_objective,f1,[g1,g2], 100)

            self.opt.set_min_objective(f_pen)
            self.opt.set_lower_bounds(bounds[:,0])
            self.opt.set_upper_bounds(bounds[:,1])
            self.opt.set_maxeval(maxfun)
            self.x_opt = self.opt.optimize(list(x0))
            self.f_opt = self.opt.last_optimum_value()
        # print(self.Penaly_fun.x_his)
        return self.create_dictionary_for_solution()


    def create_dictionary_for_solution(self):
        # CHANGE FOR THE SO FAR GOOD

        f_so_far, g_so_far, x_so_far = self.find_min_so_far()
        output_dict = {}
        output_dict['g_store']       = self.Penaly_fun.g_his
        # print(sol)
        # print(sol.diagnostic_info['xk'])
        # print(np.array([sol.diagnostic_info['xk'].tolist()]))
        # print(np.array([sol.diagnostic_info['xk']])[0])
        # print(np.array([sol.diagnostic_info['xk']])[0].astype('d'))
        output_dict['x_store']       = self.Penaly_fun.x_his
        output_dict['f_store']       = self.Penaly_fun.f_his
        output_dict['N_evals']       = len(f_so_far)
        output_dict['g_best_so_far'] = g_so_far
        output_dict['f_best_so_far'] = f_so_far
        output_dict['x_best_so_far'] = x_so_far
        output_dict['samples_at_iteration']            = np.arange(1, 1+len(f_so_far))
        return output_dict
   
    
    def find_min_so_far(self):
        """
        This function find the best solution so far, mainly used for EI
        :param argmin: Boolean that if it is True the func returns which point is the best
        :type argmin:  Boolean
        """
        length = len(self.Penaly_fun.f_his)
        x_so_far = np.inf+np.zeros([length, len(self.Penaly_fun.x_his[0])])
        f_so_far = np.inf+np.zeros(length)
        g_so_far = np.inf+np.zeros([length, self.card_of_funcs-1])
        
        # print(self.Penaly_fun.f_his)
        # print(self.Penaly_fun.x_his)
        # print(self.Penaly_fun.g_his)
        
        for iter in range(length):
            if np.sum(np.maximum(self.Penaly_fun.g_his[0], 0)) == 0:
                min_f = self.Penaly_fun.f_his[0]
            else: 
                min_f = self.Penaly_fun.f_his[0]+100
            index = 0
            min_g =self.Penaly_fun.g_his[0]#self.Penaly_fun.g_his[iter]
            if self.card_of_funcs==1:

                for i in range(iter):
                        y = self.Penaly_fun.f_his[i]
                        if y< min_f:
                            min_f   = y
                            index = i
                        f_so_far[iter] = min_f
                        x_so_far[iter] = self.Penaly_fun.x_his[index].copy()
                        
            else:
                for i in range(iter+1):
                    y = self.Penaly_fun.f_his[i]
                    if y < min_f and all(self.Penaly_fun.g_his[i]<=0):
                        min_f   = y
                        index = i
                        # print(index)
                        min_g = self.Penaly_fun.g_his[i]
                    f_so_far[iter] = min_f
                    g_so_far[iter,:] = min_g
                    x_so_far[iter] = self.Penaly_fun.x_his[index].copy()

        if self.card_of_funcs>1:
            return f_so_far, g_so_far, x_so_far
        else:
            return f_so_far, x_so_far  
    
# bounds = np.array([[-1,0.5],[-1,1.5]])
# x0 = np.array([-0.5,1.5])
    
# sol_DIRECT = DIRECTWrapper().solve(Problem_rosenbrock_test, x0, bounds, \
#                                       maxfun= 100, constraints=2)
    
  


  