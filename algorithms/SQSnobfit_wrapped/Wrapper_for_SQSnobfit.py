# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 01:15:58 2021

@author: dv516
"""

import SQSnobFit
import numpy as np
from utilities.general_utility_functions import PenaltyFunctions

# from test_functions import rosenbrock_constrained

# def Problem_rosenbrock(x):
#     f1 = rosenbrock_constrained.rosenbrock_f
#     g1 = rosenbrock_constrained.rosenbrock_g1
#     g2 = rosenbrock_constrained.rosenbrock_g2

#     return f1(x), [g1(x), g2(x)]

class SQSnobFitWrapper:
    __metaclass__ = SQSnobFit.minimize
    def __init__(self):
        self.solve_naked = SQSnobFit.minimize

    def solve(self, objfun, x0, bounds, maxfun=100, 
              constraints =0, penalty_con='l2', mu_con=1e3):
        
        self.maxfun = maxfun
        if (constraints)==0:#constraints==None:

            self.card_of_funcs = 1
            self.Penaly_fun = PenaltyFunctions(objfun, type_penalty=penalty_con, \
                                               mu = mu_con)
            f_pen = self.Penaly_fun.aug_obj
            _, sol1 = self.solve_naked(f_pen, x0, bounds, maxfun-1)
            
        else:
            #self.constraints = constraints
            self.card_of_funcs = 1 +constraints
            self.Penaly_fun = PenaltyFunctions(objfun, type_penalty=penalty_con, \
                                               mu = mu_con)
            f_pen = self.Penaly_fun.aug_obj  # functools.partial(penalized_objective,f1,[g1,g2], 100)

            # self.set_functions = [self.objective]
            # self.set_functions += [*self.constraints]
            # self.card_of_funcs = len(self.set_functions)

            _, sol1 = self.solve_naked(f_pen, x0, bounds, maxfun-1)

        return self.create_dictionary_for_solution(sol1)


    def create_dictionary_for_solution(self, sol):
        # CHANGE FOR THE SO FAR GOOD

        f_so_far, g_so_far, x_so_far = self.find_min_so_far(sol)
        output_dict = {}
        output_dict['g_store']       = self.Penaly_fun.g_his
        # print(sol)
        # print(sol.diagnostic_info['xk'])
        # print(np.array([sol.diagnostic_info['xk'].tolist()]))
        # print(np.array([sol.diagnostic_info['xk']])[0])
        # print(np.array([sol.diagnostic_info['xk']])[0].astype('d'))
        output_dict['x_store']       = sol[:,1:]
        output_dict['f_store']       = self.Penaly_fun.f_his
        output_dict['N_evals']       = len(f_so_far)
        output_dict['g_best_so_far'] = g_so_far
        output_dict['f_best_so_far'] = f_so_far
        output_dict['x_best_so_far'] = x_so_far
        output_dict['samples_at_iteration']            = np.arange(1, 1+len(f_so_far))
        return output_dict
   
    
    def find_min_so_far(self, sol):
        """
        This function find the best solution so far, mainly used for EI
        :param argmin: Boolean that if it is True the func returns which point is the best
        :type argmin:  Boolean
        """
        length = len(self.Penaly_fun.f_his)
        x_so_far = np.inf+np.zeros([length, sol.shape[1] - 1])
        f_so_far = np.inf+np.zeros(length)
        # print(self.card_of_funcs-1)
        # print(self.Penaly_fun.g_his[0])
        g_so_far = np.inf+np.zeros([length, self.card_of_funcs-1])
        
        for iter in range(length):
            min_f   = np.inf
            index = np.inf
            min_g = np.inf#self.Penaly_fun.g_his[iter]
            if self.card_of_funcs==1:

                for i in range(iter):
                        y = self.Penaly_fun.f_his[i]
                        if y< min_f:
                            min_f   = y
                            index = i
                        f_so_far[iter] = min_f
                        x_so_far[iter] = sol[index, 1:]
                        
            else:
                for i in range(iter):
                    y = self.Penaly_fun.f_his[i]
                    if y < min_f and all(self.Penaly_fun.g_his[i]<=0):
                        min_f   = y
                        index = i
                        min_g = self.Penaly_fun.g_his[i]
                    f_so_far[iter] = min_f
                    g_so_far[iter,:] = min_g
                    x_so_far[iter] = sol[index, 1:]
        
        return f_so_far, g_so_far, x_so_far


    
# bounds = np.array([[-1.5,1.5],[-1.5,1.5]])
# x0 = np.array([-1.5, 1.5])
    
# sol = SQSnobFitWrapper().solve(Problem_rosenbrock, x0, bounds, \
#                                       maxfun= 100, constraints=2)
    