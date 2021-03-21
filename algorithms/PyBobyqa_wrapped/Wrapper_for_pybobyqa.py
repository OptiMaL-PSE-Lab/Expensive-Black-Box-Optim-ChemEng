import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import numpy as np

from algorithms.Bayesian_opt_Pyro.utilities_full import BayesOpt
from test_functions import rosenbrock_constrained, quadratic_constrained

from utilities.general_utility_functions import PenaltyFunctions



import pybobyqa#.solver as solver
import functools

class PyBobyqaWrapper:
    __metaclass__ = pybobyqa.solve
    def __init__(self, solver=pybobyqa.solve):
        self.solve_naked = pybobyqa.solve

    def solve(self, objfun, x0, args=(), bounds=None, npt=None,
              rhobeg=None, rhoend=1e-8, maxfun=100, nsamples=None,
              user_params=None, objfun_has_noise=False, seek_global_minimum=False,
              scaling_within_bounds=False, do_logging=True, print_progress=False,
              constraints =0, penalty_con='l2', mu_con=1e3):
        user_params1 = {'logging.save_diagnostic_info': True}
        user_params1['logging.save_xk'] = True
        user_params1['logging.save_xk'] = True
        self.maxfun = maxfun
        if (constraints)==0:#constraints==None:

            self.card_of_funcs = 1

            sol1 = self.solve_naked(objfun, x0, args=args, bounds=bounds, npt=npt,
                  rhobeg=rhobeg, rhoend=rhoend, maxfun=maxfun, nsamples=nsamples,
                  user_params=user_params1, objfun_has_noise=objfun_has_noise,
                  seek_global_minimum=seek_global_minimum, scaling_within_bounds=scaling_within_bounds,
                  do_logging=do_logging, print_progress=print_progress)
        else:
            #self.constraints = constraints
            self.card_of_funcs = 1 +constraints
            self.Penaly_fun = PenaltyFunctions(objfun, type_penalty=penalty_con, \
                                               mu = mu_con)
            f_pen = self.Penaly_fun.aug_obj  # functools.partial(penalized_objective,f1,[g1,g2], 100)

            # self.set_functions = [self.objective]
            # self.set_functions += [*self.constraints]
            # self.card_of_funcs = len(self.set_functions)

            sol1 = self.solve_naked(f_pen, x0, args=args, bounds=bounds, npt=npt,
                  rhobeg=rhobeg, rhoend=rhoend, maxfun=maxfun, nsamples=nsamples,
                  user_params=user_params1, objfun_has_noise=objfun_has_noise,
                  seek_global_minimum=seek_global_minimum, scaling_within_bounds=scaling_within_bounds,
                  do_logging=do_logging, print_progress=print_progress)

        return self.create_dictionary_for_solution(sol1)


    def create_dictionary_for_solution(self, sol):
        # CHANGE FOR THE SO FAR GOOD


        f_so_far, g_so_far = self.find_min_so_far()
        output_dict = {}
        output_dict['g_store']       = self.Penaly_fun.g_his
        # print(sol)
        # print(sol.diagnostic_info['xk'])
        # print(np.array([sol.diagnostic_info['xk'].tolist()]))
        # print(np.array([sol.diagnostic_info['xk']])[0])
        # print(np.array([sol.diagnostic_info['xk']])[0].astype('d'))
        output_dict['x_store']       = np.array([sol.diagnostic_info['xk'].tolist()])[0].astype('d')
        output_dict['f_store']       = self.Penaly_fun.f_his
        output_dict['N_evals']       = self.maxfun
        output_dict['g_best_so_far'] = g_so_far
        output_dict['f_best_so_far'] = f_so_far
        output_dict['x_best_so_far'] = np.array([sol.diagnostic_info['xk'].tolist()])[0].astype('d')
        output_dict['TR']            = np.array([sol.diagnostic_info['delta'].tolist()])[0].astype('d')
        return output_dict


    def find_min_so_far(self):
        """
        This function find the best solution so far, mainly used for EI
        :param argmin: Boolean that if it is True the func returns which point is the best
        :type argmin:  Boolean
        """
        f_so_far = np.inf+np.zeros(self.maxfun)
        g_so_far = np.inf+np.zeros([self.maxfun, self.card_of_funcs-1])
        for iter in range(len(self.Penaly_fun.f_his)):
            min   = np.inf
            index = np.inf
            min_g = np.inf#self.Penaly_fun.g_his[iter]
            if self.card_of_funcs==1:

                for i in range(iter):
                        y = self.Penaly_fun.f_his[i]
                        if y< min:
                            min   = y
                            index = i
                        f_so_far[iter] = min
            else:
                for i in range(iter):
                    y = self.Penaly_fun.f_his[i]
                    if y < min and all(self.Penaly_fun.g_his[i]<=0):
                        min   = y
                        index = i
                        min_g = self.Penaly_fun.g_his[i]
                    f_so_far[iter] = min
                    g_so_far[iter,:] = min_g

        if self.card_of_funcs>1:
            return f_so_far, g_so_far
        else:
            return f_so_far

