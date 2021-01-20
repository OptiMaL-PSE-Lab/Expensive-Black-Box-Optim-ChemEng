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
              rhobeg=None, rhoend=1e-8, maxfun=None, nsamples=None,
              user_params=None, objfun_has_noise=False, seek_global_minimum=False,
              scaling_within_bounds=False, do_logging=True, print_progress=False):
        user_params1 = {'logging.save_diagnostic_info': True}
        user_params1['logging.save_xk'] = True
        user_params1['logging.save_xk'] = True
        sol1 = self.solve_naked(objfun, x0, args=args, bounds=bounds, npt=npt,
              rhobeg=rhobeg, rhoend=rhoend, maxfun=maxfun, nsamples=nsamples,
              user_params=user_params1, objfun_has_noise=objfun_has_noise,
              seek_global_minimum=seek_global_minimum, scaling_within_bounds=scaling_within_bounds,
              do_logging=do_logging, print_progress=print_progress)

        self.create_dictionary_for_solution(sol1)


    def create_dictionary_for_solution(self, sol):



        self.output_dict = {}



    def find_min_so_far(self, argmin=False):
        """
        This function find the best solution so far, mainly used for EI
        :param argmin: Boolean that if it is True the func returns which point is the best
        :type argmin:  Boolean
        """
        min = np.inf
        index = np.inf
        if self.card_of_funcs==1:

            for i in range(len(self.X)):
                    y = self.Y[i,0]
                    if y< min:
                        min   = y
                        index = i
        else:
            for i in range(len(self.X)):
                y = self.Y[i, 0]
                if y < min and all(self.Y[i,1:].data.numpy()<=0):
                    min   = y
                    index = i
        if argmin:
            return min, index
        else:
            return min
