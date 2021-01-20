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

    @pybobyqa.solve
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


    def create_dictionary_for_solution(self, sol):
        


        self.output_dict = {}
