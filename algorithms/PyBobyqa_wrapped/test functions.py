import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
import torch.autograd as autograd
import torch.optim as optim
from torch.distributions import constraints, transform_to
import numpy as np
import pyro
import pyro.contrib.gp as gp
from algorithms.Bayesian_opt_Pyro.utilities_full import BayesOpt
from test_functions import rosenbrock_constrained, quadratic_constrained

from utilities.general_utility_functions import PenaltyFunctions, plot_generic
assert pyro.__version__.startswith('1.5.1')
# pyro.enable_validation(True)  # can help with debugging
# pyro.set_rng_seed(0)

import pybobyqa
import functools

from algorithms.PyBobyqa_wrapped.Wrapper_for_pybobyqa import PyBobyqaWrapper
#
#
def Problem_rosenbrock(x):
    f1 = rosenbrock_constrained.rosenbrock_f
    g1 = rosenbrock_constrained.rosenbrock_g1
    g2 = rosenbrock_constrained.rosenbrock_g2

    return f1(x), [g1(x), g2(x)]


Penaly_fun = PenaltyFunctions(Problem_rosenbrock,type_penalty='l2', mu=1e3)
f_pen = Penaly_fun.aug_obj#functools.partial(penalized_objective,f1,[g1,g2], 100)



bounds = np.array([[-1.5,1.5],[-1.5,1.5]])
x0 = np.array([-0.5,1.5])

soln = PyBobyqaWrapper().solve(Problem_rosenbrock, x0, bounds=bounds.T, maxfun=20,constraints=2)#pybobyqa.solve(f_pen, x0, bounds=bounds.T)

#solution1 = BayesOpt().solve(f1, x0, bounds=bounds.T, print_iteration=True, constraints=[g1,g2])
#

for i in range(3):
    Bayes = BayesOpt()
    #
    solution2 = Bayes.solve(Problem_rosenbrock, x0, maxfun=40, acquisition='EI',bounds=bounds.T, print_iteration=True, constraints=2, casadi=True)

print("2")


#
#
# soln = PyBobyqaWrapper().solve(f1, x0, bounds=bounds.T, constraints=[g1,g2], maxfun=20)#pybobyqa.solve(f_pen, x0, bounds=bounds.T)
#
# plot_generic([soln, solution2.output_dict])
#
# f1 = quadratic_constrained.quadratic_f
# g1 = quadratic_constrained.quadratic_g
#
# f_pen = functools.partial(penalized_objective,f1,[g1,g2], 100)
#
# soln1 = pybobyqa.solve(f_pen, x0, bounds=bounds.T)
#
#
#
# solution2 = BayesOpt().solve(f1, x0, acquisition='EIC',bounds=bounds.T,
#                              print_iteration=True, constraints=[g1], casadi=True)