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

from utilities.general_utility_functions import PenaltyFunctions
assert pyro.__version__.startswith('1.5.1')
pyro.enable_validation(True)  # can help with debugging
pyro.set_rng_seed(1)

import pybobyqa
import functools






f1 = rosenbrock_constrained.rosenbrock_f
g1 = rosenbrock_constrained.rosenbrock_g1
g2 = rosenbrock_constrained.rosenbrock_g2

Penaly_fun = PenaltyFunctions(f1,[g1,g2],type_penalty='l2', mu=1e3)
f_pen = Penaly_fun.aug_obj#functools.partial(penalized_objective,f1,[g1,g2], 100)




bounds = np.array([[-1.5,1.5],[-1.5,1.5]])
x0 = np.array([0.5,0.5])

soln = pybobyqa.solve(f_pen, x0, bounds=bounds.T)


#solution1 = BayesOpt().solve(f1, x0, bounds=bounds.T, print_iteration=True, constraints=[g1,g2])
solution = BayesOpt().solve(f1, x0, acquisition='EIC',bounds=bounds.T, print_iteration=True, constraints=[g1, g2], casadi=True)


f1 = quadratic_constrained.quadratic_f
g1 = quadratic_constrained.quadratic_g

f_pen = functools.partial(penalized_objective,f1,[g1,g2], 100)

soln1 = pybobyqa.solve(f_pen, x0, bounds=bounds.T)



solution2 = BayesOpt().solve(f1, x0, acquisition='EIC',bounds=bounds.T,
                             print_iteration=True, constraints=[g1], casadi=True)
