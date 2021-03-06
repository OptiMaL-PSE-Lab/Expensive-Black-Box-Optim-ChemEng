# import sys
# sys.path.insert(1, 'test_functions')
# sys.path.insert(1, 'algorithms/simplex')
import numpy as np 
from test_functions.quadratic_constrained import * 
from test_functions.rosenbrock_constrained import * 
from algorithms.simplex.simplex_method import simplex_method

d = 2
its = 50 
bounds = np.array([[-1.5,1.5],[-1.5,1.5]])
x0 = [-0.5,1.5]

def test_func(x):
    f = rosenbrock_f
    g1 = rosenbrock_g1
    g2 = rosenbrock_g2
    return f(x),[g1(x),g2(x)]

sol = simplex_method(test_func,x0,bounds,max_iter=its,constraints=2)

print(sol) 
