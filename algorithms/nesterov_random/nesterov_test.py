import sys
sys.path.insert(1, 'test_functions')
import numpy as np 
from quadratic_constrained import * 
from rosenbrock_constrained import * 
from nesterov_random import nesterov_random


def problem_rosenbrock(x):
    f = rosenbrock_f
    g1 = rosenbrock_g1
    g2 = rosenbrock_g2
    return f(x),[g1(x),g2(x)]
    
    
d = 2
its = 50 
bounds = np.array([[-1.5,1.5],[-1.5,1.5]])
x0 = [-0.5,1.5]

sol = nesterov_random(problem_rosenbrock,x0,bounds,max_iter=its,constraints=2)

print(sol) 