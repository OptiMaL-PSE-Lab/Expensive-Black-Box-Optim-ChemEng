# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 22:40:51 2021

@author: dv516
"""

import SQSnobFit
import scipydirect
import numpy as np

f = lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

bounds = np.array([[-1.5, 1.5], [-1.5, 1.5]], dtype=float)
budget = 80      # larger budget needed for full convergence
x0 = np.array([0.5, 0.5])

# from SQSnobFit import optset
# options = optset(maxmp=len(x0)+6)

result, history = SQSnobFit.minimize(f, x0, bounds, budget)


def obj(x):
    """Six-hump camelback function"""
    
    x1 = x[0]
    x2 = x[1]
    
    f = (4 - 2.1*(x1*x1) + (x1*x1*x1*x1)/3.0)*(x1*x1) + x1*x2 + (-4 + 4*(x2*x2))*(x2*x2)
    return f

bounds = [(-3, 3), (-2, 2)]
res = scipydirect.minimize(obj, bounds)
print(res)

# bounds_ = [(-1.5, 1.5), (-1.5, 1.5)]

# test = scipydirect.minimize(f, bounds_)



