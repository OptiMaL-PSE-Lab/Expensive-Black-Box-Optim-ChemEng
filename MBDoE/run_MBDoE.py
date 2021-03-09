from MBDoE.construct_MBDoE_funs import obj_MBDoE, con_MBDoE, obj_MBDoE_moo
from algorithms.PyBobyqa_wrapped.Wrapper_for_pybobyqa import PyBobyqaWrapper
import numpy as np
import matplotlib.pyplot as plt

import numpy as np


obj = obj_MBDoE('E')


bounds = np.array([[0,1.],[0,1.],[0,1.],[0,1.]])
x0 = np.array([0.1]*4)

g = con_MBDoE

soln = PyBobyqaWrapper().solve(obj, x0, bounds=bounds.T,constraints=[g], maxfun=1000)#pybobyqa.solve(f_pen, x0, bounds=bounds.T)

obj1 = obj_MBDoE('E1')


bounds = np.array([[0,1.],[0,1.],[0,1.],[0,1.]])
x0 = np.array([0.1]*4)

g1 = con_MBDoE

soln1 = PyBobyqaWrapper().solve(obj1, x0, bounds=bounds.T,constraints=[g1], maxfun=1000)#pybobyqa.solve(f_pen, x0, bounds=bounds.T)


print(2)