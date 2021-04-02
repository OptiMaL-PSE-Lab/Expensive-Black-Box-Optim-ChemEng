from case_studies.MBDoE.construct_MBDoE_funs import obj_MBDoE, MBDoE
from algorithms.PyBobyqa_wrapped.Wrapper_for_pybobyqa import PyBobyqaWrapper
import numpy as np
import matplotlib.pyplot as plt

import numpy as np


obj = MBDoE#obj_MBDoE('E')


bounds = np.array([[0,1.],[0,1.],[0,1.],[0,1.]])
x0 = np.array([0.1]*4)

obj_uncon = lambda x: MBDoE(x)[0]
soln = PyBobyqaWrapper().solve(obj_uncon, x0, bounds=bounds.T, maxfun=1000)#pybobyqa.solve(f_pen, x0, bounds=bounds.T)

obj1 = obj_MBDoE('E1')


bounds = np.array([[0,1.],[0,1.],[0,1.],[0,1.]])
x0 = np.array([0.1]*4)

g1 = con_MBDoE

soln1 = PyBobyqaWrapper().solve(obj1, x0, bounds=bounds.T,constraints=[g1], maxfun=1000)#pybobyqa.solve(f_pen, x0, bounds=bounds.T)


print(2)