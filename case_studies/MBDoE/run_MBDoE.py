from case_studies.MBDoE.construct_MBDoE_funs import set_funcs_mbdoe

import numpy as np
import matplotlib.pyplot as plt

import numpy as np


bounds = np.array([[0,1.],[0,1.],[0,1.],[0,1.]])
x0 = np.array([0.1]*4)


#soln = PyBobyqaWrapper().solve(obj, x0, bounds=bounds.T,constraints=[g], maxfun=1000)#pybobyqa.solve(f_pen, x0, bounds=bounds.T)



bounds = np.array([[0,1.],[0,1.],[0,1.],[0,1.]])
x0 = np.array([0.1]*4)

obj1 = set_funcs_mbdoe(x0)
obj1_unc = lambda x: set_funcs_mbdoe(x)[0]

