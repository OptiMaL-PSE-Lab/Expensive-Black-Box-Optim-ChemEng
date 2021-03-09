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

params = np.linspace(0,1,50)
his_x    = []
his_obj1 = []
his_obj2 = []
for i in range(50):
    obj2 = obj_MBDoE_moo('MOO3', params[i])


    bounds = np.array([[0,1.],[0,1.],[0,1.],[0,1.]])
    x0 = np.array([0.1]*4)

    g2 = con_MBDoE

    soln2 = PyBobyqaWrapper().solve(obj2, x0, bounds=bounds.T,constraints=[g2], maxfun=1000)#pybobyqa.solve(f_pen, x0, bounds=bounds.T)
    his_x    +=[(soln2['x_best_so_far'])[-1]]
    his_obj2 += [obj1((soln2['x_best_so_far'])[-1]).copy()]
    his_obj1 += [obj((soln2['x_best_so_far'])[-1]).copy()]

#
# obj3 = obj_MBDoE('MOO1')
#
#
# bounds = np.array([[0,1.],[0,1.],[0,1.],[0,1.]])
# x0 = np.array([0.1]*4)
#
# g3 = con_MBDoE
#
# soln3 = PyBobyqaWrapper().solve(obj3, x0, bounds=bounds.T,constraints=[g3], maxfun=1000)#pybobyqa.solve(f_pen, x0, bounds=bounds.T)
#
# obj4 = obj_MBDoE('MOO2')
#
#
# bounds = np.array([[0,1.],[0,1.],[0,1.],[0,1.]])
# x0 = np.array([0.1]*4)
#
# g4 = con_MBDoE
#
# soln4 = PyBobyqaWrapper().solve(obj4, x0, bounds=bounds.T,constraints=[g4], maxfun=1000)#pybobyqa.solve(f_pen, x0, bounds=bounds.T)
#
# obj5 = obj_MBDoE('MOO3')
#
#
# bounds = np.array([[0,1.],[0,1.],[0,1.],[0,1.]])
# x0 = np.array([0.1]*4)
#
# g5 = con_MBDoE
#
# soln5 = PyBobyqaWrapper().solve(obj5, x0, bounds=bounds.T,constraints=[g5], maxfun=1000)#pybobyqa.solve(f_pen, x0, bounds=bounds.T)
#
#
# obj6 = obj_MBDoE('MOO4')
#
#
# bounds = np.array([[0,1.],[0,1.],[0,1.],[0,1.]])
# x0 = np.array([0.1]*4)
#
# g6 = con_MBDoE
#
# soln6 = PyBobyqaWrapper().solve(obj6, x0, bounds=bounds.T,constraints=[g6], maxfun=1000)#pybobyqa.solve(f_pen, x0, bounds=bounds.T)
#

print(2)