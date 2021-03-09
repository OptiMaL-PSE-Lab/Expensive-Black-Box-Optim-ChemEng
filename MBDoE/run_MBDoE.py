from MBDoE.construct_MBDoE_funs import obj_MBDoE, con_MBDoE, obj_MBDoE_moo
from algorithms.PyBobyqa_wrapped.Wrapper_for_pybobyqa import PyBobyqaWrapper
import numpy as np
import matplotlib.pyplot as plt

import numpy as np

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.model.problem import Problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import os
from pymoo.configuration import get_pymoo
from pymoo.decision_making.high_tradeoff_inverted import HighTradeoffPointsInverted
class MyProblem(Problem):

    def __init__(self):
        super().__init__(n_var=4,
                         n_obj=2,
                         n_constr=0,
                         xl=np.array([0]*4),
                         xu=np.array([1]*4),
                         elementwise_evaluation=True)

    def _evaluate(self, x, out, *args, **kwargs):
        f1_eval = obj_MBDoE('E')
        f2_eval = obj_MBDoE('E1')

        f1 = f1_eval(x)#x[0] ** 2 + x[1] ** 2
        f2 = f2_eval(x)#(x[0] - 1) ** 2 + x[1] ** 2

        g1 = 2 * (x[0] - 0.1) * (x[0] - 0.9) / 0.18
        g2 = - 20 * (x[0] - 0.4) * (x[0] - 0.6) / 4.8

        out["F"] = [f1, f2]
        out["G"] = [g1, g2]


problem = MyProblem()

algorithm = NSGA2(pop_size=100)

res = minimize(problem,
               algorithm,
               ("n_gen", 100),
               verbose=True,
               seed=1)

plot = Scatter()

F=res.F
from pymoo.factory import get_decision_making
dm = get_decision_making("high-tradeoff")
from pymoo.factory import get_problem, get_visualization, get_decomposition

I = dm.do(F)
weights = np.array([0.5, 0.5])
decomp = get_decomposition("asf")
I1 = get_decomposition("asf").do(F, weights).argmin()


plot.add(F, color="blue", alpha=0.2, s=10)
plot.add(F[I], color="red", s=30)
plot.add(F[I1], color="red", s=30)
plot.do()
plot.apply(lambda ax: ax.arrow(0, 0, 0.5, 0.5, color='black',
                               head_width=0.01, head_length=0.01, alpha=0.4))
plot.do()
plot.apply(lambda ax: ax.arrow(0, 0, 0.5, 0.5, color='black',
                               head_width=0.01, head_length=0.01, alpha=0.4))

plot.add(res.F, color="red")
plot.show()

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