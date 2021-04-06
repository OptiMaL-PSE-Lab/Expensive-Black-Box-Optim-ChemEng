# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 01:21:42 2021

@author: dv516
"""

import numpy as np
import pickle

import pyro
pyro.enable_validation(True)  # can help with debugging
pyro.set_rng_seed(1)

from algorithms.PyBobyqa_wrapped.Wrapper_for_pybobyqa import PyBobyqaWrapper
from algorithms.Bayesian_opt_Pyro.utilities_full import BayesOpt
from algorithms.nesterov_random.nesterov_random import nesterov_random
from algorithms.simplex.simplex_method import simplex_method
from algorithms.CUATRO.CUATRO import CUATRO
from algorithms.Finite_differences.Finite_differences import finite_Diff_Newton
from algorithms.Finite_differences.Finite_differences import Adam_optimizer
from algorithms.Finite_differences.Finite_differences import BFGS_optimizer
from algorithms.SQSnobfit_wrapped.Wrapper_for_SQSnobfit import SQSnobFitWrapper
from algorithms.DIRECT_wrapped.Wrapper_for_Direct import DIRECTWrapper

from case_studies.Controller_tuning.Control_system import reactor_phi, reactor_phi_2st

import matplotlib.pyplot as plt


def average_from_list(solutions_list):
    N = len(solutions_list)
    f_best_all = np.zeros((N, 100))
    for i in range(N):
        f_best = np.array(solutions_list[i]['f_best_so_far'])
        x_ind = np.array(solutions_list[i]['samples_at_iteration'])
        for j in range(100):
            ind = np.where(x_ind <= j+1)
            if len(ind[0]) == 0:
                f_best_all[i, j] = f_best[0]
            else:
                f_best_all[i, j] = f_best[ind][-1]
    f_median = np.median(f_best_all, axis = 0)
    # f_av = np.average(f_best_all, axis = 0)
    # f_std = np.std(f_best_all, axis = 0)
    f_min = np.min(f_best_all, axis = 0)
    f_max = np.max(f_best_all, axis = 0)
    return f_best_all, f_median, f_min, f_max

def plot_reactor_respRand(pi, plot, method, x0 = [.6, 310], xref = [.666, 308.489], \
                          two_states = False, N=200, T=8):
    
    if not two_states:
        ax1, ax2, ax3 = plot
        _, sys_resp, control_resp = reactor_phi(pi, x0 = x0, N = N, \
                                    T = T, return_sys_resp = True)
    else:
        ax1, ax2, ax3, ax4 = plot
        _, sys_resp, control_resp = reactor_phi_2st(pi, x0 = x0, N = N, \
                                    T = T, return_sys_resp = True)
    
    x1 = np.array(sys_resp)[:,0] ; x2 = np.array(sys_resp)[:,1]
    ax1.plot(np.arange(len(x1))/len(x1)*T, x1, label = method + ': $x_1$')
    ax1.plot([0, T], [xref[0], xref[0]], '--k')
    ax2.plot([0, T], [xref[1], xref[1]], '--k')
    ax2.plot(np.arange(len(x2))/len(x2)*T, x2, label =  method + ': $x_2$')
    if not two_states:
        u = np.array(control_resp)
        ax3.plot(np.arange(len(u))/len(u)*T, u, label = method + ': $u$')
    else:
        u1 = np.array(control_resp)[:,0] ; u2 = np.array(control_resp)[:,1]
        ax3.plot(np.arange(len(u1))/len(u1)*T, u1, label = method + ': $u_1$')
        ax4.plot(np.arange(len(u2))/len(u2)*T, u2, label = method + ': $u_2$')

    return ax1, ax2, ax3



x0 = np.array([4, 4, 4, 4])
bounds = np.array([[0, 8], [0, 8], [0, 8], [0, 8]])

max_f_eval = 100


fig1, fig2, fig3 = plt.figure(), plt.figure(), plt.figure()
ax1, ax2 = fig1.add_subplot(211), fig1.add_subplot(212)
ax3 = fig2.add_subplot()


method = 'Init'
pi = [.8746, .0257, -1.43388, -0.00131, 0.00016, 55.8692, 0.7159, .0188, .00017]
plot = (ax1, ax2, ax3)
plot_reactor_respRand(pi, plot, method, x0 = [.116, 368.489], N = 200*5, T = 20)


pi_init = [.8746, .0257, -1.43388, -0.00131, 0.00016, 0, 0, 0, 0, 0]

fig1, fig2, fig3 = plt.figure(), plt.figure(), plt.figure()
ax1, ax2 = fig1.add_subplot(211), fig1.add_subplot(212)
ax4, ax5 = fig3.add_subplot(211), fig3.add_subplot(212)

# plot = (ax1, ax2, ax3)
# plot_reactor_respRand(pi_init[:-1], plot, method, x0 = [.116, 368.489], N = 200*5, T = 20)

test_vec = [0.461594, 0.0506861,	-0.75677,	-0.00258361,	0.000315556,	7.77422,	0,	-5.31067,	0.00485185,	-0.0013037]

plot = (ax1, ax2, ax4, ax5)
# plot_reactor_respRand(test_vec, plot, method, x0 = [.116, 368.489], N = 200, T = 20, two_states = True)
plot_reactor_respRand(test_vec, plot, method, N = 200, T = 20, two_states = True)




cost = lambda x, grad: reactor_phi_2st(x, x0 = [.116, 368.489], N = 200, T = 20)

bounds = np.zeros((10, 2))

for i in range(5):
    if pi[i] > 0:
        bounds[i] = [pi[i]/2, pi[i]*2]
        bounds[i+5] = [-pi[i]*10, pi[i]*10]
    else:
        bounds[i] = [pi[i]*2, pi[i]/2]
        bounds[i+5] = [pi[i]*10, -pi[i]*10]
 
x0 = np.array(pi_init.copy())
max_f_eval = 100
 
test = DIRECTWrapper().solve(cost, x0, bounds, \
                             maxfun = max_f_eval, constraints=1)



# x1r = .666 ; x2r = 308.489 ; 
# pi = [.8746, .0257, -1.43388, -0.00131, 0.00016, 55.8692, 0.7159, .0188, .00017]
# pi_scaled = pi.copy()
# pi_scaled[0] /= x1r ; pi_scaled[1] /= x2r ; pi_scaled[2] /= x1r**2
# pi_scaled[3] /= (x1r*x2r) ; pi_scaled[4] /= x2r**2 ; pi_scaled[5] /= x1r**3
# pi_scaled[6] /= (x1r**2*x2r) ; pi_scaled[7] /= (x1r*x2r**2) ; 
# pi_scaled[8] /= x2r**3
# print(pi, pi_scaled)
# pi_scaled = pi.copy()
# pi_scaled[0] *= x1r ; pi_scaled[1] *= x2r ; pi_scaled[2] *= x1r**2
# pi_scaled[3] *= (x1r*x2r) ; pi_scaled[4] *= x2r**2 ; pi_scaled[5] *= x1r**3
# pi_scaled[6] *= (x1r**2*x2r) ; pi_scaled[7] *= (x1r*x2r**2) ; 
# pi_scaled[8] *= x2r**3
# print(pi_scaled)









