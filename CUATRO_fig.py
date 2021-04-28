# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 23:04:05 2021

@author: dv516
"""

from algorithms.CUATRO.CUATRO import CUATRO
import numpy as np
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt


def plot_trust(ax, x_trust, r):
    ax.plot(x_trust[0], x_trust[1], '*k', label = 'Trust Center')
    ax.plot([x_trust[0] - r, x_trust[0] + r,], [x_trust[1] + r, x_trust[1] + r], '--k')
    ax.plot([x_trust[0] - r, x_trust[0] + r,], [x_trust[1] - r, x_trust[1] - r], '--k')
    ax.plot([x_trust[0] - r, x_trust[0] - r,], [x_trust[1] - r, x_trust[1] + r], '--k')
    ax.plot([x_trust[0] + r, x_trust[0] + r,], [x_trust[1] - r, x_trust[1] + r], '--k')
    return ax
def plot_ellipse(ax, X, Y, Z, l = [0.1, 1, 2, 3, 4, 6, 10, 20, 50]):
    # ax.plot(x_true[0], x_true[1], '*', label = 'Minimum')
    CS = ax.contour(X,Y,Z, levels = l)
    ax.clabel(CS, inline = 1, fontsize = 10)
    return ax
def surrogate_plot(ax, X_data, Y_data, p, x_min, x_center, r):
    ax.scatter(X_data[:,0], X_data[:,1], c = 'brown', label = 'data')
    x = np.linspace(x_center[0] - r, x_center[0] + r)
    y = np.linspace(x_center[1] - r, x_center[1] + r)
    X,Y = np.meshgrid(x, y)
    a, b, c, d, e = p
    Z = a*X**2 + b*Y**2 + c*X + d*Y + e
    CS = ax.contour(X,Y,Z, levels = np.linspace(np.min(Y_data), np.max(Y_data), 5))
    ax.clabel(CS, inline = 1, fontsize = 10)
    ax.plot(x_min[0], x_min[1], marker = 'v', label = 'Trust min')
    return ax

def surrogate(x, a, b, c, d, e):
    ## Surrogate model: 
    ## ax^2 + by^2 + cx + dy + e
    ## where x[0] = x and x[1] = y
    return a*x[:,0]**2 + b*x[:,1]**2 + c*x[:,0] + d*x[:,1] + e
def surrogate_min(x, a, b, c, d, e):
    return a*x[0]**2 + b*x[1]**2 + c*x[0] + d*x[1] + e


## Himmelblau
function = lambda x: (x[0]**2 + x[1] - 11)**2 + (x[0]+x[1]**2 - 7)**2
x0 = np.zeros(2)

x_true = minimize(function, x0, tol = 1e-10)['x']
# print('True: ', x_true)
x_trust = x0
epsilon = 1e-6

N = 0
N_max = 1

# lim = 3
# x = np.linspace(-lim, lim, 200)
# y = np.linspace(-lim, lim, 200)
# X,Y = np.meshgrid(x, y)
# Z = np.exp(X + 3*Y - 0.1) + np.exp(X - 3*Y - 0.1) + np.exp(-X - 0.1)

lim = 5
x = np.linspace(-lim, lim, 200)
y = np.linspace(-lim, lim, 200)
X,Y = np.meshgrid(x, y)
Z = (X**2 + Y - 11)**2 + (X+Y**2 - 7)**2

# r = 0.5
# discount = 0.95

r = 3
discount = 0.8
threshold = 1e-5


while (np.linalg.norm(x_true - x_trust) > threshold) and (N < N_max):
    x = np.random.uniform(x_trust - r, x_trust + r, (6, 2))
    y = np.array([function(x_) for x_ in x])
    p0 = np.zeros(5)
    p, _ = curve_fit(surrogate, x, y, p0=p0)
    min_f = lambda x: surrogate_min(x, *p)
    result = minimize(min_f, x_trust, bounds = [(x_ - r, x_ + r) for x_ in x_trust])
    x_trust_min = result['x']
    
    fig = plt.figure(figsize = (6,8))
    ax = fig.add_subplot(211)
    ax = plot_ellipse(ax, X, Y, Z, l = np.logspace(-0.5, 4, 10))
    ax = plot_trust(ax, x_trust, r)
    ax = surrogate_plot(ax, x, y, p, x_trust_min, x_trust, r)
    x_trust = x_trust_min

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # ax.set_xlim([-2.5, 1.5])
    # ax.set_ylim([-1.3, 1.3])
    ax.set_xlim([-6, 6])
    ax.set_ylim([-6, 6])
    ax.legend()
    
    N += 1
    r *= discount





