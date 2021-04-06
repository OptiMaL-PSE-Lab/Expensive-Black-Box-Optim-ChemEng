# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 13:31:22 2021

@author: dv516
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def sys_model(t, x, u):
    """Function modelling the dynamic behaviour of the system in question with a control term u. 
        Returns the state of the ODE at a given point in time"""
    dx1dt = 2*x[0] + x[1] + 2*u[0] - u[1]
    dx2dt = x[0] + 2*x[1] - 2*u[0] + 2*u[1]
    return [dx1dt, dx2dt]

def integrator_function(pi, x0, xref, N, T):
    """Function which takes input pi - 1x3 array of k values- and integrates the system equation to return a system response.
    This function is called as many times as there are k triplets in the phi function input.
    """
    Dt = T/N
    
    k_vals = pi #pi can only have a single triplet of k's. 
    error1 = []
    error2 = []
    U = []
    time = [0]
    x = [x0]

    # print(k_vals)
    
    for k in range(0, N): 
        #Determine the error and store it's value
        e1 = xref[0] - x[k][0]
        e2 = xref[1] - x[k][1]
        error1.append(e1)
        error2.append(e2)
        ## No error
        # rv1 = np.random.normal(loc=0, scale=0, size=1)
        
        #Determine the controller response u_k and store it's value
        # u = k_vals[0]*e + k_vals[1]*sum(error)
        # print(e, error)
        
        u1 = k_vals[0]*e1 + k_vals[1]*sum(error1)
        u2 = k_vals[2]*e2 + k_vals[3]*sum(error2)
            
        U.append([u1, u2]) # list of control outputs for a provided k_vals

        if k < N:
            #Determine the new x value and store it
            t = time[k] + Dt

            x_new = solve_ivp(sys_model,
                              t_span = (time[k],t),
                              y0 = x[-1], 
                              args = ([u1, u2],))
            x.append(np.array(x_new["y"])[:,-1].tolist())
            time.append(t)
        
    #Calculate phi
    error1 = np.array(error1) #calc error
    error2 = np.array(error2)
    phi = np.sum(error1**2 + error2**2) # Phi function 
    
    return phi, x, U

def integrator_functionRand(pi, x0, xref, N, T, deviation = 1):
    """Function which takes input pi - 1x3 array of k values- and integrates the system equation to return a system response.
    This function is called as many times as there are k triplets in the phi function input.
    """
    Dt = T/N
    
    k_vals = pi #pi can only have a single triplet of k's. 
    error1 = []
    error2 = []
    U = []
    time = [0]
    x = [x0]

    # print(k_vals)
    
    for k in range(0, N): 
        #Determine the error and store it's value
        e1 = xref[0] - x[k][0]
        e2 = xref[1] - x[k][1]
        error1.append(e1)
        error2.append(e2)
        ## No error
        # rv1 = np.random.normal(loc=0, scale=0, size=1)
        
        #Determine the controller response u_k and store it's value
        # u = k_vals[0]*e + k_vals[1]*sum(error)
        # print(e, error)
        
        u1 = k_vals[0]*e1 + k_vals[1]*sum(error1)
        u2 = k_vals[2]*e2 + k_vals[3]*sum(error2)
            
        U.append([u1, u2]) # list of control outputs for a provided k_vals

        if k < N:
            #Determine the new x value and store it
            t = time[k] + Dt

            x_new = solve_ivp(sys_model,
                              t_span = (time[k],t),
                              y0 = x[-1], 
                              args = ([u1, u2],))
            x_new = np.array(x_new["y"])[:,-1] + \
                    deviation*np.random.normal(0, 0.1, 2)
            x.append(x_new.tolist())
            time.append(t)
        
    #Calculate phi
    error1 = np.array(error1) #calc error
    error2 = np.array(error2)
    phi = np.sum(error1**2 + error2**2) # Phi function 
    
    return phi, x, U

def phi(pi, x0 = [15, 15], xref = [10, 10], N=200, T=3, return_sys_resp = False):
    """This function takes an array of k values stored in an nd array pi and returns the value of phi associated
        with each of the triplets of k within pi."""
    if type(pi) == list:
        phi_sol, system_response, control_out = integrator_function(pi, x0, xref, N, T)
    elif isinstance(pi, np.ndarray):
        if pi.ndim == 1:
            phi_sol, system_response, control_out = integrator_function(pi.tolist(), x0, xref, N, T)
        else:
            # pi = pi.reshape(3,).tolist()
            pi = pi.reshape(4,).tolist()
            phi_sol, system_response, control_out = integrator_function(pi, x0, xref, N, T)
    else:
        raise ValueError("Pi must be a list of 3 floats or an nd.array of shape (3,)")
        
    if return_sys_resp:
        return float(phi_sol), system_response, control_out
    else:
        return float(phi_sol), [0]
    
def phi_rand(pi, x0 = [15, 15], xref = [10, 10], N=200, T=3, deviation = 1, \
             return_sys_resp = False):
    """This function takes an array of k values stored in an nd array pi and returns the value of phi associated
        with each of the triplets of k within pi."""
    if type(pi) == list:
        phi_sol, system_response, control_out = integrator_functionRand(pi, x0, xref, N, T)
    elif isinstance(pi, np.ndarray):
        if pi.ndim == 1:
            phi_sol, system_response, control_out = integrator_functionRand(pi.tolist(), x0, xref, N, T)
        else:
            # pi = pi.reshape(3,).tolist()
            pi = pi.reshape(4,).tolist()
            phi_sol, system_response, control_out = integrator_functionRand(pi, x0, xref, N, T)
    else:
        raise ValueError("Pi must be a list of 3 floats or an nd.array of shape (3,)")
        
    if return_sys_resp:
        return float(phi_sol), system_response, control_out
    else:
        return float(phi_sol), [0]


def reactor_sys(t, x, u):
    """Function modelling the dynamic behaviour of the system in question with a control term u. 
        Returns the state of the ODE at a given point in time"""
    F_s = 20 # L/s
    V = 100 # L
    T_in = 275 # K
    UA = 2e4 # J/sK
    ro = 1000 # g / L
    cp = 4.2 # J/gK
    deltaH_R = - 596619 # J/mol
    k0 = 6.85e11 # L/smol
    E = 76534.704 # J/mol
    R = 8.314 # J/molK
    Tj = 250 # K
    cA_in = 1 # mol/L
    
    cA = x[0] ; T = x[1]
    
    k_T = k0*np.exp(-E/R/T)
    
    dcAdt = (F_s/V + u)*(cA_in - cA) - 2*k_T*cA**2
    dTdt = (F_s/V + u) *(T_in - T) + 2*(-deltaH_R/ro/cp)*k_T*cA**2 - UA/V/ro/cp*(T-Tj)
    
    return [dcAdt, dTdt]

def integrator_reactor(pi, x0, xref, N, T):
    """Function which takes input pi - 1x3 array of k values- and integrates the system equation to return a system response.
    This function is called as many times as there are k triplets in the phi function input.
    """
    Dt = T/N
    
    k_vals = pi #pi can only 9 k's. 
    error1 = []
    error2 = []
    U = []
    time = [0]
    x = [x0]

    # print(k_vals)
    
    for k in range(0, N): 
        #Determine the error and store it's value
        e1 = x[k][0] - xref[0]
        e2 = x[k][1] - xref[1]
        error1.append(e1)
        error2.append(e2)
        ## No error
        # rv1 = np.random.normal(loc=0, scale=0, size=1)
        
        #Determine the controller response u_k and store it's value
        # u = k_vals[0]*e + k_vals[1]*sum(error)
        # print(e, error)
        
        u = pi[0]*e1 + pi[1]*e2 + pi[2]*e1**2 + pi[3]*e1*e2 + pi[4]*e2**2 + \
            pi[5]*e1**3 + pi[6]*e1**2*e2 + pi[7]*e2**2*e1 + pi[8]*e2**3
            
        U.append(u) # list of control outputs for a provided k_vals

        if k < N:
            #Determine the new x value and store it
            t = time[k] + Dt

            x_new = solve_ivp(reactor_sys,
                              t_span = (time[k],t),
                              y0 = x[-1], 
                              args = (u,))
            x.append(np.array(x_new["y"])[:,-1].tolist())
            time.append(t)
        
    #Calculate phi
    error1 = np.array(error1) #calc error
    error2 = np.array(error2)
    phi = np.sum(error1**2 + error2**2) # Phi function 
    
    return phi, x, U

def reactor_phi(pi, x0 = [.6, 310], xref = [.666, 308.489], N=200, T=3, \
                return_sys_resp = False):
    """This function takes an array of k values stored in an nd array pi and returns the value of phi associated
        with each of the triplets of k within pi."""
    if type(pi) == list:
        phi_sol, system_response, control_out = integrator_reactor(pi, x0, xref, N, T)
    elif isinstance(pi, np.ndarray):
        if pi.ndim == 1:
            phi_sol, system_response, control_out = integrator_reactor(pi.tolist(), x0, xref, N, T)
        else:
            # pi = pi.reshape(3,).tolist()
            pi = pi.reshape(9,).tolist()
            phi_sol, system_response, control_out = integrator_reactor(pi, x0, xref, N, T)
    else:
        raise ValueError("Pi must be a list of 3 floats or an nd.array of shape (3,)")
    
    if return_sys_resp:
        return float(phi_sol), system_response, control_out
    else:
        return float(phi_sol), [0]
    
def reactor_sys_2st(t, x, u):
    """Function modelling the dynamic behaviour of the system in question with a control term u. 
        Returns the state of the ODE at a given point in time"""
    F_s = 20 # L/s
    V = 100 # L
    T_in = 275 # K
    UA = 2e4 # J/sK
    ro = 1000 # g / L
    cp = 4.2 # J/gK
    deltaH_R = - 596619 # J/mol
    k0 = 6.85e11 # L/smol
    E = 76534.704 # J/mol
    R = 8.314 # J/molK
    Tj = 250 # K
    cA_in = 1 # mol/L
    
    cA = x[0] ; T = x[1]
    
    k_T = k0*np.exp(-E/R/T)
    
    dcAdt = (F_s/V + u[0]) *(cA_in - cA) - 2*k_T*cA**2
    dTdt = (F_s/V + u[0]) *(T_in + u[1] - T) + 2*(-deltaH_R/ro/cp)*k_T*cA**2 - UA/V/ro/cp*(T-Tj)
    
    return [dcAdt, dTdt]

def integrator_reactor_2st(pi, x0, xref, N, T):
    """Function which takes input pi - 1x3 array of k values- and integrates the system equation to return a system response.
    This function is called as many times as there are k triplets in the phi function input.
    """
    Dt = T/N
    
    k_vals = pi #pi can only have a single triplet of k's. 
    error1 = []
    error2 = []
    U = []
    time = [0]
    x = [x0]

    # print(k_vals)
    
    for k in range(0, N): 
        #Determine the error and store it's value
        e1 = x[k][0] - xref[0]
        e2 = x[k][1] - xref[1]
        error1.append(e1)
        error2.append(e2)
        ## No error
        # rv1 = np.random.normal(loc=0, scale=0, size=1)
        
        #Determine the controller response u_k and store it's value
        # u = k_vals[0]*e + k_vals[1]*sum(error)
        # print(e, error)
    
        u1 = pi[0]*e1 + pi[1]*e2 + pi[2]*e1**2 + pi[3]*e1*e2 + pi[4]*e2**2 
        u2 = pi[5]*e1 + pi[6]*e2 + pi[7]*e1**2 + pi[8]*e1*e2 + pi[9]*e2**2
            
        U.append([u1, u2]) # list of control outputs for a provided k_vals

        if k < N:
            #Determine the new x value and store it
            t = time[k] + Dt

            x_new = solve_ivp(reactor_sys_2st,
                              t_span = (time[k],t),
                              y0 = x[-1], 
                              args = ([u1, u2],))
            x.append(np.array(x_new["y"])[:,-1].tolist())
            time.append(t)
        
    #Calculate phi
    error1 = np.array(error1) #calc error
    error2 = np.array(error2)
    phi = np.sum(error1**2/xref[0]**2 + error2**2/xref[1]**2) # Phi function 
    
    return phi, x, U

def reactor_phi_2st(pi, x0 = [.6, 310], xref = [.666, 308.489], N=200, T=3, \
                return_sys_resp = False):
    """This function takes an array of k values stored in an nd array pi and returns the value of phi associated
        with each of the triplets of k within pi."""
    if type(pi) == list:
        phi_sol, system_response, control_out = integrator_reactor_2st(pi, x0, xref, N, T)
    elif isinstance(pi, np.ndarray):
        if pi.ndim == 1:
            phi_sol, system_response, control_out = integrator_reactor_2st(pi.tolist(), x0, xref, N, T)
        else:
            # pi = pi.reshape(3,).tolist()
            pi = pi.reshape(10,).tolist()
            phi_sol, system_response, control_out = integrator_reactor_2st(pi, x0, xref, N, T)
    else:
        raise ValueError("Pi must be a list of 3 floats or an nd.array of shape (3,)")
    
    if return_sys_resp:
        return float(phi_sol), system_response, control_out
    else:
        return float(phi_sol), [0]
    

# # x0 = [2, 1, 2, 1]
# x0 = [4, 4, 4, 4]

# bounds = np.array([[0, 8], [0, 8], [0, 8], [0, 8]])

# control_Adam = Adam_optimizer(phi, x0, method = 'forward', \
#                                       bounds = bounds, alpha = 0.4, \
#                                       beta1 = 0.2, beta2  = 0.1, \
#                                       max_f_eval = 100)
    
# control_ApproxHessian = finite_Diff_Newton(phi, x0, bounds = bounds, max_f_eval = 100)

# control_BFGS = BFGS_optimizer(phi, x0, bounds = bounds, max_f_eval = 100)
   
# init_radius =  2

# N_CQSO = 5
# control_CQSO_list = []
# for i in range(N_CQSO):
#     temp = cvx_quad_surr_opt(phi, x0, init_radius, bounds = bounds, \
#                              beta_red = 0.5, max_f_eval = 100)
#     control_CQSO_list.append(temp)

    
# x0 = [8, 8, 3.4, 0]

# bounds = np.array([[-10, 10], [-10, 10], [-8, 8], [-8, 8]])

# reactor_Adam = Adam_optimizer(reactor_phi, x0, method = 'forward', \
#                                       bounds = bounds, alpha = 0.4, \
#                                       beta1 = 0.2, beta2  = 0.1, \
#                                       max_f_eval = 100)
    
# reactor_ApproxHessian = finite_Diff_Newton(reactor_phi, x0, bounds = bounds, max_f_eval = 100)

# reactor_BFGS = BFGS_optimizer(reactor_phi, x0, bounds = bounds, max_f_eval = 100)
   
# init_radius =  1

# N_CQSO = 5
# reactor_CQSO_list = []
# for i in range(N_CQSO):
#     temp = cvx_quad_surr_opt(reactor_phi, x0, init_radius, bounds = bounds, \
#                              beta_red = 0.5, max_f_eval = 100)
#     reactor_CQSO_list.append(temp)

# _, sys_resp = reactor_phi([7.72024476180943, 7.833924577739739, 2.583729334999448, -1.0492540859258679], \
#                           x0 = [.6, 310], return_sys_resp = True)
# # _, sys_resp = reactor_phi([7.72024476180943, 7.833924577739739, 2.583729334999448, -1.0492540859258679], \
# #                           x0 = [.666, 308.489], return_sys_resp = True)
# x1 = np.array(sys_resp)[:,0] ; x2 = np.array(sys_resp)[:,1]
# plt.plot(np.arange(len(x1)), x1)
# plt.show()
# plt.clf()
# plt.plot(np.arange(len(x1)), x2)

# x0 = [50, 10, 50, 10]

# bounds = np.array([[0, 100], [0, 50], [0, 100], [0, 50]])

# reactor2_Adam = Adam_optimizer(reactor_phi_2st, x0, method = 'forward', \
#                                       bounds = bounds, alpha = 0.4, \
#                                       beta1 = 0.2, beta2  = 0.1, \
#                                       max_f_eval = 100)
    
# reactor2_ApproxHessian = finite_Diff_Newton(reactor_phi_2st, x0, bounds = bounds, max_f_eval = 100)

# reactor2_BFGS = BFGS_optimizer(reactor_phi_2st, x0, bounds = bounds, max_f_eval = 100)
   
# init_radius =  50

# N_CQSO = 5
# reactor2_CQSO_list = []
# for i in range(N_CQSO):
#     temp = cvx_quad_surr_opt(reactor_phi_2st, x0, init_radius, bounds = bounds, \
#                              beta_red = 0.9, max_f_eval = 100)
#     reactor2_CQSO_list.append(temp)

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

# T = 50
# pi = [.8746, .0257, -1.43388, -0.00131, 0.00016, 55.8692, 0.7159, .0188, .00017]
# pi_new = pi.copy()
# _, sys_resp, control_resp = reactor_phi(pi, x0 = [.116, 368.5], N = int(1000), \
#                                         T = T, return_sys_resp = True)

# # _, sys_resp, control_resp = reactor_phi_2st([100, 17.78006887394654, 55.110302129276256, 20.064282954116894], \
#                           # x0 = [.6, 312], return_sys_resp = True, T = T, N = int(200/3*T))

# # _, sys_resp = reactor_phi_2st([0, 0, 0, 0], \
# #                           x0 = [.11, 370], return_sys_resp = True, T = 20)

# x1 = np.array(sys_resp)[:,0] ; x2 = np.array(sys_resp)[:,1]
# u = np.array(control_resp)
# plt.plot(np.arange(len(x1))/len(x1)*T, x1)
# plt.plot([0, T], [.666, .666], '--k')
# plt.show()
# plt.clf()
# plt.plot(np.arange(len(x1))/len(x1)*T, x2)
# plt.plot([0, T], [308.5, 308.5], '--k')
# plt.show()
# plt.clf()
# plt.plot(np.arange(len(u))*T/len(u), u)
    
