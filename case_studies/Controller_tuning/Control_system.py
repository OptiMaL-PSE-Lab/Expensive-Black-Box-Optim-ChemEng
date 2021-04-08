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

def integrator_reactor_2st(pi, noise_mat, x0, xref, N, T):
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

            # x_new = solve_ivp(reactor_sys_2st,
            #                   t_span = (time[k],t),
            #                   y0 = x[-1], 
            #                   args = ([u1, u2],))
            # x.append(np.array(x_new["y"])[:,-1].tolist())
            
            x_new_list = solve_ivp(reactor_sys_2st,
                              t_span = (time[k],t),
                              y0 = x[-1], 
                              args = ([u1, u2],))

            x_new = np.array(x_new_list["y"])[:,-1] + \
                    np.array(noise_mat)*np.random.normal(0, 1, 2)

            x.append(x_new.tolist())
            
            time.append(t)
        
    #Calculate phi
    error1 = np.array(error1) #calc error
    error2 = np.array(error2)
    phi = np.sum(error1**2/xref[0]**2 + error2**2/xref[1]**2) # Phi function 
    
    return phi, x, U

def reactor_phi_2st(pi_, bounds, noise_mat, x0 = [.6, 310], xref = [.666, 308.489], N=200, T=3, \
                return_sys_resp = False):
    """This function takes an array of k values stored in an nd array pi and returns the value of phi associated
        with each of the triplets of k within pi."""
    pi = np.zeros(len(pi_))
    for i in range(len(pi_)):
        # print(pi_, bounds)
        pi[i] = bounds[i][0] + pi_[i]*(bounds[i][1] - bounds[i][0])
    # print(pi)
    if type(pi) == list:
        phi_sol, system_response, control_out = integrator_reactor_2st(pi, noise_mat, x0, xref, N, T)
    elif isinstance(pi, np.ndarray):
        if pi.ndim == 1:
            phi_sol, system_response, control_out = integrator_reactor_2st(pi.tolist(), noise_mat, x0, xref, N, T)
        else:
            # pi = pi.reshape(3,).tolist()
            pi = pi.reshape(10,).tolist()
            phi_sol, system_response, control_out = integrator_reactor_2st(pi, noise_mat, x0, xref, N, T)
    else:
        raise ValueError("Pi must be a list of 3 floats or an nd.array of shape (3,)")
    
    if return_sys_resp:
        return float(phi_sol), system_response, control_out
    else:
        return float(phi_sol), [np.maximum(float(phi_sol) - 500, 0)/1000]
    
   
def reactor_sys_2stNS(t, x, u):
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

def integrator_reactor_2stNS(pi, noise_mat, x0, xref, N, T):
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

            # x_new = solve_ivp(reactor_sys_2st,
            #                   t_span = (time[k],t),
            #                   y0 = x[-1], 
            #                   args = ([u1, u2],))
            # x.append(np.array(x_new["y"])[:,-1].tolist())
            
            x_new_list = solve_ivp(reactor_sys_2st,
                              t_span = (time[k],t),
                              y0 = x[-1], 
                              args = ([u1, u2],))

            x_new = np.array(x_new_list["y"])[:,-1] + \
                    np.array(noise_mat)*np.random.normal(0, 1, 2)

            x.append(x_new.tolist())
            
            time.append(t)
        
    #Calculate phi
    error1 = np.array(error1) #calc error
    error2 = np.array(error2)
    phi = np.sum(error1**2/xref[0]**2 + error2**2/xref[1]**2) # Phi function 
    
    return phi, x, U

def reactor_phi_2stNS(pi, noise_mat, x0 = [.6, 310], xref = [.666, 308.489], N=200, T=3, \
                return_sys_resp = False):
    """This function takes an array of k values stored in an nd array pi and returns the value of phi associated
        with each of the triplets of k within pi."""
    # print(pi)
    if type(pi) == list:
        phi_sol, system_response, control_out = integrator_reactor_2st(pi, noise_mat, x0, xref, N, T)
    elif isinstance(pi, np.ndarray):
        if pi.ndim == 1:
            phi_sol, system_response, control_out = integrator_reactor_2st(pi.tolist(), noise_mat, x0, xref, N, T)
        else:
            # pi = pi.reshape(3,).tolist()
            pi = pi.reshape(10,).tolist()
            phi_sol, system_response, control_out = integrator_reactor_2st(pi, noise_mat, x0, xref, N, T)
    else:
        raise ValueError("Pi must be a list of 3 floats or an nd.array of shape (3,)")
    
    if return_sys_resp:
        return float(phi_sol), system_response, control_out
    else:
        return float(phi_sol), [np.maximum(float(phi_sol) - 500, 0)/1000]
    




    
