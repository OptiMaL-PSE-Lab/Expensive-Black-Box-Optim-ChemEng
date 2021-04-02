import case_studies.MBDoE.utilities_leeds_pre as utilities
import case_studies.MBDoE.utilities_leeds as utilities1

import case_studies.MBDoE.ut as ut
import numpy as np
import casadi as ca

true_theta = [np.log(57.9 * 60. * 10. ** (-2)), 33.3 / 10, np.log(2.7 * 60. * 10. ** (-2)), 35.3 / 10,
              np.log(0.865 * 60. * 10. ** (-2)), 38.9 / 10, np.log(1.63 * 60. * 10. ** (-2)), 44.8 / 10]
# x_meas, u_meas, _, _, _, dt = utilities.give_data_from_sim(N_exp, PC, date, file, true_theta, info)


V, c1o, c2o = 2.7,	2.03, 4.17


lbu = [60, 0.3, 0.3, 0.3]  #[60, 0.3, 0.3, 0.3]#, 0.1]
ubu = [130, 3, 3, 3]#[130, 3, 3, 3]#, 2.0]


bounds = (lbu, ubu)

f, nu, nx, ntheta = utilities.plant_model('sensitivity')

sigma = 1e-8*np.ones(4)


u   = np.random.rand(4)

u_t = (u)*(np.array(ubu)-np.array(lbu))+np.array(lbu)


#
# np.random.seed(seed=0)
#
# T = 200.
#
#
# N_exp    = 8#5
# PC       = 'Panos'
# #PC       =  'ppets' #For laptop
# #PC       =  'ppets' #For laptop
# labbot   = True
# file     = ['NaN']*N_exp
# date     = 'zippedRuns/06-Sep-2019/20190906_UCL_Panos_3'#'03-Sep-2019/20190903_UCL_Panos'#'07-Oct-2019/Run3/20191007_UCL_SS3'##
#
# #date1      = '06-Sep-2019/20190906_UCL_Panos_3'#'trial/20190906_UCL_Panos_3'##'03-Sep-2019/20190903_UCL_Panos'#'07-Oct-2019/Run3/20191007_UCL_SS3'##
# ##info1      = '/Exp_Setup_Info_06-September-2019_11_34_19.csv'#
# #condition1 = '/Process_Conditions_Bounds_06-September-2019_11_34_19.csv'#
#
# bayesopt = True
#
# time_to_wait = 60 * 3000
# time_counter = 0
# #file0 = '/Users/' + PC + '/Dropbox/UCL/' + date + '/Peaks and Concentrations epoch_' + str(N_exp) + '_Leeds_'+lab+'_nLabBots_2.csv'
#
#
#
# if labbot==True:
#     f, nu, nx, ntheta = utilities.plant_model([])
#     #x_meas0, u_meas0, V0, c1o0, c2o0, dt0 = give_data_from_exp_recal(nu, nx, ntheta, N_exp, PC, date1, file, 1, info1)#Need to change dates
#     x_meas, u_meas, V, c1o, c2o, dt = utilities.give_data_from_exp(nu, nx, ntheta, N_exp, PC, date, file)
# #    compute_rf(nu, nx, ntheta, N_exp, PC, date, file)
#
#
#     sigma = [1e-3] * (nx[0] - 1)  # Assumed additive average noise
#     sigma0 = [1e-2] * (nx[0] - 1)
# else:
#     true_theta = [np.log(57.9 * 60. * 10. ** (-2)), 33.3 / 10, np.log(2.7 * 60. * 10. ** (-2)), 35.3 / 10,
#                   np.log(0.865 * 60. * 10. ** (-2)), 38.9 / 10, np.log(1.63 * 60. * 10. ** (-2)), 44.8 / 10]
#     x_meas, u_meas, V, c1o, c2o, dt = utilities.give_data_from_sim(N_exp, PC, date, file, true_theta, info)
#     f, nu, nx, ntheta = utilities.plant_model([])
#     sigma = np.array([2.83037629e-05, 1.74665150e-03, 8.95326671e-05, 1.14905729e-03])*0.1
# #[1e-3] * (nx[0] - 1)  # Assumed additive average noise
#     sigma0 = [1e-2] * (nx[0] - 1)
#     x_meas[:N_exp,:,1] += np.random.multivariate_normal(np.zeros([4]),diag(sigma))
#     for i in range(4):
#         for j in range(N_exp):
#             if x_meas[j, i, 1] <= 0:
#                 x_meas[j, i, 1] = 0
# #x_meas[6,:,:] = x_meas[0,:,:]
# n_points = 1
# s = 0
# # Define Noises #
#
#
#
# xp_0 = [0] * (nx[0] * ntheta[0])
#
#
#
# # -------------------------- Conduct First initial experiments -----------------
# x_0 = [0.2, 0.00, 0.00, 0.00, 0.00]
#
#
# lbx = [0] * nx[0]  # [-0.25, -inf]
# ubx = [np.inf] * nx[0]
#
# lbxp = [-np.inf] * ntheta[0] * nx[0]
# ubxp = [np.inf] * ntheta[0] * nx[0]
#
# lbtheta = [-20.,0.] * 4#(ntheta[0])#[*[-20.] * (ntheta[0]-5), *[0] * 5]
# ubtheta =  [40.] * (ntheta[0])#[*[20.] * (ntheta[0]-5), *[2] * 5]# [60] * ntheta[0]
#
# # ---------------------------------------------
# # ----------- Set values for the inputs -----------
#
# lbu = [u_meas]  # [-1, -1]
# ubu = [u_meas]  # [1, 1]
#
# x_init = np.zeros([N_exp, nx[0]])
# for i in range(nx[0]-1):
#     x_init[:N_exp, i] = x_meas[:N_exp, i, 0]
# x_init[:N_exp, -1] = c2o * u_meas[:N_exp,2]/sum(u_meas[:N_exp,i] for i in range(1,nu[0]))
#
# problem, w0, lbw, ubw, lbg, ubg, trajectories = \
#         ut.construct_NLP_collocation(N_exp, f, x_0, x_init, lbx, ubx, lbu, ubu, lbtheta,
#                               ubtheta, dt,
#                                    n_points, x_meas[:, :, 1:],
#                                   0.8+np.random.rand(ntheta[0]),#
#                                   8, 25) #-----Change x_meas1 to x_meas ------#
#
# solver = ca.nlpsol('solver', 'ipopt', problem)#, {"ipopt.hessian_approximation":"limited-memory"})#, {"ipopt.tol": 1e-10, "ipopt.print_level": 0})#, {"ipopt.hessian_approximation":"limited-memory"})
#
# # Function to get x and u trajectories from w
#
# sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
# w_opt = sol['x'].full().flatten()
#
#
# #bayopt_pe(f, lbtheta, ubtheta, nu, nx, x_meas[:,:,1:], n_points, ntheta[0], u_meas,N_exp, V, c1o, c2o, w_opt[:8])
#
#
# #obj = objective_pe(f, u_meas, x_meas[:,:,1:], N_exp, nx, n_points, nu, w_opt[:8], V, c1o, c2o)
#
#
# x_opt, u_opt, xp_opt, chi2 = trajectories(sol['x'])
# print(chi2)
# x_opt = x_opt.full()# to numpy array
# u_opt = u_opt.full()# to numpy array#
#
#
# f, _, _, _ = utilities.plant_model('sensitivity')
# x_meas1 = np.zeros([N_exp+10, nx[0],n_points+1])
#
# xp_meas = np.zeros((ntheta[0]*nx[0], N_exp * n_points))
# pp = 0
# s = 0
# mle = 0
# x_meas1 = np.zeros([N_exp, nx[0], n_points + 1])
# xmin = np.zeros(nx[0] - 1)
# xmax = np.zeros(nx[0] - 1)  # -1)
# x_meas_norm = x_meas.copy()
# for i in range(nx[0] - 1):
#     xmax[i] = np.max(x_meas[:, i, 1:])
#     if xmax[i] > 1e-9:
#         x_meas_norm[:, i, :] = x_meas[:, i, 1:] / xmax[i]
#     else:
#         x_meas_norm[:, i, :] = x_meas[:, i, 1:]
#         xmax[i] = 1.
# sd = np.array([0.005,0.005,0.002,0.0002])
# chi2 = np.zeros(17)
# for k0 in range(N_exp):
#         x11 = x_init[k0, :]# change it
#         x_meas1[s, :, 0] = np.array(x11.T[:nx[0]])
#         xp1 = np.zeros([nx[0]*ntheta[0], 1])
#         for i in range(n_points):
#             F = utilities.integrator_model(f, nu, nx, ntheta, 'embedded', 'sensitivity', dt[k0, i])
#             Fk = F(x0=ca.vertcat(x11, xp1), p=ca.vertcat(u_meas[k0, :], w_opt[:8]))
#
#             x11 = Fk['xf'][0:nx[0]]
#             xp1 = Fk['xf'][nx[0]:]
#             # + np.random.multivariate_normal([0.] * nx[0], np.diag(np. square(sigma))).T
#             x_meas1[s, :, i+1] = np.array(x11.T)
#             xp_meas[:, pp] = np.array(xp1.T)
#             pp += 1
#         s += 1
#         chi2[k0] = np.sum((x_meas[k0,:,1]-x_meas1[k0,:-1,-1])**2/sd[:]**2)
# vv1 = 0
# rif = np.zeros(N_exp*(n_points))
# for k in range(0,N_exp*(n_points),2):
#     xp_r = ca.reshape(xp_meas[:, k], (nx[0], ntheta[0]))
# #    vv = np.zeros([ntheta[0], ntheta[0], N])
# #    for i in range(0, N):
# #    for i in range(ntheta[0]):
# #        xp_r[:, i] = w_opt[i] * xp_r[:, i]
#     rif[k] = np.trace(xp_r[:-1, :].T@ np.linalg.inv(np.diag(np.square(sigma[:]))) @xp_r[:-1,:])
#     vv1 += (xp_r[:-1, :].T@ np.linalg.inv(np.diag(np.square(sigma[:]))) @xp_r[:-1,:])
# rif = rif/rif.sum()
# vv = np.linalg.inv(vv1)
#
#
# k_exp = N_exp+1
#
#
# thetas = w_opt
# # thetas = np.array([-1.99016019e+00,  4.02029750e-07,  1.13025371e+00,  1.95547106e+01,
# #        -9.07495402e+00,  3.34202879e-02, -7.39214576e+00,  3.99978552e+01])
import pickle
vv1, _, nx, _, nu, thetas, sigma,V, c1o, c2o = pickle.load( open('case_studies/MBDoE/FIM.p', 'rb'))

s = utilities1.objective( f, vv1, 1, nx, 1, nu, thetas, sigma,V, c1o, c2o,'A', u_t.reshape(1, nu[0]))

import functools
funs = functools.partial(utilities.objective, f, vv1, 1, nx, 1, nu, thetas, sigma,V, c1o, c2o)

def construct_obj_MBDoE(select_design):
    vv1, _, nx, _, nu, thetas, sigma, V, c1o, c2o = pickle.load(open('FIM2.p', 'rb'))
    # thetas = np.array([-1.99016019e+00, 4.02029750e-07, 1.13025371e+00, 1.95547106e+01,
    #                    -9.07495402e+00, 3.34202879e-02, -7.39214576e+00, 3.99978552e+01])
    #
    funs = functools.partial(utilities1.objective, f, vv1, 1, nx, 1, nu,
                             true_theta, sigma,V, c1o, c2o, select_design)
    return funs

def construct_obj_MBDoE_moo(select_design, param):
    vv1, _, nx, _, nu, thetas, sigma, V, c1o, c2o = pickle.load(open('FIM2.p', 'rb'))
    # thetas = np.array([-1.99016019e+00, 4.02029750e-07, 1.13025371e+00, 1.95547106e+01,
    #                    -9.07495402e+00, 3.34202879e-02, -7.39214576e+00, 3.99978552e+01])
    #
    funs = functools.partial(utilities1.objective_moo, f, vv1, 1, nx, 1, nu,
                             true_theta, sigma,V, c1o, c2o, select_design, param)
    return funs

def obj_norm(funs, u):
    lbu = [60, 0.3, 0.3, 0.3]  # [60, 0.3, 0.3, 0.3]#, 0.1]
    ubu = [130, 3, 3, 3]  # [130, 3, 3, 3]#, 2.0]

    bounds = (lbu, ubu)
    u_t = (u)*(np.array(ubu)-np.array(lbu))+np.array(lbu)

    u_apply = u_t.reshape(1, nu[0])
    return funs(u_apply)

def obj_MBDoE(select_design='A'):
    funs = construct_obj_MBDoE(select_design)
    return functools.partial(obj_norm, funs)


def MBDoE(x):
    obj_MBDoE(select_design='E')
    return obj_MBDoE(x), [0.]

def obj_MBDoE_moo(select_design='A', param=0.1):
    funs = construct_obj_MBDoE_moo(select_design, param)
    return functools.partial(obj_norm, funs)

def con_MBDoE(u):
    return 0.