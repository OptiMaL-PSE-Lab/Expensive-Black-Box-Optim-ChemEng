from casadi import *
from scipy import stats
import pandas as pd

def plant_model_real(sens):
    """
    Define the model that is meant to describe the physical system
    :return: model f
    """
    nx = 5
    ntheta = 8
    nu = 4
    x = MX.sym('x', nx)
    u = MX.sym('u', nu)
    theta = MX.sym('theta', ntheta)

    x_p = MX.sym('xp', np.shape(x)[0] * np.shape(theta)[0])

    k1 = exp(theta[0] - theta[1] * 1e4 / 8.314 * (1 / (u[0] + 273.15) - 1 / (90 + 273.15)))
    k2 = exp(theta[2] - theta[3] * 1e4 / 8.314 * (1 / (u[0] + 273.15) - 1 / (90 + 273.15)))
    k3 = exp(theta[4] - theta[5] * 1e4 / 8.314 * (1 / (u[0] + 273.15) - 1 / (90 + 273.15)))
    k4 = exp(theta[6] - theta[7] * 1e4 / 8.314 * (1 / (u[0] + 273.15) - 1 / (90 + 273.15)))

    r1 = k1 * x[4] * x[0]
    r2 = k2 * x[4] * x[0]
    r3 = k3 * x[4] * x[1]
    r4 = k4 * x[4] * x[2]

    xdot = vertcat(- r1 - r2, - r3 + r1, - r4 + r2, r3 + r4, - r1 - r2 - r3 - r4) #+\
           #vertcat(u[1]*0.6, 0, 0, 0, u[2]*2.4)/2 - \
           #(u[1]+u[2]+u[3]) * x/2
    # Quadrature
    L = []  # x1 ** 2 + x2 ** 2 + 1*u1 ** 2 + 1*u2**2
    # Algebraic
    alg = []

    # Calculate on the fly dynamic sensitivities without the need of perturbations
    if sens == 'sensitivity':
        xpdot = []
        for i in range(np.shape(theta)[0]):
            xpdot = vertcat(xpdot, jacobian(xdot, x) @ (x_p[nx * i: nx * i + nx])
                            + jacobian(xdot, theta)[nx * i: nx * i + nx])
            f = Function('f', [x, u, theta, x_p], [xdot, L, xpdot],
                         ['x', 'u', 'theta', 'xp'], ['xdot', 'L', 'xpdot'])
    else:
        f = Function('f', [x, u, theta], [xdot, L], ['x', 'u', 'theta'], ['xdot', 'L'])

    nu = u.shape
    nx = x.shape
    ntheta = theta.shape

    return f, nu, nx, ntheta


def plant_model(sens):
    """
    Define the model that is meant to describe the physical system
    :return: model f
    """
    nx = 5
    ntheta = 8
    nu = 4
    x = MX.sym('x', nx)
    u = MX.sym('u', nu)
    theta = MX.sym('theta', ntheta)

    x_p = MX.sym('xp', np.shape(x)[0] * np.shape(theta)[0])

    k1 = exp(theta[0] - theta[1] * 1e4 / 8.314 * (1 / (u[0] + 273.15) - 1 / (90 + 273.15)))
    k2 = exp(theta[2] - theta[3] * 1e4 / 8.314 * (1 / (u[0] + 273.15) - 1 / (90 + 273.15)))
    k3 = exp(theta[4] - theta[5] * 1e4 / 8.314 * (1 / (u[0] + 273.15) - 1 / (90 + 273.15)))
    k4 = exp(theta[6] - theta[7] * 1e4 / 8.314 * (1 / (u[0] + 273.15) - 1 / (90 + 273.15)))

    r1 = k1 * x[4] * x[0]
    r2 = k2 * x[4] * x[0]
    r3 = k3 * x[4] * x[1]
    r4 = k4 * x[4] * x[2]

    xdot = vertcat(- r1 - r2, - r3 + r1, - r4 + r2, r3 + r4, - r1 - r2 - r3 - r4) #+\
           #vertcat(u[1]*0.6, 0, 0, 0, u[2]*2.4)/2 - \
           #(u[1]+u[2]+u[3]) * x/2
    # Quadrature
    L = []  # x1 ** 2 + x2 ** 2 + 1*u1 ** 2 + 1*u2**2
    # Algebraic
    alg = []

    # Calculate on the fly dynamic sensitivities without the need of perturbations
    if sens == 'sensitivity':
        xpdot = []
        for i in range(np.shape(theta)[0]):
            xpdot = vertcat(xpdot, jacobian(xdot, x) @ (x_p[nx * i: nx * i + nx])
                            + jacobian(xdot, theta)[nx * i: nx * i + nx])
            f = Function('f', [x, u, theta, x_p], [xdot, L, xpdot],
                         ['x', 'u', 'theta', 'xp'], ['xdot', 'L', 'xpdot'])
    else:
        f = Function('f', [x, u, theta], [xdot, L], ['x', 'u', 'theta'], ['xdot', 'L'])

    nu = u.shape
    nx = x.shape
    ntheta = theta.shape

    return f, nu, nx, ntheta

def plant_model_GP(GP, GP1, sens):
    """
    Define the model that is meant to describe the physical system
    :return: model f
    """
    nx = 5
    ntheta = 8
    nu = 4
    #x = SX.sym('x', nx)
    u = SX.sym('u', nu)
    theta = SX.sym('theta', ntheta)
    s     = SX.sym('s', ntheta+nu)
    x_p = SX.sym('xp', nx * ntheta)


    mu, vf = GP.derivatives_gp()  # gp_exact_moment([], [], [], [], [*u_t[0, :].T, *theta1.T], s)

    #mu_1 = mu((vertcat(u,theta)))
    #vv = np.zeros([5, 5])
    #for i in range(5):
    #    vv[i, i] = (vf(vertcat(u,theta))[i]
    #                + trace(diag(s) @ (0.5 * hvf[i](vertcat(u,theta)).T
    #                             + Jmu(vertcat(u,theta))[i, :].T @ Jmu(vertcat(u,theta))[i, :])))
    xdot = mu((vertcat(u, theta))) + vertcat(GP1.GP_predictor1(u)[0][0], 0)
    vdot = vf((vertcat(u, theta))) + vertcat(GP1.GP_predictor1(u)[1][0], 0)#vf((vertcat(u,theta)))
    # Quadrature

    # Calculate on the fly dynamic sensitivities without the need of perturbations
    if sens == 'sensitivity':
        xpdot = []
        for i in range(np.shape(theta)[0]):
            xpdot = vertcat(xpdot, jacobian(xdot, theta)[nx * i: nx * i + nx])
            f = Function('f', [u, theta, x_p], [xdot, xpdot, vdot],
                         ['u', 'theta', 'xp'], ['xdot', 'xpdot', 'vdot'])
    else:
        f = Function('f', [u, theta], [xdot], ['u', 'theta'], ['xdot'])

    nu = u.shape
    ntheta = theta.shape

    return f, nu, nx, ntheta


def plant_model_GP_discripancy(GP1, sens):
    """
    Define the model that is meant to describe the physical system
    :return: model f
    """
    nx = 5
    ntheta = 8
    nu = 4
    #x = SX.sym('x', nx)
    u = SX.sym('u', nu)
    theta = SX.sym('theta', ntheta)
    s     = SX.sym('s', ntheta+nu)
    x_p = SX.sym('xp', nx * ntheta)


    #mu, vf = GP.derivatives_gp()  # gp_exact_moment([], [], [], [], [*u_t[0, :].T, *theta1.T], s)

    #mu_1 = mu((vertcat(u,theta)))
    #vv = np.zeros([5, 5])
    #for i in range(5):
    #    vv[i, i] = (vf(vertcat(u,theta))[i]
    #                + trace(diag(s) @ (0.5 * hvf[i](vertcat(u,theta)).T
    #                             + Jmu(vertcat(u,theta))[i, :].T @ Jmu(vertcat(u,theta))[i, :])))
    xdot = vertcat(GP1.GP_predictor1(u)[0][0], 0)
    vdot = vertcat(GP1.GP_predictor1(u)[1][0], 0)#vf((vertcat(u,theta)))
    # Quadrature

    # Calculate on the fly dynamic sensitivities without the need of perturbations
    if sens == 'sensitivity':
        xpdot = []
        for i in range(np.shape(theta)[0]):
            xpdot = vertcat(xpdot, jacobian(xdot, theta)[nx * i: nx * i + nx])
            f = Function('f', [u, theta, x_p], [xdot, xpdot, vdot],
                         ['u', 'theta', 'xp'], ['xdot', 'xpdot', 'vdot'])
    else:
        f = Function('f', [u, theta], [xdot], ['u', 'theta'], ['xdot'])

    nu = u.shape
    ntheta = theta.shape

    return f, nu, nx, ntheta



def integrator_model(f, nu, nx, ntheta, s1, s2, dt):
    """
    This function constructs the integrator to be suitable with casadi environment, for the equations of the model
    and the objective function with variable time step.
     inputs: model, sizes
     outputs: F: Function([x, u, dt]--> [xf, obj])
    """
    M = 4  # RK4 steps per interval
    DT = dt#.sym('DT')
    DT1 = DT / M
    X0 = SX.sym('X0', nx)
    U = SX.sym('U', nu)
    theta = SX.sym('theta', ntheta)
    xp0 = SX.sym('xp', np.shape(X0)[0] * np.shape(theta)[0])
    X = X0
    Q = 0
    G = 0
    S = xp0
    if s1 == 'embedded':
        if s2 == 'sensitivity':
            xdot, qj, xpdot = f(X, U, theta, xp0)
            dae = {'x': vertcat(X, xp0), 'p': vertcat(U, theta), 'ode': vertcat(xdot, xpdot)}
            opts = {'tf': dt}  # interval length
            F = integrator('F', 'cvodes', dae, opts)
        elif s2 == 'identify':
            xdot, qj, xpdot = f(X, U, theta, xp0)
            dae = {'x': vertcat(X, xp0), 'p': vertcat(U, theta), 'ode': vertcat(xdot, xpdot)}
            opts = {'tf': dt}  # interval length
            F = integrator('F', 'cvodes', dae, opts)
        else:
            xdot, qj = f(X, U, theta)
            dae = {'x': vertcat(X), 'p': vertcat(U, theta), 'ode': vertcat(xdot)}
            opts = {'tf': dt}  # interval length
            F = integrator('F', 'cvodes', dae, opts)
    else:
        if s2 == 'sensitivity':

            for j in range(M):
                k1, k1_a, k1_p = f(X, U, theta, S)
                k2, k2_a, k2_p = f(X + DT1 / 2 * k1, U, theta, S + DT1 / 2 * k1_p)
                k3, k3_a, k3_p = f(X + DT1 / 2 * k2, U, theta, S + DT1 / 2 * k2_p)
                k4, k4_a, k4_p = f(X + DT1 * k3, U, theta, S + DT1 * k3_p)
                X = X + DT1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                G = G + DT1 / 6 * (k1_a + 2 * k2_a + 2 * k3_a + k4_a)
                S = S + DT1 / 6 * (k1_p + 2 * k2_p + 2 * k3_p + k4_p)
            F = Function('F', [X0, U, theta, xp0], [X, G, S], ['x0', 'p', 'theta', 'xp0'], ['xf', 'g', 'xp'])
        else:
            for j in range(M):
                k1,_ = f(X, U, theta)
                k2,_ = f(X + DT1 / 2 * k1, U, theta)
                k3,_ = f(X + DT1 / 2 * k2, U, theta)
                k4,_ = f(X + DT1 * k3, U, theta)
                X = X + DT1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            F = Function('F', [X0, vertcat(U, theta)], [X], ['x0', 'p'], ['xf'])
    return F


def jacobian_f(f):

    """
    This function aims to compute the Jacobian of any given MX function
    input: MX function
    output: Jacobian
    """

    F_x = f.jacobian()
    return F_x


def maximum_likelihood_est(i, y, y_meas, sigma, k, ymax):
    """
    This is a function that computes the MLE for a given set of experiments
    """
#    N = 100#y_meas.shape[0]
    M = y_meas.shape[1]

    MLE = 0
    s = 0
    for j in range(M):
         MLE += 0.5*(y[j]/ymax[j] - y_meas[i][j][k]) **2 /sigma[j]**2
    return MLE


def construct_polynomials_basis(d, poly_type):

    # Get collocation points
    tau_root = np.append(0, collocation_points(d, poly_type))

    # Coefficients of the collocation equation
    C = np.zeros((d + 1, d + 1))

    # Coefficients of the continuity equation
    D = np.zeros(d + 1)

    # Coefficients of the quadrature function
    B = np.zeros(d + 1)

    # Construct polynomial basis
    for j in range(d + 1):
        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        p = np.poly1d([1])
        for r in range(d + 1):
            if r != j:
                p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j] - tau_root[r])

        # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
        D[j] = p(1.0)
        # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity
        # equation
        pder = np.polyder(p)
        for r in range(d + 1):
            C[j, r] = pder(tau_root[r])

        # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
        pint = np.polyint(p)
        B[j] = pint(1.0)

    return C, D, B


def fim_for_single_t(xpdot1, Vold, sigma, nx, ntheta,A):
    """

    :param xpdot:
    :param sigma:
    :return: FIM1
    """
    xpdot = xpdot1
    FIM1 = Vold#np.zeros([2, 2])
    for i in range(1):
        xp_r = reshape(xpdot, (nx, ntheta))
#    vv = np.zeros([ntheta[0], ntheta[0], 1 + N])
#    for i in range(0, 1 + N):
#        FIM1 += xp_r[:-1,:].T @  np.linalg.inv(np.diag(np.square(sigma[:])))  @ xp_r[:-1,:]#@ A# + np.linalg.inv(np.array([[0.01, 0, 0, 0], [0, 0.05, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0.2]]))
        FIM1 += xp_r[:-1,:].T @  inv(diag((sigma[:])))  @ xp_r[:-1,:]#@ A# + np.linalg.inv(np.array([[0.01, 0, 0, 0], [0, 0.05, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0.2]]))

#        FIM1 += xp_r.T @ inv(diag(sigma**2)) @ xp_r# + np.linalg.inv(np.array([[0.01, 0, 0, 0], [0, 0.05, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0.2]]))

#    FIM  = solve(FIM1, SX.eye(FIM1.size1()))
#   [Q, R] = qr(FIM1.expand())

    return FIM1# + 0.0001)


def collocation(f, d, s, nx, nu, lbx, ubx, lbw, ubw, w0, w,
                lbg, ubg, g, x_meas, Xk, k_exp, m, Uk, thetak, h, C, D):
    Xc = []

    for j in range(d):
        Xkj = MX.sym('X_' + str(s) + '_' + str(j), nx)
        Xc += [Xkj]
        w += [Xkj]
        lbw.extend(lbx)
        ubw.extend(ubx)
        #                ubw.extend([u_meas[k_exp][1]])
        w0.extend(x_meas[k_exp, :])#, m])
    #                w0.extend([u_meas[k_exp][1]])

    # Loop over collocation points
    Xk_end = D[0] * Xk

    for j in range(1, d + 1):
        # Expression for the state derivative at the collocation point
        xp = C[0, j] * Xk

        for r in range(d):
            xp = xp + C[r + 1, j] * Xc[r]

        # Append collocation equations
        fj, qj = f(Xc[j - 1], Uk, thetak)  # Xpc[j - 1])
        g += [(h * fj - xp)]
        lbg.extend([-1e-9] * nx)
        ubg.extend([1e-9] * nx)

        # Add contribution to the end state
        Xk_end = Xk_end + D[j] * Xc[j - 1]

    # New NLP variable for state at end of interval
    Xk = MX.sym('X_' + str(s + 1), nx)
    w += [Xk]
    lbw.extend(lbx)
    ubw.extend(ubx)  # [:-1])
    #            ubw.extend([u_meas[k_exp][1]])
    w0.extend(x_meas[k_exp, :])#, m])
    #            w0.extend([u_meas[k_exp][1]])

    # Add equality constraint
    g += [Xk_end - Xk]
    lbg.extend([-1e-9] * nx)
    ubg.extend([1e-9] * nx)
    return lbw, ubw, w0, w, lbg, ubg, g, Xk


def chisquare_test(chisquare_value, conf_level, dof):
    ref_chisquare = stats.chi2.ppf((conf_level), dof)
    p_value = 1 - stats.chi2.cdf(chisquare_value, dof)
    return ref_chisquare, chisquare_value


def objective_moo(f, V_old, N_exp, nx, n_points, nu, theta, sigma, V, c1o, c2o, select_design, param,u):

    ntheta = len(theta)
    x_meas1 = np.zeros([N_exp + 10, nx[0], n_points + 1])

    xp_meas = np.zeros((ntheta * nx[0], N_exp * n_points))
    dt      = np.zeros([N_exp,n_points])
    pp = 0
    s = 0
    x_init = np.zeros([N_exp,nx[0]])
    for i in range(nx[0] - 1):
        x_init[:N_exp, 0] = c1o * u[:N_exp, 1] / sum(u[:N_exp, i] for i in range(1, nu[0]))
    x_init[:N_exp, -1] = c2o * u[:N_exp, 2] / sum(u[:N_exp, i] for i in range(1, nu[0]))

    for k0 in range(N_exp):
        x11 = x_init[k0, :]  # change it
        x_meas1[s, :, 0] = np.array(x11.T[:nx[0]])
        xp1 = np.zeros([nx[0] * ntheta, 1])
        dt[k0, :] = V / np.sum(u[k0, 1:])/n_points
        for i in range(n_points):
            F = integrator_model(f, nu, nx, ntheta, 'embedded', 'sensitivity', dt[k0, i])
            Fk = F(x0=vertcat(x11, xp1), p=vertcat(u[k0, :], theta[:8]))

            x11 = Fk['xf'][0:nx[0]]
            xp1 = Fk['xf'][nx[0]:]
            # + np.random.multivariate_normal([0.] * nx[0], np.diag(np. square(sigma))).T
            x_meas1[s, :, i + 1] = np.array(x11.T)
            xp_meas[:, pp] = np.array(xp1.T)
            pp += 1
        s += 1

    vv1 = V_old
    for k in range(N_exp * (n_points)):
        xp_r = reshape(xp_meas[:, k], (nx[0], ntheta))
       #    vv = np.zeros([ntheta[0], ntheta[0], N])
       #    for i in range(0, N):
       #    for i in range(ntheta[0]):
       #        xp_r[:, i] = w_opt[i] * xp_r[:, i]
        vv1 += (xp_r[:-1, :].T @ np.linalg.inv(np.diag(np.square(sigma[:]))) @ xp_r[:-1, :])
        vv = np.linalg.inv(vv1)
    obj =-designs(vv1, select_design, param)#np.min(np.linalg.eig(vv1)[0])#np.log(np.linalg.det(vv1)+0.0001)#-## np.linalg.eig(vv)[0][0]#

    return obj

def objective(f, V_old, N_exp, nx, n_points, nu, theta, sigma, V, c1o, c2o, select_design,u):

    ntheta = len(theta)
    x_meas1 = np.zeros([N_exp + 10, nx[0], n_points + 1])

    xp_meas = np.zeros((ntheta * nx[0], N_exp * n_points))
    dt      = np.zeros([N_exp,n_points])
    pp = 0
    s = 0
    x_init = np.zeros([N_exp,nx[0]])
    for i in range(nx[0] - 1):
        x_init[:N_exp, 0] = c1o * u[:N_exp, 1] / sum(u[:N_exp, i] for i in range(1, nu[0]))
    x_init[:N_exp, -1] = c2o * u[:N_exp, 2] / sum(u[:N_exp, i] for i in range(1, nu[0]))

    for k0 in range(N_exp):
        x11 = x_init[k0, :]  # change it
        x_meas1[s, :, 0] = np.array(x11.T[:nx[0]])
        xp1 = np.zeros([nx[0] * ntheta, 1])
        dt[k0, :] = V / np.sum(u[k0, 1:])/n_points
        for i in range(n_points):
            F = integrator_model(f, nu, nx, ntheta, 'embedded', 'sensitivity', dt[k0, i])
            Fk = F(x0=vertcat(x11, xp1), p=vertcat(u[k0, :], theta[:8]))

            x11 = Fk['xf'][0:nx[0]]
            xp1 = Fk['xf'][nx[0]:]
            # + np.random.multivariate_normal([0.] * nx[0], np.diag(np. square(sigma))).T
            x_meas1[s, :, i + 1] = np.array(x11.T)
            xp_meas[:, pp] = np.array(xp1.T)
            pp += 1
        s += 1

    vv1 = V_old
    for k in range(N_exp * (n_points)):
        xp_r = reshape(xp_meas[:, k], (nx[0], ntheta))
       #    vv = np.zeros([ntheta[0], ntheta[0], N])
       #    for i in range(0, N):
       #    for i in range(ntheta[0]):
       #        xp_r[:, i] = w_opt[i] * xp_r[:, i]
        vv1 += (xp_r[:-1, :].T @ np.linalg.inv(np.diag(np.square(sigma[:]))) @ xp_r[:-1, :])
        vv = np.linalg.inv(vv1)
    obj =-designs(vv1, select_design)#np.min(np.linalg.eig(vv1)[0])#np.log(np.linalg.det(vv1)+0.0001)#-## np.linalg.eig(vv)[0][0]#

    return obj

def designs(x, select_design='A', param=0.1):
    if select_design=='A':
        return np.trace(x)
    elif select_design=='D':
        return np.log(np.linalg.det(x)+0.0001)
    elif select_design=='E':
        return np.min(np.linalg.eig(x)[0])
    elif select_design=='E1':
        return np.sort(np.linalg.eig(x)[0])[1]
    elif select_design=='MOO':
        return (1-param)*np.min(np.linalg.eig(x)[0])+ param*np.trace(x)/10000
    elif select_design=='MOO1':
        return (1-param)*np.log(np.linalg.det(x)+0.0001)+ param*np.trace(x)/10000
    elif select_design=='MOO2':
        return (1-param)*np.min(np.linalg.eig(x)[0])+ param*np.log(np.linalg.det(x)+0.0001)
    elif select_design=='MOO3':
        return (1-param)*np.min(np.linalg.eig(x)[0])+ param*np.sort(np.linalg.eig(x)[0])[1]


def objective_pe_mcmc(theta, kwargs):#, ):
    f, u, x_meas, N_exp, nx, n_points, nu, V, c1o, c2o, theta2 = kwargs#['f'], kwargs['u_meas'],\
                                                         #kwargs['x_meas'], kwargs['N_exp'],\
                                                         #kwargs['nx'], kwargs['n_points'],\
                                                         #kwargs['nu'], kwargs['V'],\
                                                         #kwargs['c1o'], kwargs['c2o']

    ntheta = len(theta)
    x_meas1 = np.zeros([N_exp, nx[0], n_points + 1])
    xmin = np.zeros(nx[0]-1)
    xmax = np.zeros(nx[0]-1)#-1)
    x_meas_norm = x_meas.copy()
    for i in range(nx[0]-1):
        xmax[i] = np.max(x_meas[:, i, :])
        if xmax[i] > 1e-9:
            x_meas_norm[:, i, :] = x_meas[:, i, :]/xmax[i]
        else:
            x_meas_norm[:, i, :] = x_meas[:, i, :]
            xmax[i] = 1.

    dt      = np.zeros([N_exp, n_points])
    pp = 0
    s = 0
    x_init = np.zeros([N_exp,nx[0]])
    for i in range(nx[0] - 1):
        x_init[:N_exp, 0] = c1o * u[:N_exp, 1] / sum(u[:N_exp, i] for i in range(1, nu[0]))
    x_init[:N_exp, -1] = c2o * u[:N_exp, 2] / sum(u[:N_exp, i] for i in range(1, nu[0]))
    mle = 0
    for k0 in range(N_exp):
        x11 = x_init[k0, :]  # change it
        x_meas1[s, :, 0] = np.array(x11.T[:nx[0]])
        dt[k0, :] = V / np.sum(u[k0, 1:])/n_points
        for i in range(n_points):
            F = integrator_model(f, nu, nx, ntheta+6, 'embedded', 'mope', dt[k0, i])
            Fk = F(x0=vertcat(x11), p=vertcat(u[k0, :], [*theta[:2], *theta2]))

            x11 = Fk['xf'][0:nx[0]]
            # + np.random.multivariate_normal([0.] * nx[0], np.diag(np. square(sigma))).T
            x_meas1[s, :, i + 1] = np.array(x11.T)
            pp += 1
            mle += maximum_likelihood_est(s, x_meas1[s,:-1,i+1] , x_meas_norm, [1, 1, 1, 1], i, xmax)

        s += 1


    obj = -mle#np.linalg.eig(vv1)[0][0]#

    return obj




def objective_pe(f, u, x_meas, N_exp, nx, n_points, nu, theta, V, c1o, c2o):

    ntheta = len(theta)
    x_meas1 = np.zeros([N_exp, nx[0], n_points + 1])
    xmin = np.zeros(nx[0]-1)
    xmax = np.zeros(nx[0]-1)#-1)
    x_meas_norm = x_meas.copy()
    for i in range(nx[0]-1):
        xmax[i] = np.max(x_meas[:, i, :])
        if xmax[i] > 1e-9:
            x_meas_norm[:, i, :] = x_meas[:, i, :]/xmax[i]
        else:
            x_meas_norm[:, i, :] = x_meas[:, i, :]
            xmax[i] = 1.

    dt      = np.zeros([N_exp, n_points])
    pp = 0
    s = 0
    x_init = np.zeros([N_exp,nx[0]])
    for i in range(nx[0] - 1):
        x_init[:N_exp, 0] = c1o * u[:N_exp, 1] / sum(u[:N_exp, i] for i in range(1, nu[0]))
    x_init[:N_exp, -1] = c2o * u[:N_exp, 2] / sum(u[:N_exp, i] for i in range(1, nu[0]))
    mle = 0
    for k0 in range(N_exp):
        x11 = x_init[k0, :]  # change it
        x_meas1[s, :, 0] = np.array(x11.T[:nx[0]])
        dt[k0, :] = V / np.sum(u[k0, 1:])/n_points
        for i in range(n_points):
            F = integrator_model(f, nu, nx, ntheta, 'embedded', 'mope', dt[k0, i])
            Fk = F(x0=vertcat(x11), p=vertcat(u[k0, :], theta[:8]))

            x11 = Fk['xf'][0:nx[0]]
            # + np.random.multivariate_normal([0.] * nx[0], np.diag(np. square(sigma))).T
            x_meas1[s, :, i + 1] = np.array(x11.T)
            pp += 1
            mle += maximum_likelihood_est(s, x_meas1[s,:-1,i+1] , x_meas_norm, [1, 1, 1, 1], i, xmax)

        s += 1


    obj = mle#np.linalg.eig(vv1)[0][0]#

    return obj



def give_data_from_exp(nu, nx, ntheta, N_exp, PC, date, file, info,lab):
    for i in range(1, N_exp + 1):
        file[i - 1] = '/Users/' + PC + '/Dropbox/UCL/' + date + '/Peaks and Concentrations epoch_' + str(
        i) + '_Leeds_' + lab + '_nLabBots_2.csv' # '/output_concentrations_'+str(i)+'.csv'


    size = np.shape(np.array(pd.read_csv(file[0])))
    xl = np.zeros([N_exp, size[0] - 1, 1])  # size[1]])

    for i in range(N_exp):
        xl[i, :, :] = np.array(pd.read_csv(file[i])['Concentration (mole/L)'])[1:].reshape(4, 1)
        for j in range(size[0] - 1):
            for k in range(1):
                if xl[i, j, k] < 0:
                    xl[i, j, k] = 0.

    for i in range(1, N_exp + 1):
        if i >= 10:
            file[i - 1] = '/Users/' + PC + '/Dropbox/UCL/' + date + '/Requests_' + str(i) + '.csv'
        else:
            file[i - 1] = '/Users/' + PC + '/Dropbox/UCL/' + date + '/Requests_0' + str(i) + '.csv'

    size = np.shape(np.array(pd.read_csv(file[0])))
    ul = np.zeros([N_exp, size[0], size[1]])
    for i in range(N_exp):
        ul[i, :] = np.array(pd.read_csv(file[i]))

    n_points = 1
    n = 1
    """
    change it 
    """
    x_meas = np.zeros((N_exp + 30, nx[0] - 1, n_points + 1))
    u_meas = np.zeros((N_exp + 30, nu[0]))

    # -------------- Change the concentrations --------------#
    # u[0] -----> T
    #
    """
    u[0] ---> T   #REVISIT
    u[1] ---> F1
    u[2] ---> F2
    u[3] ---> F3

    x[0] ---> c1
    x[1] ---> c3
    x[2] ---> c4
    x[3] ---> c5
    x[4] ---> c2 --- NOT
    """

    # ------------------------------------------------------- #
    dt = np.zeros([N_exp + 30, n_points])

    """""
    for i in range(N_exp):
        x_meas[i, 0, :] = xl[i, 0:n * n_points + 1:n, 1].T
        x_meas[i, 1:nx[0]-1, :] = xl[i, 0:n*n_points + 1:n, 3:(nx[0]-1)+2].T
        #x_meas[i, -1, :] = xl[i, 0:n*n_points + 1:n, 2].T
    """""

    setup = '/Users/' + PC + '/Dropbox/UCL/' + date + info#'/Exp_Setup_Info_06-September-2019_11_34_19.csv'
    setup1 = np.array(pd.read_csv(setup)['C(mole/L)'])

    c1o = setup1[1]  # 2.03
    c2o = setup1[0]  # 4.17
    volume_file = '/Users/' + PC + '/Dropbox/UCL/' + date +'/Reactor Volume, (mL) Leeds_'+lab+'.txt'
    V = np.array(pd.read_csv(volume_file, header = None))[0,0]  # 2.7

    for i in range(N_exp):
        x_meas[i, 0, n_points] = xl[i, 0]
        x_meas[i, 1, n_points] = xl[i, 3]
        x_meas[i, 2, n_points] = xl[i, 2]
        x_meas[i, 3, n_points] = xl[i, 1]

        u_meas[i, 1] = ul[i][0][1]
        u_meas[i, 2] = ul[i][0][0]
        u_meas[i, 3] = ul[i][0][2]

        u_meas[i, 0] = ul[i][0][-1]
        x_meas[i, 0, 0] = c1o * u_meas[i, 1] / sum(u_meas[i, j] for j in range(1, nu[0]))
        x_meas[i, 1, 0] = 0.
        x_meas[i, 2, 0] = 0.
        x_meas[i, 3, 0] = 0.
        dt[i, :] = V / sum(u_meas[i, 1:])  # xl[i, n:n*n_points + 1:n, 0].T - xl[i, 0:(n)*n_points :n, 0].T
    return x_meas, u_meas, V, c1o, c2o, dt


def give_data_from_exp_recal(nu, nx, ntheta, N_exp, PC, date, file, labot, info):
    for i in range(1, N_exp + 1):
        file[i - 1] = '/Users/' + PC + '/Dropbox/UCL/' + date + '/Peaks and Concentrations_' + str(
            i) + '.csv'  # '/output_concentrations_'+str(i)+'.csv'

    size = np.shape(np.array(pd.read_csv(file[0])))
    xl = np.zeros([N_exp, size[0] - 1, 1])  # size[1]])

    for i in range(N_exp):
        xl[i, :, :] = np.array(pd.read_csv(file[i])['Area'])[1:].reshape(4, 1)
        for j in range(size[0] - 1):
            for k in range(1):
                if xl[i, j, k] < 0:
                    xl[i, j, k] = 0.

    for i in range(N_exp):
        if labot == 1:
            a1 = 0.4078
            a2 = 0.7505
            a3 = 0.1939
            a4 = 0.5987
        else:
            a1 = 0.4117
            a2 = 0.7898
            a3 = 0.1967
            a4 = 0.6123
    c1 = np.zeros(N_exp)
    for i in range(N_exp):
       c1[i] = np.array(pd.read_csv(file[i])['Area'])[0]
    cr = 0.101

    for i in range(1, N_exp + 1):
        if i >= 10:
            file[i - 1] = '/Users/' + PC + '/Dropbox/UCL/' + date + '/Requests_' + str(i) + '.csv'
        else:
            file[i - 1] = '/Users/' + PC + '/Dropbox/UCL/' + date + '/Requests_0' + str(i) + '.csv'

    size = np.shape(np.array(pd.read_csv(file[0])))
    ul = np.zeros([N_exp, size[0], size[1]])
    for i in range(N_exp):
        ul[i, :] = np.array(pd.read_csv(file[i]))

    n_points = 1
    n = 1
    """
    change it 
    """
    x_meas = np.zeros((N_exp + 30, nx[0] - 1, n_points + 1))
    u_meas = np.zeros((N_exp + 30, nu[0]))

    # -------------- Change the concentrations --------------#
    # u[0] -----> T
    #
    """
    u[0] ---> T
    u[1] ---> F1
    u[2] ---> F2
    u[3] ---> F3

    x[0] ---> c1
    x[1] ---> c3
    x[2] ---> c4
    x[3] ---> c5
    x[4] ---> c2 --- NOT
    """

    # ------------------------------------------------------- #
    dt = np.zeros([N_exp + 30, n_points])

    """""
    for i in range(N_exp):
        x_meas[i, 0, :] = xl[i, 0:n * n_points + 1:n, 1].T
        x_meas[i, 1:nx[0]-1, :] = xl[i, 0:n*n_points + 1:n, 3:(nx[0]-1)+2].T
        #x_meas[i, -1, :] = xl[i, 0:n*n_points + 1:n, 2].T
    """""

    setup = '/Users/' + PC + '/Dropbox/UCL/' + date + info#'/Exp_Setup_Info_06-September-2019_11_34_19.csv'
    setup1 = np.array(pd.read_csv(setup))[0]

    c1o = setup1[2]  # 2.03
    c2o = setup1[1]  # 4.17
    V = setup1[0]  # 2.7



    for i in range(N_exp):
        u_meas[i, 1] = ul[i][0][1]
        u_meas[i, 2] = ul[i][0][0]
        u_meas[i, 3] = ul[i][0][2]

        u_meas[i, 0] = ul[i][0][-1]
        crr = cr * u_meas[i, 1] / sum(u_meas[i, j] for j in range(1, nu[0]))

        x_meas[i, 0, n_points] = 1/a1 * crr/c1[i] * xl[i, 0]
        x_meas[i, 1, n_points] = 1/a2 * crr/c1[i] * xl[i, 3]
        x_meas[i, 2, n_points] = 1/a3 * crr/c1[i] * xl[i, 2]
        x_meas[i, 3, n_points] = 1/a4 * crr/c1[i] * xl[i, 1]


        x_meas[i, 0, 0] = c1o * u_meas[i, 1] / sum(u_meas[i, j] for j in range(1, nu[0]))
        x_meas[i, 1, 0] = 0.
        x_meas[i, 2, 0] = 0.
        x_meas[i, 3, 0] = 0.
        dt[i, :] = V / sum(u_meas[i, 1:])  # xl[i, n:n*n_points + 1:n, 0].T - xl[i, 0:(n)*n_points :n, 0].T
    return x_meas, u_meas, V, c1o, c2o, dt


def give_data_from_sim(N_exp, PC, date, file, true_theta, info):

    for i in range(1, N_exp + 1):
        file[i - 1] = '/Users/' + PC + '/Dropbox/UCL/' + date + '/Peaks and Concentrations_' + str(
            i) + '.csv'  # '/output_concentrations_'+str(i)+'.csv'

    size = np.shape(np.array(pd.read_csv(file[0])))
    xl = np.zeros([N_exp, size[0] - 1, 1])  # size[1]])

    for i in range(N_exp):
        xl[i, :, :] = np.array(pd.read_csv(file[i])['Concentration (mol/L)'])[1:].reshape(4, 1)
        for j in range(size[0] - 1):
            for k in range(1):
                if xl[i, j, k] < 0:
                    xl[i, j, k] = 0.

    for i in range(1, N_exp + 1):
        if i >= 10:
            file[i - 1] = '/Users/' + PC + '/Dropbox/UCL/' + date + '/Requests_' + str(i) + '.csv'
        else:
            file[i - 1] = '/Users/' + PC + '/Dropbox/UCL/' + date + '/Requests_0' + str(i) + '.csv'

    size = np.shape(np.array(pd.read_csv(file[0])))
    ul = np.zeros([N_exp, size[0], size[1]])
    for i in range(N_exp):
        ul[i, :] = np.array(pd.read_csv(file[i]))

    n_points = 1
    n = 1

    f, nu, nx, ntheta = plant_model_real([])
    """
    change it 
    """
    x_meas = np.zeros((N_exp + 30, nx[0] - 1, n_points + 1))
    u_meas = np.zeros((N_exp + 30, nu[0]))

    # -------------- Change the concentrations --------------#
    # u[0] -----> T
    #
    """
    u[0] ---> T
    u[1] ---> F1
    u[2] ---> F2
    u[3] ---> F3

    x[0] ---> c1
    x[1] ---> c3
    x[2] ---> c4
    x[3] ---> c5
    x[4] ---> c2 --- NOT
    """

    # ------------------------------------------------------- #
    dt = np.zeros([N_exp + 32, n_points])

    """""
    for i in range(N_exp):
        x_meas[i, 0, :] = xl[i, 0:n * n_points + 1:n, 1].T
        x_meas[i, 1:nx[0]-1, :] = xl[i, 0:n*n_points + 1:n, 3:(nx[0]-1)+2].T
        #x_meas[i, -1, :] = xl[i, 0:n*n_points + 1:n, 2].T
    """""

    setup = '/Users/' + PC + '/Dropbox/UCL/' + date + info#'/Exp_Setup_Info_06-September-2019_11_34_19.csv'
    setup1 = np.array(pd.read_csv(setup))[0]

    c1o = setup1[2]  # 2.03
    c2o = setup1[1]  # 4.17
    V = setup1[0]  # 2.7

    for i in range(N_exp):

        u_meas[i, 1] = ul[i][0][1]
        u_meas[i, 2] = ul[i][0][0]
        u_meas[i, 3] = ul[i][0][2]

        u_meas[i, 0] = ul[i][0][-1]
        x_meas[i, 0, 0] = c1o * u_meas[i, 1] / sum(u_meas[i, j] for j in range(1, nu[0]))
        x_meas[i, 1, 0] = 0.
        x_meas[i, 2, 0] = 0.
        x_meas[i, 3, 0] = 0.
        dt[i, :] = V / sum(u_meas[i, 1:])  # xl[i, n:n*n_points + 1:n, 0].T - xl[i, 0:(n)*n_points :n, 0].T

    x_init = np.zeros([N_exp, nx[0]])
    for i in range(nx[0] - 1):
        x_init[:N_exp, i] = x_meas[:N_exp, i, 0]
    x_init[:N_exp, -1] = c2o * u_meas[:N_exp, 2] / sum(u_meas[:N_exp, i] for i in range(1, nu[0]))

    pp = 0
    s = 0
    for k0 in range(N_exp):
        x11 = x_init[k0, :]  # change it
        for i in range(n_points):
            F = integrator_model(f, nu, nx, ntheta, 'embedded', 'nope', dt[k0, i])
            Fk = F(x0=vertcat(x11), p=vertcat(u_meas[k0, :],true_theta))

            x11 = Fk['xf'][0:nx[0]]

            # + np.random.multivariate_normal([0.] * nx[0], np.diag(np. square(sigma))).T
            x_meas[s, :, i + 1] = np.array(x11[0:nx[0]-1].T)

        s += 1
    return x_meas, u_meas, V, c1o, c2o, dt

def give_data_from_sim_update(k_exp, x_meas, u_opt, dt, true_theta, c1o, c2o, V):
    f, nu, nx, ntheta = plant_model_real([])
    x_meas[k_exp, 0, 0] = c1o * u_opt[1] / sum(u_opt[ 1:])  # u_opt[1]/sum(u_opt[1:])
    x_meas[k_exp, 1, 0] = 0.
    x_meas[k_exp, 2, 0] = 0.
    x_meas[k_exp, 3, 0] = 0.
    x_init = np.zeros([1, x_meas.shape[1]+1])
    for i in range(nx[0]-1):
        x_init[0, i] = x_meas[k_exp, i, 0]
    x_init[0, -1] = c2o * u_opt[2] / sum(u_opt[i] for i in range(1,u_opt.shape[0]))
    x11 = x_init[0, :]
    dt[k_exp, :] = V / sum(u_opt[1:])  # sum(u_opt[1:])#xl[0, n:n * n_points + 1:n, 0].T - xl[0, 0: n * n_points :n, 0].T
    for i in range(1):
        F = integrator_model(f, nu, nx, ntheta, 'embedded', 'no', dt[k_exp, i])
        Fk = F(x0=vertcat(x11), p=vertcat(u_opt, true_theta))

        x11 = Fk['xf'][0:nx[0]]
    # + np.random.multivariate_normal([0.] * nx[0], np.diag(np. square(sigma))).T
        x_meas[k_exp,:, i+1] = np.array(x11[:-1].T)
    return x_meas, dt


def give_data_from_sim_update1(k_exp, u_opt, dt, true_theta, c1o, c2o, V):
    f, nu, nx, ntheta = plant_model_real([])
    x_meas = np.zeros([5,2])
    x_meas[0, 0] = c1o * u_opt[1] / sum(u_opt[ 1:])  # u_opt[1]/sum(u_opt[1:])
    x_meas[1, 0] = 0.
    x_meas[2, 0] = 0.
    x_meas[3, 0] = 0.
    x_init = np.zeros([1, nx[0]])
    for i in range(nx[0]-1):
        x_init[0, i] = x_meas[i, 0]
    x_init[0, -1] = c2o * u_opt[2] / sum(u_opt[i] for i in range(1,u_opt.shape[0]))
    x11 = x_init[0, :]
    dt[k_exp, :] = V / sum(u_opt[1:])  # sum(u_opt[1:])#xl[0, n:n * n_points + 1:n, 0].T - xl[0, 0: n * n_points :n, 0].T
    for i in range(1):
        F = integrator_model(f, nu, nx, ntheta, 'embedded', 'no', dt[k_exp, i])
        Fk = F(x0=vertcat(x11), p=vertcat(u_opt, true_theta))

        x11 = Fk['xf'][0:nx[0]]
    # + np.random.multivariate_normal([0.] * nx[0], np.diag(np. square(sigma))).T
        x_meas[:, i+1] = np.array(x11.T)
    return x_meas, dt


def compute_rf(nu, nx, ntheta, N_exp, PC, date, file):
    for i in range(1, N_exp + 1):
        file[i - 1] = '/Users/' + PC + '/Dropbox/UCL/' + date + '/Peaks and Concentrations_' + str(
            i) + '.csv'  # '/output_concentrations_'+str(i)+'.csv'

    size = np.shape(np.array(pd.read_csv(file[0])))
    xl = np.zeros([N_exp, size[0] - 1, 1])  # size[1]])

    for i in range(N_exp):
        xl[i, :, :] = np.array(pd.read_csv(file[i])['Concentration (mol/L)'])[1:].reshape(4, 1)
        for j in range(size[0] - 1):
            for k in range(1):
                if xl[i, j, k] < 0:
                    xl[i, j, k] = 0.

    x_is = np.zeros([N_exp, 1, 1])  # size[1]])

    for i in range(N_exp):
        x_is[i, :, :] = np.array(pd.read_csv(file[i])['Concentration (mol/L)'])[0]
        for j in range(1):
            for k in range(1):
                if x_is[i, j, k] < 0:
                    x_is[i, j, k] = 0.

    xa = np.zeros([N_exp, size[0], 1])  # size[1]])

    for i in range(N_exp):
        xa[i, :, :] = np.array(pd.read_csv(file[i])['Area']).reshape(5, 1)
        for j in range(size[0]):
            for k in range(1):
                if xa[i, j, k] < 0:
                    xa[i, j, k] = 0.

    rf = np.zeros([N_exp, 4])
    for i in range(N_exp):
        for j in range(4):
            if xl[i, j] < 1e-10:
                rf[i, j] = 0.
            else:
                rf[i, j] = xa[i, j + 1, -1]  / xl[i, j]

    rf_1 = np.zeros([N_exp, 4])
    for i in range(N_exp):
        for j in range(4):
                rf_1[i, j] = xa[i, j + 1, -1] / xa[i, 0, -1] * x_is[i, 0, 0]
    for i, c in enumerate(['SM', 'ortho', 'para', 'bis']):
        file1 = '/Users/' + PC + '/Dropbox/UCL/' + date + '/RF_' + c + '.xlsx'
        df0 = {'a': xa[:, i + 1, 0],
               'a_is': xa[:, 0, 0],
               'c': xl[:, i, 0],
               'c_is': x_is[:, 0, 0],
               'rf_with is': rf_1[:, i],
               'rf_without is': rf[:, i]}
        df = pd.DataFrame(df0)
        df.to_excel(file1, index=False)
    return rf, rf_1


def objective_cov(f, u, x_meas, N_exp, nx, n_points, nu, V, c1o, c2o,theta):

    ntheta = len(theta)
    x_meas1 = np.zeros([N_exp, nx[0], n_points + 1])
    xmin = np.zeros(nx[0]-1)
    xmax = np.zeros(nx[0]-1)#-1)
    x_meas_norm = x_meas.copy()
    for i in range(nx[0]-1):
        xmax[i] = np.max(x_meas[:, i, :])
        if xmax[i] > 1e-9:
            x_meas_norm[:, i, :] = x_meas[:, i, :]/xmax[i]
        else:
            x_meas_norm[:, i, :] = x_meas[:, i, :]
            xmax[i] = 1.

    dt      = np.zeros([N_exp, n_points])
    pp = 0
    s = 0
    x_init = np.zeros([N_exp,nx[0]])
    for i in range(nx[0] - 1):
        x_init[:N_exp, 0] = c1o * u[:N_exp, 1] / sum(u[:N_exp, i] for i in range(1, nu[0]))
    x_init[:N_exp, -1] = c2o * u[:N_exp, 2] / sum(u[:N_exp, i] for i in range(1, nu[0]))
    mle = 0
    for k0 in range(N_exp):
        x11 = x_init[k0, :]  # change it
        x_meas1[s, :, 0] = np.array(x11.T[:nx[0]])
        dt[k0, :] = V / np.sum(u[k0, 1:])/n_points
        for i in range(n_points):
            F = integrator_model(f, nu, nx, ntheta, 'embedded', 'mope', dt[k0, i])
            Fk = F(x0=vertcat(x11), p=vertcat(u[k0, :], theta[:8]))

            x11 = Fk['xf'][0:nx[0]]
            # + np.random.multivariate_normal([0.] * nx[0], np.diag(np. square(sigma))).T
            x_meas1[s, :, i + 1] = np.array(x11.T)
            pp += 1
            mle += maximum_likelihood_est(s, x_meas1[s,:-1,i+1] , x_meas_norm, [1, 1, 1, 1], i, xmax)

        s += 1


    obj = mle#np.linalg.eig(vv1)[0][0]#

    return obj
