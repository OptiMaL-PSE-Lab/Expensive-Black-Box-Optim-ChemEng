from casadi import *
import numpy as np
from pyDOE import *
from case_studies.MBDoE.Utilities_leeds import *
from scipy.stats import norm
import scipy

def construct_NLP_collocation(N_exp, f, x_0, x_init, lbx, ubx, lbu, ubu, lbtheta, ubtheta,
                              dt, N, x_meas, theta0, d, ms):
    nx = np.shape(x_init)[1]
    xmin = np.zeros(nx-1)
    xmax = np.zeros(nx-1)#-1)
    x_meas_norm = x_meas.copy()
    for i in range(nx-1):
        xmax[i] = np.max(x_meas[:, i, :])
        if xmax[i] > 1e-9:
            x_meas_norm[:, i, :] = x_meas[:, i, :]/xmax[i]
        else:
            x_meas_norm[:, i, :] = x_meas[:, i, :]
            xmax[i] = 1.
    ntheta = np.shape(lbtheta)[0]
    nu = np.shape(lbu)[2]
    C, D, B = construct_polynomials_basis(d, 'radau')
    # Start with an empty NLP
    w = []
    w0 = []
    lbw = []
    ubw = []

    g = []
    lbg = []
    ubg = []

    # For plotting x and u given w
    x_plot = []
    x_plotp = []
    u_plot = []

    mle = 0
    chi2 = 0
    s = 0

    thetak = MX.sym('theta', np.shape(lbtheta)[0])
    w += [thetak]


    lbw.extend(lbtheta)
    ubw.extend(ubtheta)

    w0.extend(theta0)
    for k_exp in range(N_exp):
        if s > 0:
           s += 1
        # "Lift" initial conditions
        Xk = MX.sym('X_'+str(s), nx)
        w += [Xk]

        lbw.extend(x_init[k_exp][:].tolist())
        ubw.extend(x_init[k_exp][:].tolist())
        w0.extend(x_init[k_exp][:].tolist())


        x_plot += [Xk]

        Uk = MX.sym('U_' + str(k_exp), nu)
        w += [Uk]
        lbw.extend(lbu[0][k_exp][:].tolist())
        ubw.extend(ubu[0][k_exp][:].tolist())
        w0.extend(ubu[0][k_exp][:].tolist())
        u_plot += [Uk]
    # Formulate the NLP
        m = 0
        for k in range(N*ms):
        # New NLP variable for the control

            h = dt[k_exp, m]/ms
        # --------------------
        # State at collocation points
# ---------------------------------
            lbw, ubw, w0, w, lbg, ubg, g, Xk = collocation(f, d, s, nx, nu, lbx,
                                                                ubx, lbw, ubw, w0, w,
                    lbg, ubg, g, x_init, Xk, k_exp, m, Uk, thetak, h, C, D)
# ---------------------------------
            if divmod(k + 1, ms)[1] == 0:
                x_plot += [Xk]

                mle += maximum_likelihood_est(k_exp, Xk[:-1], x_meas_norm, [1,1,1,1], m, xmax)

                chi2 += 2* maximum_likelihood_est(k_exp, Xk[:-1], x_meas, [0.005,0.005,0.003,0.003], m, [1.]*4)

                m += 1
            s += 1

    # Concatenate vectors
    # w = vertcat(*w)
    # g = vertcat(*g)
    # x_plot = horzcat(*x_plot)
    # u_plot = horzcat(*u_plot)
    # w0 = np.concatenate(w0)
    # lbw = np.concatenate(lbw)
    # ubw = np.concatenate(ubw)
    # lbg = np.concatenate(lbg)
    # ubg = np.concatenate(ubg)

    # Create an NLP solver
    problem = {'f': mle, 'x': vertcat(*w), 'g': vertcat(*g)}
    trajectories = Function('trajectories', [vertcat(*w)]
                            , [horzcat(*x_plot), horzcat(*u_plot), horzcat(*x_plotp), chi2], ['w'], ['x', 'u', 'xp',
                                                                                                     'chi2'])
    return problem, w0, lbw, ubw, lbg, ubg, trajectories


def construct_NLP_MBDoE_collocation(N_exp, f, x_0, xp_0, lbx, ubx, lbu, ubu, lbtheta, ubtheta, dt,
                                          N, u_meas, Vold, sigma, d, ms, c1o,c2o, V):
    nx = np.shape(x_0)[0]
    ntheta = np.shape(lbtheta)[0]
    nu = 4#np.shape(lbu)[0]
#    d = 8
    C, D, B = construct_polynomials_basis(d, 'radau')
    # Start with an empty NLP
    w = []
    w0 = []
    lbw = []
    ubw = []

    ss = np.zeros([8, 8])

    ss[:, 0] = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    ss[:, 1] = [0.5, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9, 0.9]
    ss[:, 2] = [0.2, 0.2, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9]
    ss[:, 3] = [1.0, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9, 0.9]
    ss[:, 4] = [0.1, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1]
    ss[:, 5] = [0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1]
    ss[:, 6] = [0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.1]
    ss[:, 7] = [0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9]

    A = np.zeros([8, 8,N_exp])
    for i in range(N_exp):
        A[:,:,i] = np.diag(ss[:,i])

    g = []
    lbg = []
    ubg = []

    # For plotting x and u given w
    x_plot = []
    x_plotp = []
    u_plot = []

    mle = Vold
    chi2 = 0
    s = 0

#    thetak = MX.sym('theta', np.shape(lbtheta)[0])
#    w += [thetak]

#    lbw.extend(lbtheta)
#    ubw.extend(ubtheta)

    ul = (np.array(ubu) + np.array(lbu))/2
    ur = (np.array(ubu) - np.array(lbu))/2

    lbu = [-1]*nu
    ubu = [1]*nu
 #   w0.extend(lbtheta)
    for k_exp in range(N_exp):
        if s > 0:
            s += 1
        # "Lift" initial conditions


        Uk = SX.sym('U_' + str(k_exp), nu)
        w += [Uk]
        lbw.extend(lbu)
        ubw.extend(ubu)
        w0.extend(2*np.random.rand(nu)-1)
        u_plot += [Uk]

        Xk = SX.sym('X_' + str(s), nx)
        w += [Xk]
        Xpk = SX.sym('Xp_' + str(s), nx * ntheta)
        w += [Xpk]

        lbw.extend([0.0])#[u_meas[k_exp][2]])
        lbw.extend([0] * (nx - 2))
        lbw.extend([0])

        ubw.extend([inf])#[u_meas[k_exp][2]])
        ubw.extend([0] * (nx - 2))
        ubw.extend([inf])

        w0.extend([0.0])#[u_meas[k_exp][2]])
        w0.extend([0] * (nx - 2))
        w0.extend([1.4])

        x_plot += [Xk]

        uu = Uk*ur + np.array(ul)

        g += [Xk[0] - uu[1]/(uu[1]+uu[2]+uu[3])*c1o]
        lbg.extend([-1e-8] * 1)
        ubg.extend([1e-8] * 1)

        g += [Xk[nx-1] - uu[2]/(uu[1]+uu[2]+uu[3])*c2o]
        lbg.extend([-1e-8] * 1)
        ubg.extend([1e-8] * 1)

        lbw.extend(xp_0)
        ubw.extend(xp_0)
        w0.extend(xp_0)
        x_plotp += [Xpk]


        # Formulate the NLP
        m = 0
        DTk = SX.sym('DT_' + str(0), N)
        w += [DTk]
        lbw += [0.001]*N#[*dt[1, :]]#
        ubw += [100.]*N#[*dt[1, :]]#
        w0 += [*dt[1, :]]#[0.1]*(N+1)
        Ts = [DTk]
        for i in range(N):
            g += [DTk[i] - V/(uu[1]+uu[2]+uu[3])]
            lbg += [-1e-8]
            ubg += [1e-8]
        for k in range(N*ms):
            # New NLP variable for the control

            # -------------------
            #            DTk = SX.sym('DT_' + str(s))
            #            w += [DTk]
            #            lbw += [dt[k]]
            #            ubw += [dt[k]]
            #            w0 += [dt[k]]
            #    Ts += [DTk]
            h = DTk[m]/ms
            # --------------------
            # State at collocation points
            Xc = []
            Xpc = []
            for j in range(d):
                Xkj = SX.sym('X_' + str(s) + '_' + str(j), nx)
                Xc += [Xkj]
                w += [Xkj]
                lbw.extend(lbx)
                ubw.extend(ubx)
 #               ubw.extend([1.4])
                w0.extend([0]*nx)

            # Loop over collocation points
            Xk_end = D[0] * Xk
            for j in range(d):
                Xpkj = SX.sym('Xp_' + str(s) + '_' + str(j), nx * ntheta)
                Xpc += [Xpkj]
                w += [Xpkj]
                lbw.extend([-np.inf] * (nx * ntheta))
                ubw.extend([np.inf] * (nx * ntheta))
                w0.extend([0] * (nx * ntheta))

            # Loop over collocation points
            Xpk_end = D[0] * Xpk
            for j in range(1, d + 1):
                # Expression for the state derivative at the collocation point
                xp = C[0, j] * Xk
                xpp = C[0, j] * Xpk
                for r in range(d):
                    xp = xp + C[r + 1, j] * Xc[r]
                    xpp = xpp + C[r + 1, j] * Xpc[r]

                # Append collocation equations
                fj, qj, dxpj = f(Xc[j - 1], Uk*ur + np.array(ul), lbtheta, Xpc[j - 1])#
                g += [(h * fj - xp)]
                lbg.extend([-1e-8] * nx)
                ubg.extend([1e-8] * nx)

                g += [(h * dxpj - xpp)]
                lbg.extend([-1e-8] * (nx * ntheta))
                ubg.extend([1e-8] * (nx * ntheta))
                # Add contribution to the end state
                Xk_end = Xk_end + D[j] * Xc[j - 1]

                Xpk_end = Xpk_end + D[j] * Xpc[j - 1]
            #            if int(j1) < np.shape(t_meas)[0]:
            #                if np.real(k * T / N) == t_meas[j1]:
            #                    count[k] = 1
            #                    j1 += 1
            # Add contribution to quadrature function
            #      J = J + B[j]*qj*h

            # New NLP variable for state at end of interval
            Xk = SX.sym('X_' + str(s + 1), nx)
            w += [Xk]
            lbw.extend(lbx)
            ubw.extend(ubx)
 #           ubw.extend([1.4])
            w0.extend([0]*nx)
            x_plot += [Xk]
            Xpk = SX.sym('Xp_' + str(s + 1), nx * ntheta)
            w += [Xpk]
            lbw.extend([-np.inf] * (nx * ntheta))
            ubw.extend([np.inf] * (nx * ntheta))
            w0.extend([0] * (nx * ntheta))
            x_plotp += [Xpk]
            # Add equality constraint
            g += [Xk_end - Xk]
            lbg.extend([-1e-8] * nx)
            ubg.extend([1e-8] * nx)

            g += [Xpk_end - Xpk]
            lbg.extend([-1e-8] * (nx * ntheta))
            ubg.extend([1e-8] * (nx * ntheta))
            s += 1
            if divmod(k + 1, ms)[1] == 0:
                m += 1
                mle = fim_for_single_t(Xpk, mle, sigma, nx, ntheta, 1)

            #            mle += maximum_likelihood_est(k_exp, Xk, x_meas, [1, 1, 1, 1], k)
            #            chi2 += maximum_likelihood_est(k_exp, Xk, x_meas, Xk + 0.0001, k)
            # Concatenate vectors
            # w = vertcat(*w)
            # g = vertcat(*g)
            # x_plot = horzcat(*x_plot)
            # u_plot = horzcat(*u_plot)
            # w0 = np.concatenate(w0)
            # lbw = np.concatenate(lbw)
            # ubw = np.concatenate(ubw)
            # lbg = np.concatenate(lbg)
            # ubg = np.concatenate(ubg)

            # Create an NLP solver
    #g += [np.ones([1, N]) @ DTk]
    #lbg.extend([0.5])
    #ubg.extend([2.09])
    mle = -(log(det(mle))+0.0001)
    problem = {'f': mle, 'x': vertcat(*w), 'g': vertcat(*g)}
    trajectories = Function('trajectories', [vertcat(*w)]
                                , [horzcat(*x_plot), horzcat(*u_plot), horzcat(*x_plotp), horzcat(*Ts)], ['w'], ['x',
                                                                                                                 'u',
                                                                                                                 'xp',
                                                                                                                 'Ts'])

    return problem, w0, lbw, ubw, lbg, ubg, trajectories


def bayopt_design(f, lbu, ubu, nu, nx, V_old, n_points, theta1, sigma,V, c1o, c2o):


    n_s = 2*len(theta1) + 1
    lhd = 2 * lhs(nu[0], samples=n_s) - 1

    # ------- Transform the normalized variables to the real ones -----

    set_u = lhd

    range_u = np.array([(lbu), (ubu)]).T

    u_t = (range_u.max(axis=1) - range_u.min(axis=1)) / 2 * set_u \
          + (range_u.max(axis=1) + range_u.min(axis=1)) / 2

    u_t = u_t.T

    s = np.zeros([1, n_s])
    for i in range(n_s):
        s[0, i] = objective(f, u_t[:, i].reshape(1, nu[0]), V_old, 1, nx, n_points, nu, theta1, sigma,V, c1o, c2o)

        dim = nu[0]
        min_val = 1
        min_x = None

    GP = GP_model(u_t.T, s.T, 'RBF', 10, [])
    mean, var = GP.GP_predictor([])

    def min_obj(X, mean, var):
        fs = np.max(s)
        Delta   = mean(X)-fs
        Delta_p = np.max(mean(X)-fs)
        if var(X)==0.:
            Z = 0.
        else:
            Z = (Delta)/var(X)**0.5
        return (Delta_p + (var(X)**0.5)*norm.pdf(Z)-abs(Delta)*norm.cdf(Z))#-(GP.GP_inference_np(X)[0][0] + 3 * GP.GP_inference_np(X)[1][0])#
    #
    A = np.array([[0,1.,0,-1],[0,0,1,-1]])
    ub = np.array([3, 3])
    lb = np.array([0,0])

    C = scipy.optimize.LinearConstraint(A, lb, ub)
    start = time.time()

    for i in range(2):
        min_val = np.inf
        lhd_m = 2 * lhs(nu[0], samples=10) - 1

        # ------- Transform the normalized variables to the real ones -----

        set_u_m = lhd_m

        u_t_m = (range_u.max(axis=1) - range_u.min(axis=1)) / 2 * set_u_m \
                + (range_u.max(axis=1) + range_u.min(axis=1)) / 2

        min_x = u_t_m[0, :]
        for x0 in np.vstack((u_t_m, u_t[:, np.argmax(s)])):
            res = minimize(min_obj, x0=x0, args=(mean, var), bounds=range_u, method='SLSQP', constraints = [])
            if res.fun < min_val:
                min_val = res.fun  # [0]
                min_x = res.x

        y_next = objective(f, min_x.reshape(1, nu[0]), V_old, 1, nx, n_points, nu, theta1, sigma,V, c1o, c2o)

        u_t = np.hstack((u_t, min_x.reshape(nu[0], 1)))
        s = np.hstack((s, y_next.reshape(1, 1)))
        #print(GP.hypopt)

        GP = GP_model(u_t.T, s.T, 'RBF', 3, [])
        mean, var = GP.GP_predictor([])

    #def min_obj(X):
    #    fs = np.max(s)
    #    Delta   = mean(X)-fs
    #    Delta_p = np.max(mean(X)-fs)
    #    if var(X)==0.:
    #        Z = 0.
    #    else:
    #        Z = (Delta)/var(X)**0.5
    #    return 1.#(Delta_p + (var(X)**0.5)*norm.pdf(Z)-abs(Delta)*norm.cdf(Z))#-(GP.GP_inference_np(X)[0][0] + 3 * GP.GP_inference_np(X)[1][0])#
    #
    elapsed_time_fl = (time.time() - start)
    print(elapsed_time_fl)
    return u_t, s


def bayopt_pe(f, lbu, ubu, nu, nx, x_meas, n_points, ntheta, u_meas, N_exp, V, c1o, c2o, theta_pre):


    n_s = ntheta*2+1
    lhd = 2 * lhs(ntheta, samples=n_s) - 1

    # ------- Transform the normalized variables to the real ones -----

    set_u = lhd

    range_u = np.array([lbu, ubu]).T

    u_t = (range_u.max(axis=1) - range_u.min(axis=1)) / 2 * set_u \
          + (range_u.max(axis=1) + range_u.min(axis=1)) / 2
    #u_t = np.vstack((u_t, theta_pre.reshape((-1,))))
    u_t = u_t.T
    #u_t = np.vstack((u_t, theta_pre.reshape((-1,))))
    s = np.zeros([1, n_s+1])
    for i in range(n_s+1):
        s[0, i] = objective_pe(f, u_meas, x_meas, N_exp, nx, n_points, nu, u_t[:, i], V, c1o, c2o)

        dim = nu[0]
        min_val = 1
        min_x = None

    GP = GP_model(u_t.T, s.T, 'RBF', 10, [])
    mean, var = GP.GP_predictor([])
    def min_obj(X):
        fs = np.min(s)
        Delta   = -mean(X)+fs
        Delta_p = np.min(-mean(X)+fs)
        if var(X)==0.:
            Z = 0.
        else:
            Z = (Delta)/var(X)**0.5
        return -(Delta_p + (var(X)**0.5)*norm.pdf(Z)+abs(Delta)*norm.cdf(Z))#
    start = time.time()

    for i in range(1000):
        min_val = np.inf
        lhd_m = 2 * lhs(ntheta, samples=20) - 1

        # ------- Transform the normalized variables to the real ones -----

        set_u_m = lhd_m

        u_t_m = (range_u.max(axis=1) - range_u.min(axis=1)) / 2 * set_u_m \
                + (range_u.max(axis=1) + range_u.min(axis=1)) / 2

        min_x = u_t_m#[0, :]
        for x0 in np.vstack((u_t_m, u_t[:, np.argmin(s)])):
            res = minimize(min_obj, x0=x0, bounds=range_u, method='L-BFGS-B')
            if res.fun < min_val:
                min_val = res.fun  # [0]
                min_x = res.x

        y_next = objective_pe(f, u_meas, x_meas, N_exp, nx, n_points, nu, min_x, V, c1o, c2o)


        u_t = np.hstack((u_t, min_x.reshape(ntheta, 1)))
        s = np.hstack((s, y_next.reshape(1, 1)))
        #print(GP.hypopt)

        GP = GP_model(u_t.T, s.T, 'RBF', 10, [])
        mean, var = GP.GP_predictor([])

        def min_obj(X):
            fs = np.min(s)
            Delta = -mean(X) + fs
            Delta_p = np.min(-mean(X) + fs)
            if var(X) == 0.:
                Z = 0.
            else:
                Z = (Delta) / var(X) ** 0.5
            return -(Delta_p + (var(X) ** 0.5) * norm.pdf(Z) + abs(Delta) * norm.cdf(Z))  #

    elapsed_time_fl = (time.time() - start)
    print(elapsed_time_fl)
    return u_t, s

def bayopt_design_unc(f, lbu, ubu, nu, nx, V_old, n_points, theta1, sigma,V, c1o, c2o):


    n_s = 40
    lhd = 2 * lhs(nu[0], samples=n_s) - 1

    # ------- Transform the normalized variables to the real ones -----

    set_u = lhd

    range_u = np.array([lbu, ubu]).T

    u_t = (range_u.max(axis=1) - range_u.min(axis=1)) / 2 * set_u \
          + (range_u.max(axis=1) + range_u.min(axis=1)) / 2

    u_t = u_t.T

    s = np.zeros([1, n_s])
    for i in range(n_s):
        s[0, i] = objective(f, u_t[:, i].reshape(1, nu[0]), V_old, 1, nx, n_points, nu, theta1, sigma,V, c1o, c2o)

        dim = nu[0]
        min_val = 1
        min_x = None

    GP = GP_model(u_t.T, s.T, 'RBF', 10)

    def min_obj(X):
        return -(GP.GP_inference_np(X)[0][0] + 3 * GP.GP_inference_np(X)[1][0])

    #
    n_s_m = 10
    lhd_m = 2 * lhs(nu[0], samples=n_s) - 1

    # ------- Transform the normalized variables to the real ones -----

    set_u_m = lhd_m

    u_t_m = (range_u.max(axis=1) - range_u.min(axis=1)) / 2 * set_u_m \
            + (range_u.max(axis=1) + range_u.min(axis=1)) / 2

    min_x = u_t_m[0, :]
    start = time.time()

    for i in range(50):
        min_val = np.inf

        for x0 in np.vstack((u_t_m, u_t[:, np.argmax(s)])):
            res = minimize(min_obj, x0=x0, bounds=range_u, method='L-BFGS-B')
            if res.fun < min_val:
                min_val = res.fun  # [0]
                min_x = res.x

        y_next = objective(f, min_x.reshape(1, nu[0]), V_old, 1, nx, n_points, nu, theta1, sigma,V, c1o, c2o)

        u_t = np.hstack((u_t, min_x.reshape(nu[0], 1)))
        s = np.hstack((s, y_next.reshape(1, 1)))

        GP = GP_model(u_t.T, s.T, 'RBF', 2)

        def min_obj(X):
            return -(GP.GP_inference_np(X)[0][0] + 3 * np.exp(-0.1 * i) * GP.GP_inference_np(X)[1][0] ** 0.5)

    elapsed_time_fl = (time.time() - start)
    print(elapsed_time_fl)
    return u_t, s

def train_GP_for_parametric(f,N_pre, u_s, nu, nx, ntheta, lbu, ubu, lbx, ubx, lbtheta, ubtheta, npoints, theta1, vv1, V, c1o, c2o):


    n_exp     = 200
    N         = n_exp
    set_u     = 2 * lhs(nu[0], samples=N) - 1
    set_x     = 2 * lhs(nx[0], samples=N) - 1
    set_theta = 2 * lhs(ntheta[0], samples=N) - 1

    # ------- Transform the normalized variables to the real ones -----

    range_u     = np.array([lbu, ubu]).T
    range_x     = np.array([lbx, ubx]).T
    range_theta = np.array([lbtheta, ubtheta]).T

    u_t     = (range_u.max(axis=1) - range_u.min(axis=1)) / 2 * set_u \
          + (range_u.max(axis=1) + range_u.min(axis=1)) / 2
#    x0_t    = (range_x.max(axis=1) - range_x.min(axis=1)) / 2 * set_x \
#          + (range_x.max(axis=1) + range_x.min(axis=1)) / 2
    theta_t = np.random.multivariate_normal(theta1, np.linalg.pinv(vv1),N)#(range_theta.max(axis=1)/2 - range_theta.min(axis=1)/2) / 2 * set_theta \
          #+ (range_theta.max(axis=1)/2 + range_theta.min(axis=1)/2) / 2
    #u_t     = u_t.T
#    x0_t    = x0_t.T

    #u_t     = u_t.reshape(N, nu[0])
    u_t     = np.vstack((u_t, u_s))
    theta_t = np.vstack((theta_t, [theta1]*N_pre))
    n_exp  += N_pre
    N       = n_exp
    """
    Generate data-set
    """
    N_test = 10
    theta  = np.zeros([(ntheta[0]), npoints * N])
    his_x0 = np.zeros([nx[0], npoints * N])
    his_x  = np.zeros([nx[0], npoints * N])
    his_x1 = np.zeros([nx[0], npoints * N])
    his_x2 = np.zeros([nx[0], npoints * N])
    h_s    = np.zeros([nx[0], nx[0], npoints * N])
    u_t1   = np.zeros([nu[0], npoints * N])
    u_t    = u_t.reshape(N, nu[0])



    dt      = np.zeros([N,npoints])

    x_init = np.zeros([N,nx[0]])
    for i in range(nx[0] - 1):
        x_init[:N, 0] = c1o * u_t[:N, 1] / sum(u_t[:N, i] for i in range(1, nu[0]))
    x_init[:N, -1] = c2o * u_t[:N, 2] / sum(u_t[:N, i] for i in range(1, nu[0]))
    s              = 0
    for k0 in range(N):
        x11            = x_init[k0, :]  # change it
        xp1 = np.zeros([nx[0] * ntheta[0], 1])
        dt[k0, :] = V / np.sum(u_t[k0, 1:])/npoints
        for i in range(npoints):
            his_x0[:, k0] = np.array(x11.T[:nx[0]])
            F             = integrator_model(f, nu, nx, ntheta, 'embedded', 'sensitivity', dt[k0, i])
            Fk            = F(x0=vertcat(x11, xp1), p=vertcat(u_t[k0, :], theta_t[k0, :]))

            x11 = Fk['xf'][0:nx[0]]
            xp1 = Fk['xf'][nx[0]:]

            # + np.random.multivariate_normal([0.] * nx[0], np.diag(np. square(sigma))).T
            his_x[:, s] = np.array(x11.T)
#            xp_meas[:, pp] = np.array(xp1.T)
#            pp += 1
            s += 1

    GP = GP_model(np.array([*u_t[:n_exp, :].T, *theta_t[:n_exp, :].T]).T, his_x[:, :n_exp].T, 'RBF', 1, [])
    #import pickle
    #pickle.dump(GP,open( "GP_param2.p", "wb" ))
    #GP = pickle.load(open("GP_param.p", "rb"))
    #s = np.block([[np.zeros([u_t.shape[1], u_t.shape[1]]), np.zeros([u_t.shape[1], len(theta1)])],
    #              [np.zeros([len(theta1), u_t.shape[1]]), np.linalg.pinv(vv1)]])
    #mean_sample, var_sample = GP.step_Uncert_prop([*u_t[0, :].T, *theta1.T], s)
    #mu, vf, Jmu, Jvf, hmu, hvf = GP. derivatives_gp()#gp_exact_moment([], [], [], [], [*u_t[0, :].T, *theta1.T], s)

    #mu_1 = mu([*u_t[0, :].T, *theta1.T])
    #vv = np.zeros([5,5])
    #for i in range(5):
    #    vv[i, i] =  (vf([*u_t[0, :].T, *theta1.T]) [i]
    #          + trace(s @ (0.5 * hvf[i]([*u_t[0, :].T, *theta1.T]).T
    #                       + Jmu([*u_t[0, :].T, *theta1.T])[i,:] @ Jmu([*u_t[0, :].T, *theta1.T])[i,:].T)))
#
#    for i in range(N):
#        his_x1[:, i], _ = GP.GP_inference_np(np.array([*u_t[i, :].T, *theta_t[i, :].T]))
    return GP, u_t[:n_exp, :], theta_t[:n_exp,:], his_x[:, :n_exp]

def train_GP_for_parametric1(f,N_pre, u_s, nu, nx, ntheta, lbu, ubu, lbx, ubx, lbtheta, ubtheta, npoints, theta1, vv1, V, c1o, c2o):


    n_exp     = 500
    N         = n_exp
    set_u     = 2 * lhs(nu[0], samples=N) - 1
    set_x     = 2 * lhs(nx[0], samples=N) - 1
    set_theta = 2 * lhs(ntheta[0], samples=N) - 1

    # ------- Transform the normalized variables to the real ones -----

    range_u     = np.array([lbu, ubu]).T
    range_x     = np.array([lbx, ubx]).T
    range_theta = np.array([theta1-2, theta1+2]).T

    u_t     = (range_u.max(axis=1) - range_u.min(axis=1)) / 2 * set_u \
          + (range_u.max(axis=1) + range_u.min(axis=1)) / 2
#    x0_t    = (range_x.max(axis=1) - range_x.min(axis=1)) / 2 * set_x \
#          + (range_x.max(axis=1) + range_x.min(axis=1)) / 2
#    theta_t = np.random.multivariate_normal(theta1, np.linalg.pinv(vv1),N)#(range_theta.max(axis=1)/2 - range_theta.min(axis=1)/2) / 2 * set_theta \
          #+ (range_theta.max(axis=1)/2 + range_theta.min(axis=1)/2) / 2
    #u_t     = u_t.T
#    x0_t    = x0_t.T
    pos = np.zeros([N, 1])
    theta_0 = np.zeros([N, 8])
    k = 0
    theta_t1 = (range_theta.max(axis=1) - range_theta.min(axis=1)) / 2 * set_theta \
               + (range_theta.max(axis=1) + range_theta.min(axis=1)) / 2
#    for i in range(N):

#        pos[i] = (theta_t1[i, :] - theta1).reshape((-1,1)).T @ vv1 @ (theta_t1[i, :] - theta1).reshape((-1,1)) - 10
#        if pos[i] <= 0:
#            theta_0[k, :] = theta_t1[i, :]
#            k += 1
#    theta_t = theta_0[:k,:]
    theta_t = theta_t1
    k     = N
    n_exp = k
    #u_t     = u_t.reshape(N, nu[0])
    u_t     = np.vstack((u_t[:k,:], u_s))
    theta_t = np.vstack((theta_t, [theta1]*N_pre))
    k  += N_pre
    N = k
    """
    Generate data-set
    """
    N_test = 10
    theta  = np.zeros([(ntheta[0]), npoints * N])
    his_x0 = np.zeros([nx[0], npoints * N])
    his_x  = np.zeros([nx[0], npoints * N])
    his_x1 = np.zeros([nx[0], npoints * N])
    his_x2 = np.zeros([nx[0], npoints * N])
    h_s    = np.zeros([nx[0], nx[0], npoints * N])
    u_t1   = np.zeros([nu[0], npoints * N])
    u_t    = u_t.reshape(N, nu[0])


    dt      = np.zeros([N,npoints])

    x_init = np.zeros([N,nx[0]])
    for i in range(nx[0] - 1):
        x_init[:N, 0] = c1o * u_t[:N, 1] / sum(u_t[:N, i] for i in range(1, nu[0]))
    x_init[:N, -1] = c2o * u_t[:N, 2] / sum(u_t[:N, i] for i in range(1, nu[0]))
    s              = 0
    for k0 in range(N):
        x11            = x_init[k0, :]  # change it
        xp1 = np.zeros([nx[0] * ntheta[0], 1])
        dt[k0, :] = V / np.sum(u_t[k0, 1:])/npoints
        for i in range(npoints):
            his_x0[:, k0] = np.array(x11.T[:nx[0]])
            F             = integrator_model(f, nu, nx, ntheta, 'embedded', 'sensitivity', dt[k0, i])
            Fk            = F(x0=vertcat(x11, xp1), p=vertcat(u_t[k0, :], [*theta_t[k0, :1],*theta1[1:]]))

            x11 = Fk['xf'][0:nx[0]]
            xp1 = Fk['xf'][nx[0]:]

            # + np.random.multivariate_normal([0.] * nx[0], np.diag(np. square(sigma))).T
            his_x[:, s] = np.array(x11.T)
#            xp_meas[:, pp] = np.array(xp1.T)
#            pp += 1
            s += 1

    GP = GP_model(np.array([*u_t[:n_exp-20, :].T, *theta_t[:n_exp-20, :1].T]).T, his_x[:, :n_exp-20].T, 'RBF', 10, [])
    #import pickle
    #pickle.dump(GP,open( "GP_param2.p", "wb" ))
    #GP = pickle.load(open("GP_param.p", "rb"))
    #s = np.block([[np.zeros([u_t.shape[1], u_t.shape[1]]), np.zeros([u_t.shape[1], len(theta1)])],
    #              [np.zeros([len(theta1), u_t.shape[1]]), np.linalg.pinv(vv1)]])
    #mean_sample, var_sample = GP.step_Uncert_prop([*u_t[0, :].T, *theta1.T], s)
    #mu, vf, Jmu, Jvf, hmu, hvf = GP. derivatives_gp()#gp_exact_moment([], [], [], [], [*u_t[0, :].T, *theta1.T], s)

    #mu_1 = mu([*u_t[0, :].T, *theta1.T])
    #vv = np.zeros([5,5])
    #for i in range(5):
    #    vv[i, i] =  (vf([*u_t[0, :].T, *theta1.T]) [i]
    #          + trace(s @ (0.5 * hvf[i]([*u_t[0, :].T, *theta1.T]).T
    #                       + Jmu([*u_t[0, :].T, *theta1.T])[i,:] @ Jmu([*u_t[0, :].T, *theta1.T])[i,:].T)))
#
#    for i in range(N):
#        his_x1[:, i], _ = GP.GP_inference_np(np.array([*u_t[i, :].T, *theta_t[i, :].T]))
    import GPy
    X = np.array([*u_t[:n_exp - 20, :].T, *theta_t[:n_exp - 20, :1].T]).T
    X_n = ((X - X.mean(axis=0)) / X.std(axis=0))
    kernel = GPy.kern.RBF(input_dim=4, ARD=True)
    Y = his_x[:, :n_exp - 20].T
    Y_n = ((Y - Y.mean(axis=0)) / Y.std(axis=0))
    m = GPy.models.GPRegression(X_n, Y_n, kernel)
    m.optimize_restarts(num_restarts=20)

    from sklearn import preprocessing
    return GP, u_t[:n_exp, :], theta_t[:n_exp,:], his_x[:, :n_exp]




def update_GP_parametric(GP, u_t, theta_t, his_x):

    GP = GP_model(np.array([*u_t.T, *theta_t.T]).T, his_x.T, 'RBF', 6, [])
    return GP,  u_t, theta_t, his_x

def MBDoE_GP(N_exp, f, Delta, x_0, xp_0, lbx, ubx, lbu, ubu, lbtheta, ubtheta, dt,
             N, uk, Vold, sigma, d, c1o, c2o, V, Delta1):
    nx = np.shape(x_0)[0]
    ntheta = np.shape(lbtheta)[0]
    nu = 4  # np.shape(lbu)[0]
    #    d = 8
    C, D, B = construct_polynomials_basis(d, 'radau')
    # Start with an empty NLP
    w = []
    w0 = []
    lbw = []
    ubw = []

    ss = np.zeros([8, 8])

    ss[:, 0] = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    ss[:, 1] = [0.5, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9, 0.9]
    ss[:, 2] = [0.2, 0.2, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9]
    ss[:, 3] = [1.0, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9, 0.9]
    ss[:, 4] = [0.1, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1]
    ss[:, 5] = [0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1]
    ss[:, 6] = [0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.1]
    ss[:, 7] = [0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9]

    A = np.zeros([8, 8, N_exp])
    for i in range(N_exp):
        A[:, :, i] = np.diag(ss[:, i])

    g = []
    lbg = []
    ubg = []

    # For plotting x and u given w
    x_plot = []
    x_plotp = []
    u_plot = []
    v_plot = []
    mle = Vold
    chi2 = 0
    s = 0

    #    thetak = SX.sym('theta', np.shape(lbtheta)[0])
    #    w += [thetak]

    #    lbw.extend(lbtheta)
    #    ubw.extend(ubtheta)

    ul = (np.array(ubu) + np.array(lbu)) / 2
    ur = (np.array(ubu) - np.array(lbu)) / 2

    lbu = [-1] * nu
    ubu = [1] * nu
    #   w0.extend(lbtheta)
    for k_exp in range(N_exp):
        if s > 0:
            s += 1
        # "Lift" initial conditions

        Uk = SX.sym('U_' + str(k_exp), nu)
        w += [Uk]
        lbw.extend(lbu)
        ubw.extend(ubu)
        w0.extend(2 * np.random.rand(nu) - 1)
        u_plot += [Uk]

        #g += [sum1(Uk**2)]
        #lbg.extend([0.0] * 1)
        #ubg.extend([0.2] * 1)


        Xk = SX.sym('X_' + str(s), nx)
        w += [Xk]
        vk = SX.sym('v_' + str(s), nx)
        w += [vk]
        Xpk = SX.sym('Xp_' + str(s), nx * ntheta)
        w += [Xpk]

        lbw.extend([0.0])  # [u_meas[k_exp][2]])
        lbw.extend([0] * (nx - 2))
        lbw.extend([0])

        ubw.extend([100])  # [u_meas[k_exp][2]])
        ubw.extend([0] * (nx - 2))
        ubw.extend([100])

        w0.extend([c1o*uk[1]/sum(uk[1:])])  # [u_meas[k_exp][2]])
        w0.extend([0] * (nx - 2))
        w0.extend([c2o*uk[2]/sum(uk[1:])])

        lbw.extend([0.0]*nx)
        ubw.extend([inf]*nx)
        w0.extend([0.0]*nx)  # [u_meas[k_exp][2]])

        x_plot += [Xk]
        v_plot += [vk]
        x_plotp += [Xpk]

        uu = (Uk) * ur + np.array(ul)

        g += [Xk[0] - uu[1] / (uu[1] + uu[2] + uu[3]) * c1o]
        lbg.extend([-1e-8] * 1)
        ubg.extend([1e-8] * 1)

        g += [Xk[nx - 1] - uu[2] / (uu[1] + uu[2] + uu[3]) * c2o]
        lbg.extend([-1e-8] * 1)
        ubg.extend([1e-8] * 1)

        lbw.extend(xp_0)
        ubw.extend(xp_0)
        w0.extend(xp_0)

        # Formulate the NLP
        m = 0
        DTk = SX.sym('DT_' + str(0), N)
        w += [DTk]
        lbw += [0.001] * N  # [*dt[1, :]]#
        ubw += [100.] * N  # [*dt[1, :]]#
        w0 += [*dt[1, :]]  # [0.1]*(N+1)
        Ts = [DTk]
        for i in range(N):
            g += [DTk[i] - V / (uu[1] + uu[2] + uu[3])]
            lbg += [-1e-8]
            ubg += [1e-8]
        for k in range(N):
            # New NLP variable for the control

            h = DTk[m]
            # --------------------

            fj, dxpj, vj = f(Uk * ur + np.array(ul), lbtheta, Xpk)  #
            Xk_end = fj
            Xpk_end = dxpj
            vk_end  = vj

            # New NLP variable for state at end of interval
            Xk = SX.sym('X_' + str(k + 1), nx)
            w += [Xk]

            lbw.extend(lbx)
            ubw.extend(ubx)
            w0.extend([c1o * uk[1] / sum(uk[1:])])  # [u_meas[k_exp][2]])
            w0.extend([0] * (nx - 2))
            w0.extend([c2o * uk[2] / sum(uk[1:])])
            x_plot += [Xk]

            vk = SX.sym('v_' + str(s), nx)
            w += [vk]

            lbw.extend([0] * nx)
            ubw.extend(Delta)
            w0.extend([0] * nx)
            v_plot += [vk]

            Xpk = SX.sym('Xp_' + str(k + 1), nx * ntheta)
            w += [Xpk]
            lbw.extend([-np.inf] * (nx * ntheta))
            ubw.extend([np.inf] * (nx * ntheta))
            w0.extend([0] * (nx * ntheta))

            x_plotp += [Xpk]
            # Add equality constraint
            g += [Xk_end - Xk]
            lbg.extend([-1e-8] * nx)
            ubg.extend([1e-8] * nx)

            g += [vk_end - vk]
            lbg.extend([-1e-8] * nx)
            ubg.extend([1e-8] * nx)

            g += [Xpk_end - Xpk]
            lbg.extend([-0.00001] * (nx * ntheta))
            ubg.extend([0.00001] * (nx * ntheta))

           # g += [sum1(vk**2)-Delta]
           # lbg.extend([-inf])
           # ubg.extend([0.])


            g += [Xk[0] + 3*sqrt(vk[0]+1e-20)]
            lbg.extend([0.0])
            ubg.extend([0.1])

            g += [sum1((Uk - (uk-np.array(ul))/ur)**2)-Delta1]
            lbg.extend([-100])
            ubg.extend([0.0])
            #g += [Xk[-2] + 3*sqrt(vk[0])]
            #lbg.extend([-inf])
            #ubg.extend([0.05])
            s += 1
            m += 1
            mle = fim_for_single_t(Xpk, mle, sigma, nx, ntheta, 1)


    mle = -(log(det(mle)) + 0.0001)
    problem = {'f': mle, 'x': vertcat(*w), 'g': vertcat(*g)}
    trajectories = Function('trajectories', [vertcat(*w)]
                            , [horzcat(*x_plot), horzcat(*u_plot), horzcat(*x_plotp),horzcat(*v_plot), horzcat(*Ts)], ['w'], ['x',
                                                                                                             'u',
                                                                                                             'xp',
                                                                                                              'v',
                                                                                                             'Ts'])

    return problem, w0, lbw, ubw, lbg, ubg, trajectories



def construct_NLP_MBDoE_collocation_nom(N_exp, f, x_0, xp_0, lbx, ubx, lbu, ubu, lbtheta, ubtheta, dt,
                                          N, u_meas, Vold, sigma, d, ms, c1o,c2o, V):
    nx = np.shape(x_0)[0]
    ntheta = np.shape(lbtheta)[0]
    nu = 4#np.shape(lbu)[0]
#    d = 8
    C, D, B = construct_polynomials_basis(d, 'radau')
    # Start with an empty NLP
    w = []
    w0 = []
    lbw = []
    ubw = []

    ss = np.zeros([8, 8])

    ss[:, 0] = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    ss[:, 1] = [0.5, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9, 0.9]
    ss[:, 2] = [0.2, 0.2, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9]
    ss[:, 3] = [1.0, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9, 0.9]
    ss[:, 4] = [0.1, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1]
    ss[:, 5] = [0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1]
    ss[:, 6] = [0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.1]
    ss[:, 7] = [0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9]

    A = np.zeros([8, 8,N_exp])
    for i in range(N_exp):
        A[:,:,i] = np.diag(ss[:,i])

    g = []
    lbg = []
    ubg = []

    # For plotting x and u given w
    x_plot = []
    x_plotp = []
    u_plot = []

    mle = Vold
    chi2 = 0
    s = 0

#    thetak = MX.sym('theta', np.shape(lbtheta)[0])
#    w += [thetak]

#    lbw.extend(lbtheta)
#    ubw.extend(ubtheta)

    ul = (np.array(ubu) + np.array(lbu))/2
    ur = (np.array(ubu) - np.array(lbu))/2

    lbu = [-1]*nu
    ubu = [1]*nu
 #   w0.extend(lbtheta)
    for k_exp in range(N_exp):
        if s > 0:
            s += 1
        # "Lift" initial conditions


        Uk = SX.sym('U_' + str(k_exp), nu)
        w += [Uk]
        lbw.extend(lbu)
        ubw.extend(ubu)
        w0.extend(2*np.random.rand(nu)-1)
        u_plot += [Uk]

        Xk = SX.sym('X_' + str(s), nx)
        w += [Xk]
        Xpk = SX.sym('Xp_' + str(s), nx * ntheta)
        w += [Xpk]

        lbw.extend([0.0])#[u_meas[k_exp][2]])
        lbw.extend([0] * (nx - 2))
        lbw.extend([0])

        ubw.extend([inf])#[u_meas[k_exp][2]])
        ubw.extend([0] * (nx - 2))
        ubw.extend([inf])

        w0.extend([0.0])#[u_meas[k_exp][2]])
        w0.extend([0] * (nx - 2))
        w0.extend([1.4])

        x_plot += [Xk]

        uu = Uk*ur + np.array(ul)

        g += [Xk[0] - uu[1]/(uu[1]+uu[2]+uu[3])*c1o]
        lbg.extend([-1e-8] * 1)
        ubg.extend([1e-8] * 1)

        g += [Xk[nx-1] - uu[2]/(uu[1]+uu[2]+uu[3])*c2o]
        lbg.extend([-1e-8] * 1)
        ubg.extend([1e-8] * 1)

        lbw.extend(xp_0)
        ubw.extend(xp_0)
        w0.extend(xp_0)
        x_plotp += [Xpk]


        # Formulate the NLP
        m = 0
        DTk = SX.sym('DT_' + str(0), N)
        w += [DTk]
        lbw += [0.001]*N#[*dt[1, :]]#
        ubw += [100.]*N#[*dt[1, :]]#
        w0 += [*dt[1, :]]#[0.1]*(N+1)
        Ts = [DTk]
        for i in range(N):
            g += [DTk[i] - V/(uu[1]+uu[2]+uu[3])]
            lbg += [-1e-8]
            ubg += [1e-8]
        for k in range(N*ms):
            # New NLP variable for the control

            # -------------------
            #            DTk = SX.sym('DT_' + str(s))
            #            w += [DTk]
            #            lbw += [dt[k]]
            #            ubw += [dt[k]]
            #            w0 += [dt[k]]
            #    Ts += [DTk]
            h = DTk[m]/ms
            # --------------------
            # State at collocation points
            Xc = []
            Xpc = []
            for j in range(d):
                Xkj = SX.sym('X_' + str(s) + '_' + str(j), nx)
                Xc += [Xkj]
                w += [Xkj]
                lbw.extend(lbx)
                ubw.extend(ubx)
 #               ubw.extend([1.4])
                w0.extend([0]*nx)

            # Loop over collocation points
            Xk_end = D[0] * Xk
            for j in range(d):
                Xpkj = SX.sym('Xp_' + str(s) + '_' + str(j), nx * ntheta)
                Xpc += [Xpkj]
                w += [Xpkj]
                lbw.extend([-np.inf] * (nx * ntheta))
                ubw.extend([np.inf] * (nx * ntheta))
                w0.extend([0] * (nx * ntheta))

            # Loop over collocation points
            Xpk_end = D[0] * Xpk
            for j in range(1, d + 1):
                # Expression for the state derivative at the collocation point
                xp = C[0, j] * Xk
                xpp = C[0, j] * Xpk
                for r in range(d):
                    xp = xp + C[r + 1, j] * Xc[r]
                    xpp = xpp + C[r + 1, j] * Xpc[r]

                # Append collocation equations
                fj, qj, dxpj = f(Xc[j - 1], Uk*ur + np.array(ul), lbtheta, Xpc[j - 1])#
                g += [(h * fj - xp)]
                lbg.extend([-1e-8] * nx)
                ubg.extend([1e-8] * nx)

                g += [(h * dxpj - xpp)]
                lbg.extend([-1e-8] * (nx * ntheta))
                ubg.extend([1e-8] * (nx * ntheta))
                # Add contribution to the end state
                Xk_end = Xk_end + D[j] * Xc[j - 1]

                Xpk_end = Xpk_end + D[j] * Xpc[j - 1]
            #            if int(j1) < np.shape(t_meas)[0]:
            #                if np.real(k * T / N) == t_meas[j1]:
            #                    count[k] = 1
            #                    j1 += 1
            # Add contribution to quadrature function
            #      J = J + B[j]*qj*h

            # New NLP variable for state at end of interval
            Xk = SX.sym('X_' + str(s + 1), nx)
            w += [Xk]
            lbw.extend(lbx)
            ubw.extend(ubx)
 #           ubw.extend([1.4])
            w0.extend([0]*nx)
            x_plot += [Xk]
            Xpk = SX.sym('Xp_' + str(s + 1), nx * ntheta)
            w += [Xpk]
            lbw.extend([-np.inf] * (nx * ntheta))
            ubw.extend([np.inf] * (nx * ntheta))
            w0.extend([0] * (nx * ntheta))
            x_plotp += [Xpk]
            # Add equality constraint
            g += [Xk_end - Xk]
            lbg.extend([-1e-8] * nx)
            ubg.extend([1e-8] * nx)

            g += [Xpk_end - Xpk]
            lbg.extend([-1e-8] * (nx * ntheta))
            ubg.extend([1e-8] * (nx * ntheta))
            s += 1
            if divmod(k + 1, ms)[1] == 0:
                m += 1
                g += [Xk[0]]  # + 3*sqrt(vk[0])]
                lbg.extend([0.0])
                ubg.extend([0.1])
                mle = fim_for_single_t(Xpk, mle, sigma, nx, ntheta, 1)

            #            mle += maximum_likelihood_est(k_exp, Xk, x_meas, [1, 1, 1, 1], k)
            #            chi2 += maximum_likelihood_est(k_exp, Xk, x_meas, Xk + 0.0001, k)
            # Concatenate vectors
            # w = vertcat(*w)
            # g = vertcat(*g)
            # x_plot = horzcat(*x_plot)
            # u_plot = horzcat(*u_plot)
            # w0 = np.concatenate(w0)
            # lbw = np.concatenate(lbw)
            # ubw = np.concatenate(ubw)
            # lbg = np.concatenate(lbg)
            # ubg = np.concatenate(ubg)

            # Create an NLP solver
    #g += [np.ones([1, N]) @ DTk]
    #lbg.extend([0.5])
    #ubg.extend([2.09])
    mle = -(log(det(mle))+0.0001)
    problem = {'f': mle, 'x': vertcat(*w), 'g': vertcat(*g)}
    trajectories = Function('trajectories', [vertcat(*w)]
                                , [horzcat(*x_plot), horzcat(*u_plot), horzcat(*x_plotp), horzcat(*Ts)], ['w'], ['x',
                                                                                                                 'u',
                                                                                                                 'xp',
                                                                                                                 'Ts'])

    return problem, w0, lbw, ubw, lbg, ubg, trajectories



def construct_NLP_MBDoE_collocation_nom1(N_exp, f, x_0, xp_0, lbx, ubx, lbu, ubu, lbtheta, ubtheta, dt,
                                          N, uk, Vold, sigma, d, ms, c1o,c2o, V, Delta, Delta1, gp):
    nx = np.shape(x_0)[0]
    ntheta = np.shape(lbtheta)[0]
    nu = 4#np.shape(lbu)[0]
#    d = 8
    C, D, B = construct_polynomials_basis(d, 'radau')
    # Start with an empty NLP
    w = []
    w0 = []
    lbw = []
    ubw = []

    ss = np.zeros([8, 8])

    ss[:, 0] = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    ss[:, 1] = [0.5, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9, 0.9]
    ss[:, 2] = [0.2, 0.2, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9]
    ss[:, 3] = [1.0, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9, 0.9]
    ss[:, 4] = [0.1, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1]
    ss[:, 5] = [0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1]
    ss[:, 6] = [0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.1]
    ss[:, 7] = [0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9]

    A = np.zeros([8, 8,N_exp])
    for i in range(N_exp):
        A[:,:,i] = np.diag(ss[:,i])

    g = []
    lbg = []
    ubg = []

    # For plotting x and u given w
    x_plot = []
    x_plotp = []
    u_plot = []
    m_plot = []
    mle = Vold
    chi2 = 0
    s = 0

#    thetak = MX.sym('theta', np.shape(lbtheta)[0])
#    w += [thetak]

#    lbw.extend(lbtheta)
#    ubw.extend(ubtheta)

    ul = (np.array(ubu) + np.array(lbu))/2
    ur = (np.array(ubu) - np.array(lbu))/2

    lbu = [-1]*nu
    ubu = [1]*nu
 #   w0.extend(lbtheta)
    for k_exp in range(N_exp):
        if s > 0:
            s += 1
        # "Lift" initial conditions


        Uk = SX.sym('U_' + str(k_exp), nu)
        w += [Uk]
        lbw.extend(lbu)
        ubw.extend(ubu)
        w0.extend(((uk - np.array(ul)) / ur) )
        u_plot += [Uk]

        Xk = SX.sym('X_' + str(s), nx)
        w += [Xk]
        Xpk = SX.sym('Xp_' + str(s), nx * ntheta)
        w += [Xpk]

        lbw.extend([0.0])#[u_meas[k_exp][2]])
        lbw.extend([0] * (nx - 2))
        lbw.extend([0])

        ubw.extend([inf])#[u_meas[k_exp][2]])
        ubw.extend([0] * (nx - 2))
        ubw.extend([inf])

        w0.extend([0.0])#[u_meas[k_exp][2]])
        w0.extend([0] * (nx - 2))
        w0.extend([1.4])

        x_plot += [Xk]

        uu = Uk*ur + np.array(ul)

        g += [Xk[0] - uu[1]/(uu[1]+uu[2]+uu[3])*c1o]
        lbg.extend([-1e-8] * 1)
        ubg.extend([1e-8] * 1)

        g += [Xk[nx-1] - uu[2]/(uu[1]+uu[2]+uu[3])*c2o]
        lbg.extend([-1e-8] * 1)
        ubg.extend([1e-8] * 1)

        lbw.extend(xp_0)
        ubw.extend(xp_0)
        w0.extend(xp_0)
        x_plotp += [Xpk]


        # Formulate the NLP
        m = 0
        DTk = SX.sym('DT_' + str(0), N)
        w += [DTk]
        lbw += [0.001]*N#[*dt[1, :]]#
        ubw += [100.]*N#[*dt[1, :]]#
        w0 += [*dt[1, :]]#[0.1]*(N+1)
        Ts = [DTk]
        v_plot = []
        for i in range(N):
            g += [DTk[i] - V/(uu[1]+uu[2]+uu[3])]
            lbg += [-1e-8]
            ubg += [1e-8]
        for k in range(N*ms):
            # New NLP variable for the control

            # -------------------
            #            DTk = SX.sym('DT_' + str(s))
            #            w += [DTk]
            #            lbw += [dt[k]]
            #            ubw += [dt[k]]
            #            w0 += [dt[k]]
            #    Ts += [DTk]
            h = DTk[m]/ms
            # --------------------
            # State at collocation points
            Xc = []
            Xpc = []
            for j in range(d):
                Xkj = SX.sym('X_' + str(s) + '_' + str(j), nx)
                Xc += [Xkj]
                w += [Xkj]
                lbw.extend(lbx)
                ubw.extend(ubx)
 #               ubw.extend([1.4])
                w0.extend([0]*nx)

            # Loop over collocation points
            Xk_end = D[0] * Xk
            for j in range(d):
                Xpkj = SX.sym('Xp_' + str(s) + '_' + str(j), nx * ntheta)
                Xpc += [Xpkj]
                w += [Xpkj]
                lbw.extend([-np.inf] * (nx * ntheta))
                ubw.extend([np.inf] * (nx * ntheta))
                w0.extend([0] * (nx * ntheta))

            # Loop over collocation points
            Xpk_end = D[0] * Xpk
            for j in range(1, d + 1):
                # Expression for the state derivative at the collocation point
                xp = C[0, j] * Xk
                xpp = C[0, j] * Xpk
                for r in range(d):
                    xp = xp + C[r + 1, j] * Xc[r]
                    xpp = xpp + C[r + 1, j] * Xpc[r]

                # Append collocation equations
                fj, qj, dxpj = f(Xc[j - 1], Uk*ur + np.array(ul), lbtheta, Xpc[j - 1])#
                g += [(h * fj - xp)]
                lbg.extend([-1e-8] * nx)
                ubg.extend([1e-8] * nx)

                g += [(h * dxpj - xpp)]
                lbg.extend([-1e-7] * (nx * ntheta))
                ubg.extend([1e-7] * (nx * ntheta))
                # Add contribution to the end state
                Xk_end = Xk_end + D[j] * Xc[j - 1]

                Xpk_end = Xpk_end + D[j] * Xpc[j - 1]
            #            if int(j1) < np.shape(t_meas)[0]:
            #                if np.real(k * T / N) == t_meas[j1]:
            #                    count[k] = 1
            #                    j1 += 1
            # Add contribution to quadrature function
            #      J = J + B[j]*qj*h

            # New NLP variable for state at end of interval
            Xk = SX.sym('X_' + str(s + 1), nx)
            w += [Xk]
            lbw.extend(lbx)
            ubw.extend(ubx)
 #           ubw.extend([1.4])
            w0.extend([0]*nx)
            x_plot += [Xk]
            Xpk = SX.sym('Xp_' + str(s + 1), nx * ntheta)
            w += [Xpk]
            lbw.extend([-np.inf] * (nx * ntheta))
            ubw.extend([np.inf] * (nx * ntheta))
            w0.extend([0] * (nx * ntheta))
            x_plotp += [Xpk]
            # Add equality constraint
            g += [Xk_end - Xk]
            lbg.extend([-1e-8] * nx)
            ubg.extend([1e-8] * nx)

            g += [Xpk_end - Xpk]
            lbg.extend([-1e-8] * (nx * ntheta))
            ubg.extend([1e-8] * (nx * ntheta))


            s += 1
            if divmod(k + 1, ms)[1] == 0:
                m += 1

                muj, _, vj = gp(Uk * ur + np.array(ul), lbtheta, np.zeros([ntheta*nx,1]))  #
                muk_end = muj
                vk_end = vj

                muk = SX.sym('mu_' + str(k + 1), nx)
                w += [muk]
                mu1, _, _ = gp(uk, lbtheta, np.zeros([ntheta*nx,1]))  #
                m_plot += [muk]

                lbw.extend([-inf]*nx)
                ubw.extend([inf]*nx)
                w0.extend(np.array(mu1))  # [u_meas[k_exp][2]])

                vk = SX.sym('v_' + str(s), nx)
                w += [vk]

                lbw.extend([0.] * nx)
                ubw.extend([0.001]*nx)
                w0.extend([1e-5] * nx)
                v_plot += [vk]


                g += [muk_end - muk]
                lbg.extend([-1e-7] * nx)
                ubg.extend([1e-7] * nx)
                g += [vk_end - vk]
                lbg.extend([-1e-8] * nx)
                ubg.extend([1e-8] * nx)




                g += [Xk[0]+muk[0] + 3*sqrt((vk[0]+1e-8))]
                lbg.extend([0.0])
                ubg.extend([0.099])

                mle = fim_for_single_t(Xpk, mle, 1e-9+vk[:-1], nx, ntheta, 1)
                #mle = fim_for_single_t(Xpk, mle, sigma, nx, ntheta, 1)#np.array(sigma)**2+vk[:-1], nx, ntheta, 1)

                g += [sum1((Uk - (uk - np.array(ul)) / ur) ** 2) - Delta1]
                lbg.extend([-100])
                ubg.extend([0.0])
            #            mle += maximum_likelihood_est(k_exp, Xk, x_meas, [1, 1, 1, 1], k)
            #            chi2 += maximum_likelihood_est(k_exp, Xk, x_meas, Xk + 0.0001, k)
            # Concatenate vectors
            # w = vertcat(*w)
            # g = vertcat(*g)
            # x_plot = horzcat(*x_plot)
            # u_plot = horzcat(*u_plot)
            # w0 = np.concatenate(w0)
            # lbw = np.concatenate(lbw)
            # ubw = np.concatenate(ubw)
            # lbg = np.concatenate(lbg)
            # ubg = np.concatenate(ubg)

            # Create an NLP solver
    #g += [np.ones([1, N]) @ DTk]
    #lbg.extend([0.5])
    #ubg.extend([2.09])
    mle = -(log(det(mle))+0.0001)# + 0.1*sum1(vk.T@vk)
    problem = {'f': mle, 'x': vertcat(*w), 'g': vertcat(*g)}
    trajectories = Function('trajectories', [vertcat(*w)]
                                , [horzcat(*x_plot), horzcat(*u_plot), horzcat(*x_plotp),
                                   horzcat(*v_plot), horzcat(*m_plot), horzcat(*Ts)], ['w'], ['x',
                                                                                                                 'u',
                                                                                                                 'xp',
                                                                                                                 'v_opt',
                                                                                                                 'mu',
                                                                                                                 'Ts'])

    return problem, w0, lbw, ubw, lbg, ubg, trajectories



def MCMC_for_model(NS,theta,sigma_s, u_meas,N_exp, x_init, dt):
    from utilities_leeds import plant_model
    n_points = 30
    f, nu, nx, ntheta = plant_model('sensitivity')
    x_meas3 = np.zeros([N_exp + 10, nx[0], n_points + 1])
    xp_meas = np.zeros((ntheta[0] * nx[0], N_exp * n_points))
    mle = 0
    n_points = 30
    x_meas3 = np.zeros([NS,N_exp, nx[0], n_points + 1])
    theta1 = np.random.multivariate_normal(theta,sigma_s,NS)
    for mc in range(NS):
        pp = 0
        s = 0
        for k0 in range(N_exp):
            x11 = x_init[k0, :]  # change it
            x_meas3[mc,s, :, 0] = np.array(x11.T[:nx[0]])
            xp1 = np.zeros([nx[0] * ntheta[0], 1])
            for i in range(n_points):
                F = integrator_model(f, nu, nx, ntheta, 'embedded', 'sensitivity', dt[k0, 0] / n_points)
                Fk = F(x0=vertcat(x11, xp1), p=vertcat(u_meas[k0, :], theta1[mc]))

                x11 = Fk['xf'][0:nx[0]]
                xp1 = Fk['xf'][nx[0]:]
                # + np.random.multivariate_normal([0.] * nx[0], np.diag(np. square(sigma))).T
                x_meas3[mc, s, :, i + 1] = np.array(x11.T)
                pp += 1
            s += 1
    return x_meas3



def compute_Hessian(f, X):

    size  = np.shape(X)[1]
    sizey = np.shape(f(X)[0])[1]
    dmdx  = np.zeros([size, size])
    dsdx  = np.zeros([size, size])
    v = np.eye(size)

    for i in range(size):
        for j in range(size):
                X_left   = X.copy()
                X_right  = X.copy()
                X_c1     = X.copy()
                X_c2     = X.copy()

                X_left  +=  1e-4 * v[i,:] + 1e-4 * v[j,:]
                X_right += -1e-4 * v[i,:] - 1e-4 * v[j,:]
                X_c1    += -1e-4 * v[i,:] + 1e-4 * v[j,:]
                X_c2    +=  1e-4 * v[i,:] - 1e-4 * v[j,:]

#                dmdx[i, j] =  (GP(X_left)[0].reshape((1,-1,)) -2 * GP(X_c)[0].reshape((1,-1,)) + GP(X_right)[0].reshape((1,-1,)))/(2/1e-4)**2
                dsdx[i, j] =  (f(X_left)[1].reshape((1,-1,)) -f(X_c1)[1].reshape((1,-1,))-f(X_c2)[1].reshape((1,-1,)) + f(X_right)[1].reshape((1,-1,)))/(2 * 1e-4)**2
    return dmdx, dsdx
