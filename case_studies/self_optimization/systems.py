from casadi import *


class Static_PDE_reaction_system:
    # Parameters

    def __init__(self):



        self.theta = np.array([np.log(57.9 * 60. * 10. ** (-2)), 33.3 / 10, np.log(2.7 * 60. * 10. ** (-2)), 35.3 / 10,
                  np.log(0.865 * 60. * 10. ** (-2)), 38.9 / 10, np.log(1.63 * 60. * 10. ** (-2)), 44.8 / 10])
        self.DH    = 0.
        self.D1    = 0.25
        self.cp    = 100
        self.rho   = 1000
        self.kappa = 0.59
        self.UA    = 5*10**5
        self.umax  = np.array([150, 10])
        self.umin  = np.array([40, 0.5])

        A = np.array([[-1,1,0,0,-1],
                      [-1,0,1,0,-1],
                      [0,-1,0,1,-1],
                      [0,0,-1,1,-1]]).T
        self.A     = A
        self.x0    = [0.5, 0.,0.,0.,1.]
        self.N     = 200
        self.dx    = 1/self.N
        self.xd, self.xa, self.u, self.ODEeq, self.Aeq, self.states, self.algebraics, self.inputs = self.DAE_system()
        self.eval = self.integrator_system()

    def df(self,f, i):
        dfdx = (f[i + 1] - f[i - 1]) / (2 * self.dx)
        return dfdx
    def df1(self,f, i):
        dfdx = (f[i] - f[i - 1]) / (self.dx)
        return dfdx

    def d2f(self,f, i):
        d2fdx2 = (f[i + 1] - 2 * f[i] + f[i - 1]) / (self.dx ** 2)
        return d2fdx2

    # def f1(self,u, T, Da):
    #     f1 = Da * (1 - u) * exp(T / (1 + T / self.gamma))
    #     return f1
    #
    # def f2(self,u, T, Tw, Da):
    #     f2 = self.B * Da * (1 - u) * exp(T / (1 + T / self.gamma)) + self.beta * Tw
    #     return f2

    def Arrhenius(self,theta, T):
        k = exp(theta[0::2] - theta[1::2] * 1e4 / 8.314 * (1 / (T + 273.15) - 1 / (90 + 273)))
        return k

    def reaction_rate(self, T, x):
        A = self.A
        k = self.Arrhenius(self.theta, T)
        r = []  # MX.sym('r', (A.shape[1],1))
        for i in range(A.shape[1]):
            r1 = x[0] ** np.heaviside(-A[0, i], 0) * k[i]
            for j in range(1, A.shape[0]):
                if A[j, i] != 0:
                    r1 = x[j] ** np.heaviside(-A[j, i], 0) * r1
            r = vertcat(r, r1)
        xdot = A @ r  # +\
        return xdot


    def DAE_system(self):
        # Define vectors with names of states
        states = ['x']
        nd = len(states)
        xd = SX.sym('xd', nd)
        for i in range(nd):
            globals()[states[i]] = xd[i]

        # Define vectors with names of algebraic variables
        algebraics = ['X']*self.N*6
        na = len(algebraics)
        xa = SX.sym('xa', na)
        for i in range(na):
            globals()[algebraics[i]] = xa[i]

        inputs = ['Tw', 'Da']
        nu = len(inputs)
        u = SX.sym("u", nu)
        for i in range(nu):
            globals()[inputs[i]] = u[i]

        # Reparametrization
        # reaction rate
        x_sm    = xa[:self.N]
        x_ortho = xa[self.N:2*self.N]
        x_para  = xa[2*self.N:3*self.N]
        x_bis   = xa[3*self.N:4*self.N]
        x_morth = xa[4*self.N:5*self.N]
        T       = xa[5*self.N:6*self.N]
        A = self.A
        x_conc = [x_sm, x_ortho, x_para, x_bis, x_morth]

        x_res = np.zeros((self.N*6, 1))
        Aeq = []
        index = 0
        for j in range(5):
            for i in range(self.N):
                if i==0:
                    x_res[index,0] = self.D1*(x_conc[j][1] - x_conc[j][0])/self.dx\
                                 + u[1]*(self.x0[j] - x_conc[j][0])
                    Aeq      += [self.D1*(x_conc[j][1] - x_conc[j][0])/self.dx\
                                 + u[1]*(self.x0[j] - x_conc[j][0])]
                elif i ==self.N-1:
                    x_res[index,0] = (x_conc[j][self.N-1] - x_conc[j][self.N-2])
                    Aeq      += [(x_conc[j][self.N-1] - x_conc[j][self.N-2])]
                else:
                    x_res[index,0] = self.D1 * self.d2f(x_conc[j], i)\
                                 - u[1]*self.df(x_conc[j], i)\
                                 + self.reaction_rate(
                        T[i],[x_sm[i], x_ortho[i], x_para[i], x_bis[i], x_morth[i]])[j]
                    Aeq += [self.D1 * self.d2f(x_conc[j], i)\
                                 - u[1]*self.df(x_conc[j], i)\
                                 + self.reaction_rate(
                        T[i],[x_sm[i], x_ortho[i], x_para[i], x_bis[i], x_morth[i]])[j]]
                index +=1
        for i in range(self.N):

            Aeq += [ T[i] - (-(u[0]-60)*exp(-i*self.dx*(self.UA/(self.cp*self.rho*u[1])))+u[0])]
        # Define vectors with banes of input variables

        ODEeq = [0 * x]

        # Declare algebraic equations


        return xd, xa, u, ODEeq, Aeq, states, algebraics, inputs

    def integrator_system(self):
        """
        This function constructs the integrator to be suitable with casadi environment, for the equations of the model
        and the objective function with variable time step.
        inputs: NaN
        outputs: F: Function([x, u, dt]--> [xf, obj])
        """

        xd, xa, u, ODEeq, Aeq, states, algebraics, inputs = self.DAE_system()
        VV = Function('vfcn', [xa, u], [vertcat(*Aeq)], ['w0', 'u'], ['w'])
        solver = rootfinder('solver', 'newton', VV)#, {'error_on_fail':False})

        return solver

    def objective(self, u, initial_sol=np.array([0.])):
        u_new = u*(self.umax-self.umin) + self.umin
        if initial_sol.all()==np.array([0.]):
            inintial = np.array([*np.zeros([self.N * 5])+0.1,*np.array([u_new[0]] * self.N)])
        else:
            inintial = initial_sol

        x = self.eval(inintial, u_new)+ 0.001*np.random.randn(inintial.shape[0])
        obj = -log(x[self.N*2-1])# + 0.5 * np.random.normal(0., 1)

        return obj.toarray()[0,0]

    def objective_noise_less(self, u, initial_sol=np.array([0.])):
        u_new = u*(self.umax-self.umin) + self.umin
        if initial_sol.all()==np.array([0.]):
            inintial = np.array([*np.zeros([self.N * 5])+0.1,*np.array([u_new[0]] * self.N)])
        else:
            inintial = initial_sol

        x = self.eval(inintial, u_new)

        obj = -log(x[self.N*2-1])
        return obj.toarray()[0,0]

    def constraint1(self, u, initial_sol=np.array([0.])):
        u_new = u*(self.umax-self.umin) + self.umin
        if initial_sol.all()==np.array([0.]):
            inintial = np.array([*np.zeros([self.N * 5])+0.1,*np.array([u_new[0]] * self.N)])
        else:
            inintial = initial_sol

        x = self.eval(inintial, u_new)
        pcon1 = x[4*self.N-1]/0.025-1+ 0.01*np.random.randn(self.N)

        return pcon1.toarray()[0,0]


    def constraint1_noise_less(self, u, initial_sol=np.array([0.])):
        u_new = u*(self.umax-self.umin) + self.umin
        if initial_sol.all()==np.array([0.]):
            inintial = np.array([*np.zeros([self.N * 5])+0.1,*np.array([u_new[0]] * self.N)])
        else:
            inintial = initial_sol

        x = self.eval(inintial, u_new)
        pcon1 = x[4*self.N-1]/0.025-1

        return pcon1.toarray()[0,0]

    def constraint2(self, u, initial_sol=np.array([0.])):
        u_new = u*(self.umax-self.umin) + self.umin
        if initial_sol.all()==np.array([0.]):
            inintial = np.array([*np.zeros([self.N * 5])+0.1,*np.array([u_new[0]] * self.N)])
        else:
            inintial = initial_sol

        x = self.eval(inintial, u_new)
        pcon1 = x[5*self.N-1]/0.6-1

        return pcon1.toarray()[0,0]

    def constraint_agg_1(self, u, initial_sol=np.array([0.])):
        u_new = u*(self.umax-self.umin) + self.umin
        if initial_sol.all()==np.array([0.]):
            inintial = np.array([*np.zeros([self.N * 5])+0.1,*np.array([u_new[0]] * self.N)])
        else:
            inintial = initial_sol

        x = self.eval(inintial, u_new)
        con  = x[3*self.N:4*self.N]/0.025-1 + 0.01*np.random.randn(self.N)
        rho  = 200
        gmax = np.max(con)

        exp_term = exp((con-gmax)*rho)
        KS  = gmax + 1/rho * log(sum1(exp_term))

        pcon1 = KS
        return pcon1.toarray()[0,0]

    def constraint_agg_1_noiseless(self, u, initial_sol=np.array([0.])):
        u_new = u*(self.umax-self.umin) + self.umin
        if initial_sol.all()==np.array([0.]):
            inintial = np.array([*np.zeros([self.N * 5])+0.1,*np.array([u_new[0]] * self.N)])
        else:
            inintial = initial_sol

        x = self.eval(inintial, u_new)
        con  = x[3*self.N:4*self.N]/0.025-1
        rho  = 200
        gmax = np.max(con)

        exp_term = exp((con-gmax)*rho)
        KS  = gmax + 1/rho * log(sum1(exp_term))

        pcon1 = KS
        return pcon1.toarray()[0,0]




    def constraint_agg_2(self, u, initial_sol=np.array([0.])):
        u_new = u*(self.umax-self.umin) + self.umin
        if initial_sol.all()==np.array([0.]):
            inintial = np.array([*np.zeros([self.N * 5])+0.1,*np.array([u_new[0]] * self.N)])
        else:
            inintial = initial_sol

        x = self.eval(inintial, u_new)
        con  = x[4*self.N:5*self.N]/0.6-1
        rho  = 200
        gmax = np.max(con[:])
        exp_term = exp((con-gmax)*rho)
        KS  = gmax + 1/rho * log(sum1(exp_term))

        pcon1 = KS

        return pcon1.toarray()[0,0]
