#------------------------------------------------
#  This file contains extra functions to perform
#  additional operations needed everywhere
#  e.g. Create objective with penalties.
#  Generate initial points for the model-based methods

import functools
import warnings
import numpy as np
def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + '\n'

class PenaltyFunctions:
    def __init__(self, f, g, type_penalty='l2', mu=100):
        self.f     = f
        self.g     = g

        self.f_his = []
        self.g_his = []

        self.type_p = type_penalty
        self.aug_obj = self.augmented_objective(mu)

    def create_quadratic_penalized_objective(self, mu, order, x):

        obj = self.f(x)
        self.f_his += [obj.copy()]
        n_con = len(self.g)
        g_tot = np.zeros(n_con)
        for i in range(n_con):
            g_tot[i] = self.g[i](x)
            obj += mu * max(g_tot[i], 0) ** order
        self.g_his += [g_tot]
        return obj

    def augmented_objective(self, mu):
        """

        :param mu: The penalized parameter
        :type mu: float
        :return:  obj_aug
        :rtype:   function
        """
        if self.type_p == 'l2':
            warnings.formatwarning = custom_formatwarning

            warnings.warn(
                'L2 penalty is used with parameter ' + str(mu))

            obj_aug = functools.partial(self.create_quadratic_penalized_objective,
                                        mu, 2)
        elif self.type_p == 'l1':
            warnings.formatwarning = custom_formatwarning

            warnings.warn(
                'L1 penalty is used with parameter ' + str(mu))

            obj_aug = functools.partial(self.create_quadratic_penalized_objective,
                                        mu, 1)
        else:
            mu_new = 100
            warnings.formatwarning = custom_formatwarning

            warnings.warn(
                'WARNING: Penalty type is not supported. L2 penalty is used instead with parameter ' + str(mu_new))

            obj_aug = functools.partial(self.create_quadratic_penalized_objective,
                                        mu_new, 2)
        return obj_aug