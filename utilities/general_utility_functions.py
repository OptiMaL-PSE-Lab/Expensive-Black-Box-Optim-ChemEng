#------------------------------------------------
#  This file contains extra functions to perform
#  additional operations needed everywhere
#  e.g. Create objective with penalties.
#  Generate initial points for the model-based methods
import matplotlib.pyplot as plt
import matplotlib
import functools
import warnings
import numpy as np
def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + '\n'

class PenaltyFunctions:
    """
    BE CAREFULL TO INITIALIZE THIS FUNCTION BEFORE USE!!
    IT TAKES HISTORICAL DATA WITH IT
    """
    def __init__(self, f_total, type_penalty='l2', mu=100):
        self.f = f_total
        #self.g     = g

        self.f_his = []
        self.g_his = []

        self.type_p = type_penalty
        self.aug_obj = self.augmented_objective(mu)

    def create_quadratic_penalized_objective(self, mu, order, x):

        funcs = self.f(x)
        obj   = funcs[0]
        card_of_funcs = len(funcs[1])+1
        if type(obj) == float:
            self.f_his += [obj]
        else:
            self.f_his += [obj.copy()]
        n_con = card_of_funcs-1
        g_tot = np.zeros(n_con)
        for i in range(n_con):
            g_tot[i] = funcs[1][i]
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




def plot_generic(output):
    csfont = {'fontname': 'Times New Roman'}

    # plt.rcParams['font.sans-serif'] = "Arial"
    plt.rcParams['font.family'] = "Times New Roman"

    # grid_shape = (1, 2)
    # fig = plt.figure()
    ft = int(12)
    font = {'size': ft}
    plt.rc('font', **font)
    plt.rc('text', usetex=True)
    params = {'legend.fontsize': 15,
              'legend.handlelength': 2}

    No_algs       = len(output)

    g_store       = []
    x_store       = []
    f_store       = []
    N_evals       = []
    g_best_so_far = []
    f_best_so_far = []
    x_best_so_far = []
    TR            = []
    for i in range(No_algs):
        g_store       += [output[i]['g_store']]
        x_store       += [output[i]['x_store']]
        f_store       += [output[i]['f_store']]
        N_evals       += [output[i]['N_evals']]
        g_best_so_far += [output[i]['g_best_so_far']]
        f_best_so_far += [output[i]['f_best_so_far']]
        x_best_so_far += [output[i]['x_best_so_far']]
        TR            += [output[i]['TR']]


    colors = ['#A8383B',  '#226765','#AA6B39', '#328A2E']
    labels = ['pyBOBYQA', 'BayesOpt', 'Penalty based BayesOpt']
    fig, ax = plt.subplots(1, 1)
    for i in range(No_algs):
        iters  = np.linspace(1,N_evals[i],N_evals[i])
        plt.step(iters, f_best_so_far[i], color=colors[i], label=labels[i], marker='None')

    plt.xlabel('Evaluations')
    plt.ylabel(r'Best $f$ so far')

    ax.set_yscale('log')
    ax.tick_params(right=True, top=True, left=True, bottom=True)
    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis='y', which='minor',direction="in")

    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.tick_params()
    ax.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.xticks(np.linspace(1,N_evals[0],N_evals[0]))
    plt.legend()
    plt.tight_layout()
