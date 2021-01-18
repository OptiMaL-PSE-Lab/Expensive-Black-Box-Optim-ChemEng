# Bayesian_opt_Pyro
This repo aims to solve Bayesian optimization using [Pyro](https://pyro.ai/) and [pytorch simple](https://pytorch.org/). The objective function and the constraints can be defined in numpy format. The repository implements the following: 

**Unconstrained Optimization**

The unconstrained optimization can minimize one of the following acquisition functions: mean, Lower confidence bound or negative expected improvement. 
The default solver here is BDGS via Pyro. 

**Constrained Optimization**

The constrained optimization can minimize one of the following acquisition functions: mean, Lower confidence bound or negative expected improvement. The constraints can be satisfied with respect to the mean (*probabilistic to be included*)
[Casadi](https://web.casadi.org/) is used and ipopt. 

## Installation

```bash
git clone https://github.com/UCL/Bayesian_opt_Pyro.git
```
Additional packages needed 
```bash
pip install casadi 
pip3 install pyro-ppl
pip install sobol_seq
pip install pyDOE
```

## Options for solver
 The value depited is the default one.
 
***objective: (REQUIRED)***           *Objective to be minimized*
 
***xo***                      *initial point. It is not required*

***bounds (REQUIRED)***       *Bounds for the decision variable*
 
***maxfun=20***                 *Number of iterations*
 
***N_initial=4***                *Number of initial points*
 
***select_kernel='Matern52'***    *Kernel for Gaussian process*

***acquisition='LCB'***           *Acquisition function*

***casadi=False***                *Solve the problem via casadi and ipopt (this is used for constrained problems*

***constraints = None***          *No constraints by defaults*

***probabilistic=False***        *To be implemented for probabilistic constraints*

***print_iteration=False***       *Print iterations*


## Example
```python
from Bayesian_opt_Pyro.utilities_full import BayesOpt
assert pyro.__version__.startswith('1.5.1')
pyro.enable_validation(True)  # can help with debugging
pyro.set_rng_seed(1)


def f1(x):
    return (6 * x[0] - 2)**2 * np.sin(12 * x[0] - 4)
def g1(x):
    return (6 * x[0] - 2)  - 1

lower = np.array([0.0]*1)
upper = np.array([1.]*1)

solution1 = BayesOpt().solve(f1, [0], bounds=(lower,upper), acquisition='LCB', print_iteration=True, constraints=[g1])


print(solution1)
```

