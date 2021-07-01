# Expensive Black-Box Optimization for Chemical Engineering applications
This repository provides state-of-the-art black box optimization techniques for chemical engineering problems.

Here, we consider a constrained optimization problem of the following form:

<img src="https://latex.codecogs.com/svg.image?\begin{aligned}&space;&space;&space;&space;&space;&&space;\underset{\textbf{x}}{\text{min.}}&space;&space;&space;&space;&space;&&&space;f(\textbf{x},&space;\omega)&space;\\&space;&space;&space;&space;&space;&&space;\text{s.t.}&space;&space;&space;&space;&space;&space;&&&space;&space;g_k(\textbf{x},&space;\omega)&space;\leq&space;0&space;\text{&space;,&space;}&space;k&space;=&space;1,&space;\dots,&space;n_g&space;\\&space;&space;&space;&space;&space;&&&&space;\textbf{x}&space;\in&space;[\textbf{x}^L,&space;\textbf{x}^U]\\&space;&space;&space;&space;&space;&&&&space;\textbf{x}&space;\in&space;\mathbb{R}^{n_x}\\\end{aligned}&space;" title="\begin{aligned} & \underset{\textbf{x}}{\text{min.}} && f(\textbf{x}, \omega) \\ & \text{s.t.} && g_k(\textbf{x}, \omega) \leq 0 \text{ , } k = 1, \dots, n_g \\ &&& \textbf{x} \in [\textbf{x}^L, \textbf{x}^U]\\ &&& \textbf{x} \in \mathbb{R}^{n_x}\\\end{aligned} " />

This includes one black-box objective f, black-box constraints g<sub>k</sub>, continuous variables x, and input box-bounds x<sup>L</sup> and x<sup>U</sup>.
ω denotes the potential presence of stochasticity.

### Methods
The methods examined in this work come from the model-based, direct search and finite difference philosophies and involve well known implementations. For the existing implementations, we created a wrapper so that the solvers take the same input as our in-house implementations.

The algorithms implemented can be found in the *algorithms* folder and include:
- **Bayesian optimization**: In-house, model-based
- **CUATRO** (local and global): In-house, model-based
- **DIRECT-L** randomized: direct search, adapted from the [NLopt library](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/)
- **Newton, BFGS, Adam**: Finite differences, In-house
- **Py-BOBYQA**: model-based, as found [here](https://pypi.org/project/Py-BOBYQA/)
- **SQSnobfit**: model-based, as found [here](https://pypi.org/project/SQSnobFit/)
- **Nesterov**: smoothed finite differences, in-house
- **Simplex/Nelder-Mead**: direct, in-house

### Applications

The case studies are intended to present the behaviour of typical chemical/process engineering problems while remaining tractable.
These include two constrained test functions (*test_functions*) and five chemical engineering applications (*case_studies*):
- **Quadratic constrained**: 2-dimensional (2-d), deterministic and stochastic, convex
- **Rosenbrock constrained**: 2-d, deterministic and stochastic
- **Real-Time Optimization**: 2-d, deterministic and stochastic, constrained, convex
- **Self-Optimizing Reactor**: 2-d, deterministic and stochastic, constrained, convex
- **Model-Based Design of Experiments**: 4-d, determinsitic
- **PID controller tuning**: 4-d, deterministic and stochastic
- **Control Law Synthesis**: 10-d, stochastic

### Aim

The aim is to equip the practitioner with the information to choose a DFO algorithm based on their problem's size and how well-behaved, and stochastic their application is.

The subsequent comparison is based on the assumption that the black boxes that are queried are very expensive, meaning that the optimum should be found in the lowest number of function evaluations. Emphasis is placed on the discussion of how different algorithms can handle increasing levels of stochasticity inherent to the problems and on constraint satisfaction.

### Dependencies and Installation

- *environment.yml*: Lists all packages plus the version used. Using `conda env create --file environment.yml` in the anaconda prompt will recreate the dependencies

- *requirements.txt*: Lists all packages plus the version used. If the reader prefers pip, a virtual environment should be created and the author's dependencies recreated with `pip install -r requirements.txt`


### Implementation

The functions that are used are defined as follows:

```def problem(x):
    f = @(x) ..
    g1 = @(x) ..
    g2 = @(x) ..

    return f(x), [g1(x), g2(x)]
```

where the first output denotes the objective function, and where the second output denotes the list of constraint functions. If the problem has no black-box constraints, then the list should be empty `return f(x), []`.

The solver (wrappers) require only three essential inputs, the problem function, the initial guess, and the input bounds. Methods that rely on penalty functions to map the constraints also require the number of constraints. The function evaluation budget has a default value of 100.

```x0 = np.array([0.1, 0.2, 0.3]) # for 3-d problems
 bounds = np.array([[0,1], [0,1], [0,1]])
 max_f_eval = 100

# Example of solver call
 dictionary = SQSnobFitWrapper().solve(problem, x0, bounds, \
                                    maxfun = max_f_eval, constraints=2)
```


The input argument formats can change between solvers and examples of each use can be found within *problem_comparisons.py*  for each case study comparison *problem_comp* folder.

### Plots

The comparisons within each folder generate a number of plots, the most important of which are stored within the *Publication plots* folder of each case study.

For the real-time optimization case study for example, this folder contains the function evaluation versus number of function evaluations convergence plot solution bands for the model-based, and other methods for the deterministic and randomized case study.
For the random case, the model-based convergence plot looks as follows (`RTORand_Model.svg`):

<img align = center src="./RTO_comp/Publication plots/RTORand_Model.svg">

It also contains the convergence of the best methods in the 2-d solution space if applicable (`RTORand_2D_convergence_best.svg`):

<img align = center src="./RTO_comp/Publication plots/RTORand_2D_convergence_best.svg">

Finally, it contains box plots of how the optimality gap changes for the best methods at increasing levels of stochasticity if applicable (`RTO_feval50ConvergenceLabel.svg`):

<img align = center src="./RTO_comp/Publication plots/RTO_feval50ConvergenceLabel.svg">

Other plot folders of the RTO comparison contain detailed convergence plots for each method on each case study.
