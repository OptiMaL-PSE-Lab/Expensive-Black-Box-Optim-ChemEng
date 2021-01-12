'''
Exact Reference: 

Simionescu, P.A.; Beale, D. (September 29 – October 2, 2002). 
New Concepts in Graphic Visualization of Objective Functions (PDF). 
ASME 2002 International Design Engineering Technical Conferences 
and Computers and Information in Engineering Conference. 
Montreal, Canada. pp. 891–897. Retrieved 7 January 2017.
'''

def rosenbrock_f(x):
    '''
    Unconstrained Rosenbrock function (objective)
    '''
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

def rosenbrock_g1(x):
    '''
    Rosenbrock cubic constraint
    g(x) <= 0 
    '''
    return (x[0] - 1)**3 - x[1] + 1


def rosenbrock_g2(x):
    '''
    Rosenbrock linear constraint
    g(x) <= 0 
    '''
    return x[0] + x[1] - 1.8