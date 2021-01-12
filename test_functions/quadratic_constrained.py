def quadratic_f(x):
    '''
    test objective 
    '''
    return x[0]**2 + 10 * x[1]**2 + x[0] * x[1]

def quadratic_g(x):
    '''
    test constraint 
    g(x) <= 0 
    '''
    return 1 - x[0] - x[1] 