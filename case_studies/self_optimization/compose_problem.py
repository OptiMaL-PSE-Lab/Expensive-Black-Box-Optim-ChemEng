from case_studies.self_optimization import systems
import numpy as np

def RTO(x):
    # x = extract_FT(x)
    plant = systems.Static_PDE_reaction_system()
    f = plant.objective
    g1 = plant.constraint_agg_1
    return f(x), [g1(x)]

def RTO_noiseless(x):
    # x = extract_FT(x)
    plant = systems.Static_PDE_reaction_system()
    f = plant.objective_noise_less
    g1 = plant.constraint_agg_1_noiseless
    return f(x), [g1(x)]

def RTO_end_point_constraint(x):
    # x = extract_FT(x)
    plant = systems.Static_PDE_reaction_system()
    f = plant.objective
    g1 = plant.constraint1
    return f(x), [g1(x)]

def RTO_end_point_constraint_noiseless(x):
    # x = extract_FT(x)
    plant = systems.Static_PDE_reaction_system()
    f = plant.objective_noise_less
    g1 = plant.constraint1_noise_less
    return f(x), [g1(x)]



#Initial points (you could ignore this)
X        = np.array([[0., 1.], [0.1, 1.], [0., 0.9]])#np.array([[0., 0.], [0.1, 0.], [0., 0.1]])
#Intil point
xo       = np.array([0.0,0.85])#np.array([0.0,0.05])


print('RTO: ', RTO([0,0.85]))
print('RTO_end_point_constraint', RTO_end_point_constraint([0,0.85]))
print('RTO_end_point_constraint_noiseless', RTO_end_point_constraint_noiseless([0,0.85]))
print('RTO_noiseless', RTO_noiseless([0,0.85]))

print(2)

