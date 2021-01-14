import numpy as np 



def nesterov_random(f,x0,bounds,max_iter,constraints):
    d = len(x0)
    con_d = len(constraints)
    x_store = np.zeros((max_iter,d))
    g_store = np.zeros((max_iter,con_d))
    f_store = np.zeros(max_iter)
    x = x0
    f_eval_count = 0 
    alpha = 0.01
    mu = 0.1
    con_weight = 10000
    for i in range(max_iter):
        x_store[i,:] = x
        f_store[i] = f(x)
        
        u = np.random.normal(0,1,(1,d))
        B = np.array([[1,0],[0,1]])
        forw = (x + mu * u)[0,:]
        back = (x - mu * u)[0,:]
        
        f_forw = f(forw)
        f_eval_count += 1
        for i in range(len(constraints)):
            f_forw += con_weight * max(0,constraints[i](forw))
            
        f_back = f(back)
        f_eval_count += 1
        for i in range(len(constraints)):
            f_back += con_weight * max(0,constraints[i](back))
            
        g = ((f_forw - f_back)/(2*mu))*B@u.T
        x = x - alpha * (g.T)[0,:]
        
   # computing final constraint violation 
    con_viol = [0 for i in range(len(constraints))]
    act_cons = [0 for i in range(len(constraints))]
    for i in range(len(constraints)):
        con_val = constraints[i](x)
        con_viol[i] = con_val 
    con_viol_total = sum(max(0,con_viol[i]) for i in range(len(con_viol)))         
    output_dict = {}
    output_dict['x'] = x
    output_dict['f'] = f(x)
    output_dict['g'] = con_viol 
    output_dict['g_viol'] =  con_viol_total
    output_dict['g_store'] = g_store
    output_dict['x_store'] = x_store
    output_dict['f_store'] = f_store 
    output_dict['f_evals'] = f_eval_count
    return output_dict
    
