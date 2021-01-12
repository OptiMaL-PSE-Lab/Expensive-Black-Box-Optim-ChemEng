import numpy as np 
import matplotlib.pyplot as plt 
from itertools import combinations,product
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio 
import os

''' 
    Tom Savage, United Kingdom, Process Systems Engineering.
    22/11/2020
'''

# function to be optimized
def squared_exponential(x):
    return sum(x[i]**2 for i in range(len(x)))


def plot_simplex(simplex_collection,its):
    # getting number of simplexes in collection
    overall_sim_num = len(simplex_collection[:,0,0])
    
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    # iterating over all simplexes provided
    for sim_num in range(overall_sim_num):
        simplex = simplex_collection[sim_num,:,:]
        n = len(simplex[:,0]) # nodes in simplex
        # generating face combinations
        comb = np.array(list(combinations(np.arange(n),d)))[:-1,:]
        
       # iterating over faces 
        for i in range(n-1):
            face = np.zeros((n-1,d)) # allocating face array
            # defining each face
            for j in range(n-1):
                face[j,:] = simplex[comb[i,j]]
            # plotting each face with a scaled alpha 
            collection = Poly3DCollection(face,linewidths=1.5,\
                alpha=(1-(1-(sim_num/(overall_sim_num)))*0.75))
            collection.set_facecolor('tab:blue')
            collection.set_edgecolor('k')
            ax.add_collection3d(collection)
    # setting boundary limits
    ax.set_xlim3d(-5,5)
    ax.set_ylim3d(-5,5)
    ax.set_zlim3d(-5,5)
    
    # defining viewing rotation based on iteration
    ax.view_init(10,its*2)
    
    # plotting two wire-frame cubes to better show perspective
    r = [-5, 5]
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s-e)) == r[1]-r[0]:
            ax.plot3D(*zip(s, e), color="k")
    r = [-7, 7]
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s-e)) == r[1]-r[0]:
            ax.plot3D(*zip(s, e), color="k",alpha=0.1)
    plt.tight_layout()

    # saving each figure corresponding to iteration for 
    # later gif conversion 
    plt.savefig(str(its)+'.png')
    return 
    
    
def simplex_method(f,x0,its,bounds,plot):
    iterations = its
    d = len(x0) # dimension 
    f_range = (bounds[:,1] - bounds[:,0])*0.5 # range of initial simplex
    x_nodes = np.random.normal(x0,f_range,(d+1,d)) # creating nodes
    simplex_store = np.array([x_nodes])
    plot_simplex(simplex_store,0) # plot initial simplex
    f_nodes = np.zeros((len(x_nodes[:,0]),1)) # function value at each node
    # evaluating function 
    for i in range(d+1):
        f_nodes[i,:] = f(x_nodes[i,:])
        
    for its in range(1,iterations):
        sorted_nodes = np.argsort(f_nodes[:,0])
        best_nodes = x_nodes[sorted_nodes[:-1]]
        # centroid of all bar worst nodes
        centroid = np.mean(best_nodes,axis=0)
        # reflection of worst node
        x_reflected = centroid + (centroid - x_nodes[sorted_nodes[-1],:])
        f_reflected =  f(x_reflected) 
        # accept reflection? 
        if f_reflected < f_nodes[sorted_nodes[-2]] and \
            f_reflected > f_nodes[sorted_nodes[0]]:
                x_nodes[sorted_nodes[-1],:] = x_reflected
                f_nodes[sorted_nodes[-1],:] = f_reflected
        # try expansion of reflected then accept? 
        elif f_reflected < f_nodes[sorted_nodes[0]]:
                x_expanded = centroid + 2*(x_reflected-centroid)
                f_expanded = f(x_expanded)
                if f_expanded < f_reflected:
                    x_nodes[sorted_nodes[-1],:] = x_expanded
                    f_nodes[sorted_nodes[-1],:] = f_expanded
                else: # ...expansion worse so accept reflection 
                    x_nodes[sorted_nodes[-1],:] = x_reflected
                    f_nodes[sorted_nodes[-1],:] = f_reflected
        else: # all else fails, contraction of worst internal of simplex
            x_contracted = centroid + 0.5*(x_nodes[sorted_nodes[-1],:]-centroid)
            f_contracted = f(x_contracted)
            if f_contracted < f_nodes[sorted_nodes[-1]]:
                x_nodes[sorted_nodes[-1],:] = x_contracted
                f_nodes[sorted_nodes[-1],:] = f_contracted
        # add new simplex to store, repeat!
        simplex_store = np.append(simplex_store,[x_nodes],axis=0)
        
        if plot == True:
            plot_simplex(simplex_store[-3:],its)
            
    return 


f = squared_exponential
d = 3
its = 10 # number of iterations, frames, etc...
bounds = np.array([[-5,5] for i in range(d)])
x0 = np.random.uniform(-12,9,d)

plot = True
simplex_method(f,x0,its,bounds,plot)

# converting saved images to a GIF file. 

if plot == True:
    images = [] # creating image array
    for filename in range(its): # iterating over images
        # adding each image to the array 
        images.append(imageio.imread(str(filename)+'.png')) 
        # this then deletes the image file from the folder
        # os.remove(str(filename)+'.png')

        
    imageio.mimsave('simplex.gif', images) # this then saves the array of images as a gif
