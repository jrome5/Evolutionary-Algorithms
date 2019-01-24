# -*- coding: utf-8 -*-
import numpy as np
from math import sin,cos,pi,sqrt, exp
class Particle():
    def __init__(self, init_pos, bounds):
        self.pos = init_pos #particle position, bounds assumed symmetic
        self.vel = np.random.uniform(-1.0,1.0,num_dimensions) #particle velocity
        self.best_pos = []
        self.cost = -1.0 #individual cost
        self.best_cost = -1.0 #best individual cost
        self.bounds = bounds
        self.informants = [] #result of best informant 
        
    def evaluate(self, costfn):
        self.cost = costfn(self.pos)
        #check if best
        if(self.best_cost == -1):#initial condition
            self.best_pos = self.pos
            self.best_cost = self.cost
            
        elif(self.cost <= self.best_cost):
            self.best_pos = self.pos
            self.best_cost = self.cost
    def update_infromants(self, informants):
        best = None
        if(best == None or informants[i].cost < best.cost):
            best = informants[i]
        self.informants = best
#        print(self.informants)
        return
    
    def update_velocity(self, swarm_best_pos):
        alpha = 0.1 #inertia weighting
        beta = 2 #cognative constant
        gamma = 1. #social constant
        delta = 1. #global constant
        #beta -> delta sum to 4 as suggested
        for i in range(0,num_dimensions):
            #get random weights
            a = np.random.uniform(0,alpha,1)
            b = np.random.uniform(0,beta,1)
            c = np.random.uniform(0,gamma,1)
            d = np.random.uniform(0,delta,1)
            #calculate velocity components
            cognative_vel = b*np.subtract(self.best_pos[i],self.pos[i])
            social_vel = c*np.subtract(self.informants.pos[i], self.pos[i])
            global_vel = d*np.subtract(swarm_best_pos[i], self.pos[i])
            #sum velocities
            self.vel[i] = np.add(np.multiply(a, self.vel[i]), cognative_vel)
            self.vel[i] = np.add(self.vel[i],social_vel)
            self.vel[i] = np.add(self.vel[i],global_vel)
        
    def update_pos(self):
        self.pos = np.add(self.pos, self.vel)
        self.pos = np.clip(self.pos,self.bounds[0], self.bounds[1]) #position does not fall out of limit

def PSO(costfn, init_pos, bounds, ni = 6, num_particles=100, maxiter=2000): #PSEUDO code from Page 57 of Essentials of MetaHeuristics book
    '''Parameters
    costfn = Cost function used for optimization
    init_pos = initial position of swarm, N dimensions
    bounds = boundary of the function
    ni = number of informants per particle
    num_particles = 100, maximum reccomended
    max_iter = 2000, large number for reliable results
    '''
    global num_dimensions
    num_dimensions = len(init_pos)
    
    swarm_best_cost = -1 #group best cost
#    swarm_best_pos = []
    
    #create swarm with particles inside the boundaries
    swarm = []
    best_values = []
    informants_list = [None]*num_particles #list of particle's informant particle indexes
    for i in range(0,num_particles): #populate swarm, informants
        swarm.append(Particle(init_pos,bounds))
        informants_list[i] = (np.random.randint(0,num_particles-1,ni))
        
    for p in range(0,maxiter): 
        for x in range(0,num_particles):
            swarm[x].evaluate(costfn) #find fitness
            if(swarm_best_cost == -1 or swarm[x].cost <= swarm_best_cost): #fittest particle has minimal cost
                swarm_best_cost = swarm[x].cost
                swarm_best_pos = swarm[x].pos
            informants = [swarm[a] for a in informants_list[x]] #informants array using index list
            swarm[x].update_infromants(informants)
            
        for x in range(0,num_particles):
            swarm[x].update_velocity(swarm_best_pos)
            swarm[x].update_pos()
        best_values.append(swarm_best_cost)
        print("best position = %s with cost %s" %(swarm_best_pos,swarm_best_cost))
    return best_values  #best cost per iteration
'''Benchmark Optimization Functions'''
def Sphere(x):
    #bounds -5,5
    total = 0
    for i in range(len(x)):
        total += x[i]**2
    return total

def Rastrigin(x):
    #bounds -5.12,5.12
    A = 10
    n = len(x)
    total = 0
    for i in range(n):
        total += x[i]**2 - A*cos(2*pi*x[i])
    return A*n + total

def Rosenbrock(x):
    #min at f(1,1,1...1) = 0
    total = 0
    for i in range(len(x)-1):
        total += ((100*(x[i]**2-x[i+1])**2+(x[i]-1)**2))
    return total
        
def Griewank(x):
    #min at f(0) = 0
    total = 0
    prod = 0
    for i in range(len(x)):
        total += x[i]**2/4000
        prod *= cos(x[i]/sqrt(i+1))+1
    return total - prod
def Cigar(x):
    #bounds -5,5
    total = 0
    for i in range(1, len(x)):
        total += x[i]**2
    total = x[0]**2 + 10*total
    return total
        
if __name__ == "__main__":
    initial = [5.0,5.0,5.0,5.0] #initial vales. same as GA
    bounds = [[-5.12,5.12], [-5.12, 5.12], [-5.12, 5.12], [-600, 600], [-5.12,5.12]] #function limits [min,max] for each dimension
    test_list = [Sphere, Rastrigin, Rosenbrock, Griewank, Cigar] #functions to be used as benchmarks. listed for loop use
    average = []
    for i in range(1):
        best_cost = []
        for j in range(1):
            best_cost.append(PSO(test_list[0],initial,bounds[i]))
        average.append(np.average(best_cost, axis=0))
    np.savetxt("PSO_finalros - alpha.csv", np.transpose(average), delimiter=",")