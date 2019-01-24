# -*- coding: utf-8 -*-

import numpy as np
from math import cos, pi, sqrt
class Chromosome:
    def __init__(self, initial, bounds):
        self.cost = -1.0
        self.pos = np.random.uniform(bounds[0], bounds[1], (len(initial))) #random position, bounds symmetic
        self.genes = len(initial)
        
    def evaluate(self, costfn):
        self.cost = costfn(self.pos)
        
def tournament(P): #selection (PSEUDO code from Essentials of metaheuristics)
    t = 5
    r = int(np.random.randint(0,len(P), 1))
    best = P[r]
    for i in range(0,t):
        r = int(np.random.randint(0,len(P), 1))
        n = P[r]
        if(n.cost < best.cost):
            best = n
    return best

def crossover(Pa, Pb, children_size=2): #two point crossover (PSEUDO code from Essentials of metaheuristics)
    l = Pa.genes
    c, d = np.random.randint(1,l,2)
    if(c>d):
        temp = c
        c = d
        d = temp
    if(c!=d):
        for i in range(c,d-1):
            temp = Pa.pos[i]
            Pa.pos[i] = Pb.pos[i]
            Pb.pos[i] = temp
    else:
        crossover(Pa, Pb)
    return Pa, Pb

def mutate(child, mutation_prob, bounds):
    l = child.genes
    for i in range(1,l):
        if(mutation_prob >= np.random.uniform(0,1,1)):
#            child.pos[i] = bounds[int(np.random.randint(0,1,1))] #boundary mutation
            child.pos[i] += 0.05*child.pos[i]*np.random.normal(0,1,1)
    child.pos = np.clip(child.pos, bounds[0], bounds[1]) #values to not exceed boundaries
    return child
#elitism, keep fitist so far
def GA(func1,initial, bounds, mutate_prob = 0.1,popsize=100, max_iter=2000):
    '''Parameters
    func1 = Cost function used for optimization
    initial = initial position of swarm, N dimensions
    bounds = boundaries of the function
    mutate_prob = 0.1, reccomendation
    popsize = 100, population size. maximum reccomended
    max_iter = 2000, large number for reliable results
    '''
    popsize
    P = []
    best = None
    better = None
    best_values = []
    for i in range(popsize):
        P.append(Chromosome(initial, bounds))
    
    for p in range(max_iter):
        for i in range(len(P)):
            P[i].evaluate(func1) #find fitness
            if(best == None or better == None or P[i].cost < best.cost): #determine best by lowest cost
                better = best
                best = P[i]
        Q = []
        Q.append(best) #keep best 2
        Q.append(better)
        for i in range(int((len(P)-1)/2)):
            Pa = tournament(P)
            Pb = tournament(P)
            Ca,Cb = crossover(Pa, Pb)
            Q.append(mutate(Ca, mutate_prob, bounds)) #mutate children
            Q.append(mutate(Cb, mutate_prob, bounds))
            #print(P[i].cost)
        P = Q
#        print("best cost = " + str(best.cost))
        print("best position = %s with cost %s" %(best.pos,best.cost))
        best_values.append(best.cost)
    return best_values
            

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
    initial = [5.0,5.0,5.0,5.0]
    bounds = [[-5.12,5.12], [-5.12, 5.12], [-5.12,5.12], [-600, 600], [-5.12,5.12]] #function limits [min,max] for each dimension
    test_list = [Sphere, Rastrigin, Rosenbrock, Griewank, Cigar]
    average = []
    for i in range(1):
        print(i)
        best_cost = []
        for j in range(1):
            best_cost.append(GA(test_list[i],initial,bounds[i]))
        average.append(np.average(best_cost, axis=0))
    np.savetxt("GAcsv", np.transpose(average), delimiter=",")