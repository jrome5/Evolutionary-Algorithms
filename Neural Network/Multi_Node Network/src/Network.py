# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

'''
Created on 28 Aug 2018

@author: Jack
'''
import numpy as np
from math import atan2
class Network():
    def __init__(self, sizes):
        self.num = len(sizes)
        self.sizes = sizes
        #weights
        self.biases = [0.05*np.random.randn(i, 1) for i in sizes[1:]]
        self.weights =[0.05*np.random.randn(j, i) for i, j in zip(sizes[:-1], sizes[1:])]
        self.z = [np.zeros(i) for i in sizes[1:]]
        self.dz = [] #derivative of z with respect to w
        self.da = [np.zeros(i) for i in sizes[1:]] #derivative of a with respect to z
        self.dc = [np.zeros(i) for i in sizes[1:]] #derivative of cost with resoect to a
        self.dw = [np.zeros((j,i)) for i, j in zip(sizes[:-1], sizes[1:])] #change in weights
        self.db = [np.zeros(i) for i in sizes[1:]] #change in biases
        self.a = [np.zeros(i) for i in sizes]
        self.loss = []
        self.lastloss = 1000 #dummy value
        self.test = []
        self.predictor_accuracy  = 0
        
    def propagate(self,train_input, train_output, alpha, method = "Sigmoid"):
        '''make random batch'''
        loss = []
        train_size = len(trainX)
        #batch_index = np.random.randint(0,len(train_output), batch_size)
        self.reset_params() #reset dz,dw and db
        for k in range(train_size):
        #for k in range(1):
            self.feed_forward(train_input[k], method)
            self.backprob(train_output[k], method)
            #print(self.a[-1]-train_output[k])
            loss.append(np.sum([(y-a)**2 for a,y in zip(self.a[-1],train_output[k])])) #square loss
#            loss.append([cross_entropy_Loss(a, y) for a,y in zip(self.a[-1],train_output[k])])
        '''adjust weights and biases'''
        for i in range(self.num-1):
            for j in range(self.sizes[i+1]):
                self.weights[i][j] = np.divide(np.subtract(self.weights[i][j], np.multiply(alpha,self.dw[i][j])),train_size)
                self.biases[i][j] = self.biases[i][j] - alpha*self.db[i][j]/train_size
        '''Calculate cost'''
        #print(loss)
        self.lastloss = np.sum(loss)  #euclidean loss
        self.loss.append(self.lastloss)
        return
    
    def feed_forward(self, input_x, method = "Sigmoid"):
        self.a[0] = input_x #set first row of 'a' to be the train input
        #'a' is shape (N, h1, output)
        #'weights' are shape (N, h1)
        for i in range(0, self.num-2, 1): #iterate layers
            for j in range(0,self.sizes[i+1],1): #iterate nodes in layer except output
                #print(i,j)
                self.z[i][j] = np.dot(self.weights[i][j], self.a[i]) + self.biases[i][j]
                self.a[i+1][j] = Activation(self.z[i][j], method)
        for j in range(self.sizes[-1]): #iterate output later for softmax
            self.z[-1][j] = np.dot(self.weights[-1][j], self.a[-2]) + self.biases[-1][j]
        for j in range(0,self.sizes[-1],1): #iterate output layer and calculate softmax
            #self.a[-1][j] = softmax(self.z[-1], self.z[-1][j])
            self.a[-1][j] = Activation(self.z[-1][j], method)    
    def backprob(self, y, method = "Sigmoid"): #Note, iterate i = layer, j = node
        #output layer
        #da = number, dc = number, dz = a[N-1] = [list]
        self.da[-1] = ActivationPrime(self.z[-1], method) 
        self.dc[-1] = -2*np.subtract(y,self.a[-1]) #derivative of cost
        for j in range(self.sizes[-1]):
            self.dw[-1][j] = self.dw[-1][j] + (self.a[-2]*self.dc[-1][j]*self.da[-1][j]) #derivative of activation* derivative of cost * deriv of weights 
            self.db[-1] = self.db[-1] + (self.dc[-1][j]*self.da[-1])
        #other layers
        for i in range(self.num-3, -1, -1):
            self.da[i] = ActivationPrime(self.z[i], method) 
            #self.dc[i] = 2*np.subtract(self.a[-1],y) #derivative of cost
            for j in range(self.sizes[i+1]):
                self.dw[i][j] = self.dw[i][j] + (self.a[i]*self.da[i][j]) #derivative of activation * deriv of prediction 
                self.db[i][j] = self.db[i][j] + (self.da[i][j])
        return
            
    def reset_params(self):
        self.da = [np.zeros(i) for i in self.sizes[1:]] #derivative of a with respect to z
        self.dc = [np.zeros(i) for i in self.sizes[1:]] #derivative of cost with resoect to a
        self.dz = [np.zeros(i) for i in self.sizes[0:]]
        self.dw = [np.zeros((j,i)) for i, j in zip(self.sizes[:-1], self.sizes[1:])]
        self.db = [np.zeros(i) for i in self.sizes[1:]]
        return
     
    def testing(self, testX, testY, indices):
        self.reset_params()
        for i in range(len(testX)):
            index = indices[i]
            self.feed_forward(testX[i])
            pred = 0
            for j in range(10):
                if(self.a[-1][pred] < self.a[-1][j]):
                    pred = j
            if(pred == index):
                self.predictor_accuracy = self.predictor_accuracy + 1
        return
    
def cross_entropy_Loss(y, a):
   # loss is performed on entire vector inputs using numpy
   if(a == 0): #adjust results that will break the log method
       a = 0.000000000001
   elif(a==1):
       a = 0.999999999999
   return -(y*np.log(a)+(1-y)*np.log(1-a))
   #return -np.add(np.multiply(y,np.log(a)),np.multiply(np.subtract(1,y), np.log(np.subtract(1,a))))


def Activation(z, method): #return sigmoid of value
    if(method =="Sigmoid"):
        return 1/(1+np.exp(-z)) #sigmoid
    elif(method =="tanh"):
        return (2/(1+(np.exp(-2*z))))-1#tanh
    else:
        return atan2(z)

def ActivationPrime(z, method): #differential of sigmoid
    if(method == "Sigmoid"):
        return Activation(z, method)*(1-Activation(z, method))
    elif(method =="tanh"):
        return 1-(Activation(z, method)**2)
    else:
        return 1/((z**2)+1)

def softmax(matrix, a):
    total = 0.0
    for i in range(len(matrix)):
        total = total + np.exp(matrix[i])
    return np.divide(np.exp(a), total)

def open_csv(filepath):
    import csv
    with open(filepath + ".csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        matrix = []
        answers = []
        for row in csv_reader:
            if(line_count > 5000):
                return np.asarray(matrix), np.asarray(answers)
            if line_count > 1:
                matrix.append(np.asarray(row[1:]).astype(float))
                answers.append(int(row[0]))
            line_count += 1
        print(f'Processed {line_count} lines.')
    return np.asarray(matrix), np.asarray(answers)
if __name__ == '__main__':
    '''Get Inputs'''
#    trainX = np.genfromtxt("fashionmnist/fashion-mnist_train.csv", delimiter = ",")
#    trainX = np.delete(trainX, 0, axis=0)
#    trainY = np.array(trainX[:,0])
#    trainX = np.delete(trainX, 0, axis = 1)
#    
#    testX = np.genfromtxt("fashionmnist/fashion-mnist_test.csv", delimiter = ",")
#    testX = np.delete(testX, 0, axis=0)
#    testY = np.array(testX[:,0])
#    testX = np.delete(testX, 0, axis = 1)
    trainX, trainY = open_csv("fashionmnist/fashion-mnist_train")
    testX, testY = open_csv("fashionmnist/fashion-mnist_test")
    '''Normalize matrices'''
    trainX = trainX/255
    testX = testX/255
    '''Convert Y to matrix'''
    Y = np.zeros((len(trainY),10))
    for i in range(len(trainY)):
        Y[i][int(trainY[i])] = 1.0
        
    tY = np.zeros((len(testY),10))
    for i in range(len(testY)):
        tY[i][int(testY[i])] = 1.0
    '''Network'''
    alpha = 0.01
    N = Network((len(trainX[0]), 16, 16, 10))
#    L = Network((len(trainX[0]), 4, 3, 10))
    M = Network((len(trainX[0]), 18, 16, 12, 10))
    O = Network((len(trainX[0]), 30, 10))
    iterations = 0 
    method = "Sigmoid" #Switch to "Sigmoid" for other experiment
    
    print("Training")
    while(iterations < 1000):
        N.propagate(trainX, Y, alpha, method)
#        L.propagate(trainX, Y, alpha, method)
        M.propagate(trainX, Y, alpha, method)
        O.propagate(trainX, Y, alpha, method)
        iterations = iterations + 1
#        if(iterations%100 == 0):
#            alpha = alpha*0.1
#            print(N.lastloss)

        alpha = alpha*np.exp(-0.000001*iterations)
#        print(Ns.lastloss, alpha)
        print(N.lastloss, M.lastloss, O.lastloss, alpha)
    import matplotlib.pyplot as plt
    #create cost graph
    plt.plot(np.asanyarray(N.loss))
    plt.title("Costs with learning rate = " + str(alpha))
    plt.ylabel("Cost")
    plt.xlabel("Iterations")
    plt.show()# costValues array with the costs for each iteration
    #Testing
    Error = np.zeros(testX.shape[0])
    #Plot test error graph
    plt.plot(np.asanyarray(N.test), "ro")
    plt.title("Error in test data. Learning Rate = " + str(alpha))
    plt.ylabel("Differnce in Yhat and Y")
    plt.xlabel("Test Number")
    plt.show()
    np.savetxt("data/N.csv", N.loss, delimiter = ",")
    np.savetxt("data/M.csv", M.loss, delimiter = ",")
#    np.savetxt("data/L.csv", L.loss, delimiter = ",")
    np.savetxt("data/O.csv", O.loss, delimiter = ",")
    