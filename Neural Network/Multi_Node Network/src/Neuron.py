# -*- coding: utf-8 -*-

'''
Created on 28 Aug 2018

@author: Jack
'''
import numpy as np

class Neuron():
    def __init__(self, learning_rate, pos, layer):
        self.pos = pos
        self.layer = layer
        self.bias = 0
        self.alpha = learning_rate
        #later variables
        self.loss = 0
        self.db = 0
        self.weights = []
        self.softmax = 0
        self.y_hat = -1
        self.input_X = []
    def grad_calc(self, output_Y): #i = index
        self.dz = np.subtract(self.y_hat, output_Y)
        self.dw = np.dot(self.dz, np.transpose(self.input_X))
        self.db = np.sum(self.dz)
        return
    
    def grad_calc_hidden(self, weights2, dz2):
        self.dz = np.multiply(np.dot(weights2, dz2), d_tanh(self.z))
    def loss_accumulation(self):
        self.loss = self.loss / self.m
        self.dw = np.divide(self.dw, self.m)
        self.db /= self.m
        return
    
    def learningRules(self):
        self.weights = np.subtract(self.weights, np.multiply(self.alpha,self.dw))
        self.bias = self.bias - self.alpha*self.db
        return
    
    def set_weights(self, input_X):
        #self.input_X = input_X
        self.m = input_X.shape[0]
        self.weights = 2*(np.random.random_sample(self.m)-0.5)
        self.nx = np.prod(input_X.shape[1:]) #vector length
        
    def forward_prob(self, input_X, weights = 0):
        self.z = np.dot(self.weights, input_X) + self.bias
        self.y_hat = SigV(self.z)
        self.input_X = input_X
        
    def find_loss(self, output_Y):
        self.loss = np.sum(cross_entropy_Loss(output_Y, self.y_hat))
        self.loss = np.nan_to_num(self.loss) #replace nan values with 0
                
        
def cross_entropy_Loss(y, a):
   # loss is performed on entire vector inputs using numpy
   return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

              
def SigV(matrix): #return sigmoid matrix
    return np.divide(1,1+np.exp(-matrix))

def Sigmoid(z): #return sigmoid of value
    return 1/(1+np.exp(-z))
    
def vectorise(matrix): #input matrix, get (1,nx) vector
    n = np.prod(matrix.shape[1:])
    return np.reshape(matrix, (matrix.shape[0], n))

def tanh(z):
    return np.divide((np.subtract(np.exp(z),np.exp(-z)),np.add(np.exp(z),np.exp(-z))))

def d_tanh(z):
    return (1/((np.power(z,2))+1))



