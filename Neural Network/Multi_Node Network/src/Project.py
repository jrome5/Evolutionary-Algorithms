# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 23:43:52 2018

@author: jackr
"""
import csv
import numpy as np
def open_csv(filepath):
    with open(filepath + ".csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        matrix = []
        answers = []
        for row in csv_reader:
            if(line_count > 1000):
                return np.asarray(matrix), np.asarray(answers)
            if line_count > 1:
                matrix.append(np.asarray(row[1:]).astype(float))
                answers.append(int(row[0]))
            line_count += 1
        print(f'Processed {line_count} lines.')
    return np.asarray(matrix), np.asarray(answers)

def softmax(matrix, n, length):
    total = np.zeros(length)
    for i in range (length):
        total[i] = np.sum(np.exp(n.y_hat))
    return np.divide(np.exp(n.y_hat), total)

def init_layer(N, alpha, layername):
    layer = []
    for i in range(N):
        n = Neuron(alpha, i, layername)
        layer.append(n)
    return np.asarray(layer)
if __name__ == '__main__':
    '''Get Inputs'''
    trainX = np.genfromtxt("fashionmnist/fashion-mnist_train.csv", delimiter = ",")
    trainX = np.delete(trainX, 0, axis=0)
    trainY = np.array(trainX[:,0])
    trainX = np.delete(trainX, 0, axis = 1)
    
    testX = np.genfromtxt("fashionmnist/fashion-mnist_test.csv", delimiter = ",")
    testX = np.delete(testX, 0, axis=0)
    testY = np.array(testX[:,0])
    testX = np.delete(testX, 0, axis = 1)
    #trainX, trainY = open_csv("fashionmnist/fashion-mnist_train")
    #testX, testY = open_csv("fashionmnist/fashion-mnist_test")
    from Neuron import Neuron
    '''Normalize matrices'''
    trainX = trainX/255
    testX = testX/255
    Y = np.zeros((len(trainY),10))
    for i in range(len(trainY)):
        Y[i][int(trainY[i])] = 1
    '''Define first hidden layer'''
    h1 = init_layer(16, 0.05, "hidden1")
    h2 = init_layer(16, 0.05, "hidden2")
    output = init_layer(10, 0.05, "output")
    '''Set up weights'''
    for i in h1:
        i.set_weights(trainX[0])
    for i in h2:
        i.set_weights(h1)
    for i in output:
        i.set_weights(h2)
    '''Probagate'''
    J = []
    loss = 1000
    j = 0
    while(loss > 1e-2):
        '''Forward Probagate 1 layer at a time'''
        for n in h1:
            n.forward_prob(np.transpose(trainX))
        print("first hidden prob")
        a1 = np.array([i.y_hat for i in h1])
        for n in h2:
            n.forward_prob(a1)
        print("second hidden prob")
        a2 = np.array([i.y_hat for i in h2])        
        for n in output:
            n.forward_prob(a2)
        print("output prob")
        a3 = np.array([i.y_hat for i in output])
        '''Backward Probagate 1 layer at a time'''
        k = 0
        for n in output:
            sm = softmax(a3, n, trainX.shape[0])
            #n.backward_prob(Y[:,k])
            n.find_loss(Y[:,k])
            n.grad_calc(Y[:,k])
            n.loss_accumulation()
            n.learningRules()
            k = k + 1
        print("output back prob")
        for n in h2:
            n.backward_prob()
        j = j + 1
        
    
    