'''
Created on 28 Aug 2018

@author: Jack
'''
import numpy as np
import matplotlib.pyplot as plt
import h5py

class Neuron():
    def __init__(self, input_X, output_Y, learning_rate):
        self.nx = np.prod(input_X.shape[1:]) #vector length
        self.input_X = vectorise(input_X)
        self.input_X = np.divide(self.input_X, 255) #normalise
        self.output_Y = output_Y
        self.m = input_X.shape[0] #numper of samples
        #weights
        self.weights = np.zeros(self.nx)
        self.y_hat = np.zeros(self.m)
        self.bias = 0
        self.alpha = learning_rate
        self.z = np.zeros(self.m)
        #later variables
        self.loss = 0
        self.dw = np.zeros(self.nx) #weight change of size nx
        self.db = 0
        self.dz = np.zeros(self.m)
       

    def grad_calc(self): #i = index
        self.dz = np.subtract(self.y_hat,self.output_Y[0])
        self.dw = np.dot(self.dz, self.input_X)
        self.db = np.sum(self.dz)
        return
    
    def loss_accumulation(self):
        self.loss = self.loss / self.m
        self.dw = np.divide(self.dw, self.m)
        self.db /= self.m
        return
    
    def learningRules(self):
        self.weights = np.subtract(self.weights, np.multiply(self.alpha,self.dw))
        self.bias = self.bias - self.alpha*self.db
        return

    def forward_prob(self):
        self.z = np.add(np.dot(self.weights, np.transpose(self.input_X)),self.bias)
        self.y_hat = SigV(self.z)
        self.loss = np.sum(cross_entropy_Loss(self.output_Y[0], self.y_hat))
        self.loss = np.nan_to_num(self.loss) #replace nan values with 0
        
    def backward_prob(self):
        self.grad_calc()
        self.loss_accumulation()
        self.learningRules()
        
def cross_entropy_Loss(y, a):
   # loss is performed on entire vector inputs using numpy
   return -np.add(np.multiply(y,np.log(a)),np.multiply(np.subtract(1,y), np.log(np.subtract(1,a))))

              
def SigV(matrix): #return sigmoid matrix
    return np.divide(1,1+np.exp(-matrix))

def Sigmoid(z): #return sigmoid of value
    return 1/(1+np.exp(-z))
    
def vectorise(matrix): #input matrix, get (1,nx) vector
    n = np.prod(matrix.shape[1:])
    return np.reshape(matrix, (matrix.shape[0], n))

if __name__ == '__main__':
    # Loading the dataset
    train_dataset = h5py.File('../trainCats.h5', "r")
    trainSetX = np.array(train_dataset["train_set_x"][:]) # your train set features
    trainSetY = np.array(train_dataset["train_set_y"][:]) # your train set labels
    trainSetY = trainSetY.reshape((1, trainSetY.shape[0]))
    
    # Loading the test set
    test_dataset = h5py.File('../testCats.h5', "r")
    testSetX = np.array(test_dataset["test_set_x"][:]) # your test set features
    testSetY = np.array(test_dataset["test_set_y"][:]) # your test set labels
    testSetY = testSetY.reshape((1, testSetY.shape[0]))
    
    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    m = trainSetX.shape[0] #size of dataset
    #initalize node
    alpha = 0.05 #low for smoother cost descent, high for rapid but quick regression
    neuron = Neuron(trainSetX, trainSetY, alpha)
    J = [] #list of losses for graph
    j = 0 #iteration count
    Loss = 1 #dummy value
    
    #Training
    while(Loss >1e-3 and j < 500): #very close to zero gradient or taken too long
        j = j + 1
        neuron.forward_prob()
        neuron.backward_prob()
        loss = neuron.loss
        J.append(loss)
    print(str(j) + " iterations")    
    
    #create cost graph
    plt.plot(np.asanyarray(J))
    plt.title("Costs with learning rate = " + str(alpha))
    plt.ylabel("Cost")
    plt.xlabel("Iterations")
    plt.show()# costValues array with the costs for each iteration
    
    #Testing
    Error = np.zeros(testSetX.shape[0])
    testX = vectorise(testSetX)
    testX = np.divide(testX, 255) #normalise
    y_hat = np.zeros(testSetX.shape[0])
    y_hat = np.add(np.dot(neuron.weights, np.transpose(testX)), neuron.bias)
    y_hat = SigV(y_hat)
    Error = np.absolute(np.subtract(y_hat, testSetY[0]))
    
    #Plot test error graph
    plt.plot(np.asanyarray(Error), "ro")
    plt.title("Error in test data. Learning Rate = " + str(alpha))
    plt.ylabel("Differnce in Yhat and Y")
    plt.xlabel("Test Number")
    plt.show()
