## ---------------------------- ##
##
## Example student submission code for autonomous driving challenge.
## You must modify the train and predict methods and the NeuralNetwork class. 
## 
## ---------------------------- ##

import numpy as np
import cv2
from tqdm import tqdm
import time
from scipy import optimize

#New complete class, with changes:
class Neural_Network(object):
    def __init__(self, Lambda=0):        
        #Define Hyperparameters
        self.inputLayerSize = 60*64
        self.outputLayerSize = 64
        self.hiddenLayerSize = 30

        # Glorot Initialization
        limit = np.sqrt(6 / (self.inputLayerSize + self.hiddenLayerSize))
        self.W1 = np.random.uniform(-limit, limit, (self.inputLayerSize, self.hiddenLayerSize))
        limit = np.sqrt(6 / (self.hiddenLayerSize + self.outputLayerSize))
        self.W2 = np.random.uniform(-limit, limit, (self.hiddenLayerSize, self.outputLayerSize))
        
        #Regularization Parameter:
        self.Lambda = Lambda
        
    def forward(self, X):
        #Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3) 
        return yHat
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)/X.shape[0] + (self.Lambda/2)*(np.sum(self.W1**2)+np.sum(self.W2**2))
        return J
        
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        #Add gradient of regularization term:
        dJdW2 = np.dot(self.a2.T, delta3)/X.shape[0] + self.Lambda*self.W2
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        #Add gradient of regularization term:
        dJdW1 = np.dot(X.T, delta2)/X.shape[0] + self.Lambda*self.W1
        
        return dJdW1, dJdW2
    
    #Helper functions for interacting with other methods/classes
    def getParams(self):
        #Get W1 and W2 Rolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    def setParams(self, params):
        #Set W1 and W2 using single parameter vector:
        W1_start = 0
        W1_end = self.hiddenLayerSize*self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], \
                             (self.inputLayerSize, self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], \
                             (self.hiddenLayerSize, self.outputLayerSize))
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

def train(path_to_images, csv_file):
    '''
    First method you need to complete. 
    Args: 
    path_to_images = path to jpg image files
    csv_file = path and filename to csv file containing frame numbers and steering angles. 
    Returns: 
    NN = Trained Neural Network object 
    '''
    

    # You may make changes here if you wish. 
    # Import Steering Angles CSV
    data = np.genfromtxt(csv_file, delimiter = ',')
    frame_nums = data[:,0]
    steering_angles = data[:,1]

    train_im = np.empty((1500, 3840))
    for i in range(1500):
        im_full = cv2.imread(path_to_images + '/' 
                        + str(int(frame_nums[i])).zfill(4) + '.jpg')
        im_full = cv2.resize(im_full, (60, 64),interpolation = cv2.INTER_AREA)
        im_full = cv2.cvtColor(im_full, cv2.COLOR_BGR2GRAY)
        im_full = im_full/255
        im_full = np.ravel(im_full)
        train_im[i] = im_full
    # test_im = np.empty((500, 3840))
    # for i in range(1000,1500):
    #     im_full = cv2.imread(path_to_images + '/' 
    #                     + str(int(frame_nums[i])).zfill(4) + '.jpg')
    #     im_full = cv2.resize(im_full, (60, 64),interpolation = cv2.INTER_AREA)
    #     im_full = cv2.cvtColor(im_full, cv2.COLOR_BGR2GRAY)
    #     im_full = np.ravel(im_full)
    #     test_im[i-1000] = im_full

    angle_bins = np.zeros((1500, 64))
    bins = np.linspace(-180,180,64)
    pos = np.digitize(steering_angles, bins)
    stand = np.zeros((1,64))
    stand[0][27:36]=[0.1,0.32,0.61,0.89,1,0.89,0.61,0.32,0.1]
    for i in range(1500):
        angle_bins[i] = np.roll(stand,pos[i]-32)

    train_an = angle_bins
    # test_an = angle_bins[1000:1500]

    # Train your network here. You'll probably need some weights and gradients!
    X = train_im
    y = train_an
    NN=Neural_Network(Lambda=0.0001)
    num_iterations = 2000
    #alpha = 1e-4
    alpha = 1e-3
    beta1= 0.4
    beta2= 0.97
    epsilon= 1e-08

    grads =  NN.computeGradients(X = X, y = y)
    m0 = np.zeros(len(grads)) #Initialize first moment vector
    v0 = np.zeros(len(grads)) #Initialize second moment vector
    t = 0.0

    losses = [] #For visualization
    mt = m0
    vt = v0

    for i in tqdm(range(num_iterations)):
        t += 1
        grads = NN.computeGradients(X = X, y = y)
        mt = beta1*mt + (1-beta1)*grads
        vt = beta2*vt + (1-beta2)*grads**2
        mt_hat = mt/(1-beta1**t)
        vt_hat = vt/(1-beta2**t)

        params = NN.getParams()
        new_params = params - alpha*mt_hat/(np.sqrt(vt_hat)+epsilon)
        NN.setParams(new_params)
        losses.append(NN.costFunction(X = X, y = y))        
    return NN


def predict(NN, image_file):
    '''
    Second method you need to complete. 
    Given an image filename, load image, make and return predicted steering angle in degrees. 
    '''
    im_full = cv2.imread(image_file)
    im_full = cv2.resize(im_full, (60, 64),interpolation = cv2.INTER_AREA)
    im_full = cv2.cvtColor(im_full, cv2.COLOR_BGR2GRAY)
    im_full = im_full/255
    im_full = np.ravel(im_full)
    T=NN.forward(im_full)
    bins = np.linspace(-180,180,64)
    return bins[np.argmax(T)]