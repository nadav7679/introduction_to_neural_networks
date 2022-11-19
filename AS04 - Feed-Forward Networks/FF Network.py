"""
FeedForward network class, with biases and all dat good stuff.
By: Nadav Porat

"""



import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from IPython.core.debugger import set_trace
import idx2numpy

#---------- Activation function ----------#


def g(z):
    return 1.0/(1.0+np.exp(-z))

def g_prime(z):
    return g(z)*(1-g(z))




#---------- The FeedForward Network Class ----------#

class Network(object):
    
    def __init__(self, sizes, step = 0.3):
        
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.step = step
        self.days_in_gym = 0
        
        self.weights = []
        self.network = []
        self.deltas = []
        
        for layer in range(len(sizes) - 1):
           
            if layer == len(sizes)-2:
                
                W = 2*np.random.random((sizes[layer]+1, sizes[layer+1])) - 1  # Initilazing weights, the +1 is for a bias column
            else:    
                W = 2*np.random.random((sizes[layer] + 1, sizes[layer+1] +1)) - 1  # Initilazing weights, the +1 is for a bias column
            self.weights.append(W)
        
        for size in sizes:
            
            if size != self.sizes[-1]:
                delta = np.zeros(size + 1)
            
            else:
                delta = np.zeros(size)
            
            self.deltas.append(delta)
        
    def feedForward(self, a):
        
        self.network = []
        a = np.append(a, 1) # Adding a bias neuron to the input (biases always have the value 1)
        self.network.append(a)
        
        for m in range(1, self.num_layers):
            
            if m < self.num_layers-1:
                b = np.zeros(self.sizes[m] + 1)  # Again, this is the bias
                b[-1] = 1 
                
            else:
                b = np.zeros(self.sizes[m])   # The last layer (the output) shouldn't have a bias

            W = self.weights[m-1]
            for i in range(self.sizes[m]):
                b[i] = g( np.dot(a, W[:,i]) )
                
            self.network.append(b)
            a = np.copy(b)

        return a
    
   
    def backProp(self, inpt, expected):
        
        self.days_in_gym += 1
        output = self.feedForward(inpt)   # Propegating the input throughout the network
        
        for i in range(self.sizes[-1]):   # Setting the deltas for the last layer (the output)
            
            W = self.weights[-1]
            self.deltas[-1][i] = g_prime( np.dot(self.network[-2], W[:, i] ) ) * (expected[i] - output[i])
            
        
        for m in range(self.num_layers-1 , 1, -1):
            
            W = self.weights[m-1]   #W_(m-1 -> m)
            W_prev = self.weights[m-2]   # W_(m-2 -> m-1)
            
            for j in range(self.sizes[m-1] +1):
                h_prev = np.dot(self.network[m-2], W_prev[:, j])            
                h = np.dot( W[j, :], self.deltas[m])
                    
                self.deltas[m-1][j] = g_prime(h_prev) * h
            
        for m in range(1, self.num_layers):
            W = self.weights[m-1]

            for i in range(np.shape(W)[1]):
                W[:, i] += self.step * self.network[m-1] * self.deltas[m][i]   # Updating da' weights
                
    def clear(self):
        sizes = self.sizes
        self.weights = []
        self.network = []
        self.deltas = []

        for layer in range(len(sizes) - 1):

            if layer == len(sizes)-2:

                W = 2*np.random.random((sizes[layer]+1, sizes[layer+1])) - 1  # Initilazing weights, the +1 is for a bias column
            else:    
                W = 2*np.random.random((sizes[layer] + 1, sizes[layer+1] +1)) - 1  # Initilazing weights, the +1 is for a bias column
            self.weights.append(W)

        for size in sizes:

            if size != self.sizes[-1]:
                delta = np.zeros(size + 1)

            else:
                delta = np.zeros(size)

            self.deltas.append(delta)



#---------- Helpfull Functions ----------#


def get_training_set(mat, size):
    
    train_set = np.zeros((size,3))
    x_lim = np.shape(mat)[0]
    y_lim = np.shape(mat)[1]
    
    for i in range(size):
        x = np.random.randint(x_lim-1)
        y = np.random.randint(y_lim-1)
        train_set[i][0], train_set[i][1] = x, y 
        
        if mat[x][y] == 1:
            train_set[i][2] = 0
            
        else:
            train_set[i][2] = 1
    
    
    return train_set

def isInStar(pic, point):
    """  
    if type(point[0]) != int or type(point[1]) != int:
        
        point[0] *= np.shape(pic)[0]
        point[1] *= np.shape(pic)[1]
    """
    res = pic[point[0]][point[1]]
    
    if res == 1:
        return 0
    else:
        return 1
    

def matrixToArray(mat):
    arr = []
    for i in range(np.shape(mat)[0]):
        for j in range(np.shape(mat)[1]):        
            arr.append(mat[i][j])
    return arr


def arrayToMatrix(arr, n, m): # n is the numer of rows, m is the number of columes
    mat = np.zeros((n,m))
    count = 0
    
    for i in range(n):
        for j in range(m):
            mat[i][j] = arr[count]
            count += 1
    return mat


def jpgToMatrix(path):
    im = Image.open(path)

    im_np = np.asarray(im)
    try:
        im_np = im_np[:, :, 0]
    except IndexError:
        pass
    im_np = np.where(im_np<128, -1, 1)
    
    return im_np

    
def overlap(arr1,arr2):

    if not len(arr1) == len(arr2):
        print ("Error - The configuration you have entered needs to be a Cell array with length = N")
        return


    summ = 0
    for i in range(len(arr1)):
        summ += arr1[i] * arr2[i]

    return summ/len(arr1)

