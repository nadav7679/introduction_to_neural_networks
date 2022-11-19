

# Hopfield model - Network class
## By Nadav Porat
# ------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import idx2numpy
from scipy.special import erfc
from PIL import Image
import matplotlib.animation as animation
from IPython.display import HTML
get_ipython().run_line_magic('matplotlib', 'inline')




# -------------- Helpful functions ------------------#
def heaviside(value):
    if value > 0:
        return 1
    else:
        return -1
    

def arrayToCell(arr): # A small helpfull function that makes Cell arrays from int arrays
    res = []
    for i in arr:
        res.append(heaviside(i-0.2))
    return res



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

    
def overlap(arr1,arr2):

    if not len(arr1) == len(arr2):
        print ("Error - The configuration you have entered needs to be a Cell array with length = N")
        return


    summ = 0
    for i in range(len(arr1)):
        summ += arr1[i] * arr2[i]

    return summ/len(arr1)


def noiseItUp(arr, p = 0.3):
    N = len(arr)
    amount = int(p*N)
    res = np.copy(arr)
    
    for i in range(amount):
        res[np.random.randint(N)] *= -1
    return res


def jpgToMatrix(path):
    im = Image.open(path)

    im_np = np.asarray(im)
    try:
        im_np = im_np[:, :, 0]
    except IndexError:
        pass
    im_np = np.where(im_np<128, -1, 1)
    
    return im_np



# -------------- The Network Class ------------------#


class Network:
   
    def __init__(self,N,error=0.01): # Creats a Neuronal network with N neurons and maximum error.
        
                                     # N += 1   # IMPORTENT:  I add one artificial Cell to the Network,
        netTemp = []                 # its job is to prevent the system from going in the "Inverted" direction
        weights = np.zeros((N,N))
        
        for i in range(N):
            netTemp.append(1)
        
        self.length = N
        self.net = netTemp
        self.maxError = error
        self.weights = weights
        self.numOfMem = 0
        self.pError = 0
        self.maxMem = 0
        
        prob = 0
        while  prob < self.maxError:
            self.maxMem += 1
            prob = 0.5*erfc(np.sqrt(N/(2*self.maxMem)))   # Calculating the maximum amount of memories that are allowed
                                                          #to enter the Network
            
    
    def insertMemory(self,mem):   # Inserting a memory to the system, mem needs to be an N lengthed array
        self.numOfMem += 1
        N = self.length
        
        if len(mem) != len(self.net):   # Dealing with wrong enteries
            print ("Error - The memory you have entered needs to be a Cell array with length = N") 
            return
        
    #    if self.numOfMem != 1:
    #        self.weights *= (self.numOfMem-1)     # This line's intent is to remove the previous normalization.
                                             #You can imagine how annoyed iv'e beeb when I realized I missed this petty detial at first.
        for i in range(len(mem)):
            for j in range(len(mem)):
                if i == j: 
                    self.weights[i][j] = 0
                
                else:
                    self.weights[i][j] +=  (mem[i] * mem[j])/N   # Adding in the Generalized-Hebb-rule weights 
      
    #    self.weights *= 1/self.numOfMem   # Re-normalization
                    
                    
        
        
        self.pError = 0.5*erfc(np.sqrt(self.length/(2*self.numOfMem)))   # Recalculating the current error of the Network with the new memory
        
    def update(self):   # Updating a single random Cell in the network
        
        omega = self.weights
        N = self.length
        net = self.net
        num = np.random.randint(0,N)
        
        
        summ = 0
        for i in range(N):
            summ += omega[num][i]*net[i]   # Weighted sum
            
        self.net[num] = heaviside(summ)   # Updating the num Cell
        
        
        
    def quickUpdate(self):   # Updating the system N times (N number of Cells)
        for i in range(self.length):
            self.update()
            
            
    def printNet(self):
        
        for i in range(self.length):
            if i == 0:
                print("[%d" %self.net[i], end="")
            elif i == self.length-1:
                print(", %d]" %self.net[i])   # Just trying to make it pretty :)
            else:
                print(", %d" %self.net[i],end="")
                
    
    def setNet(self,config):   # Setting some configuration to the system
        
        if not len(config) == self.length:
            print ("Error - The configuration you have entered needs to be a Cell array with length = N")
            return
        
        for i in range(self.length):
            self.net[i] = heaviside(config[i])
            
    
    def clear(self):
        N = self.length
                                            
        netTemp = []                 
        weights = np.zeros((N,N))
        
        for i in range(N):
            netTemp.append(1)
        
        self.net = netTemp
        self.weights = weights
        self.numOfMem = 0
        self.pError = 0
        

    
    def energy(self):
        summ = 0
        
        for i in range(self.length):
            for j in range(self.length):
                if i == j: continue
                summ += self.weights[i][j]*self.net[i]*self.net[j]
                
        return (-1/2)*summ
    
    def crosstalk(self,mem, num = -100):
        
# The memory inserted (mem) must be a memory that is already embedded in the Network (otherwise the result has no meaning)
# The crosstalk is calculated for a single cell denoted with num. If a cell is not given, one is chosen randomly.

        N = self.length
        if num == -100 :  num = np.random.randint(0,N)
        weights = self.weights
        summ = 0

        if not len(mem) == self.length:
            print ("Error - The configuration you have entered needs to be a Cell array with length = N")
            
        for i in range(N):
            for j in range(N):
                weights[i][j] -= mem[i]

            summ += weights[num][i]*mem[i]    
        
        return summ*mem[num]*(-1)
            
            
        
        
        
    

# -------------- Network assisting functions ------------------#

def loadMatToNet(Net,mat):
    mem = matrixToArray(mat)
    mem1 = arrayToCell(mem)
    Net.insertMemory(mem1)
    



def makeVid(net,updates = -1,test = None, name = None, title  = None):
    if updates == -1: updates = 5*net.length
    ims = []
    length = int(np.sqrt(net.length))
    step = 5*(1+int(length/100))   # The step size is taken from trial and error
    fig = plt.figure(figsize=(6,6))

    
    for i in range(updates):
      
        if i%step == 0:
        
            if test != None:
                net_copy = np.copy(net.net)
                
                for j in range(net.length):
                    
                    if net_copy[j] != test[j]:
                        net_copy[j] = -0.2
                        
                if title != None: plt.title(title)
                mat = arrayToMatrix(net_copy,length,length)
                im = plt.imshow(mat, animated=True,cmap="gray")
                plt.axis("off")

            else:

                if title != None: plt.title(title)
                mat = arrayToMatrix(net.net,length,length)
                im = plt.imshow(mat, animated=True,cmap="gray")
                plt.axis("off")

            ims.append([im])

        net.update()


    ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True,repeat_delay=500)
    HTML(ani.to_html5_video())
    
    if name != None:
        ani.save(name + ".mp4")

    return

