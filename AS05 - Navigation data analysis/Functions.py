"""
Neural Networks AS05 - Data analysis
By: Nadav Porat ; 207825506

"""

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.special import gammaincc


# In[2]:



def load_da_data(filename):   # loading the data
    from scipy.io import loadmat
    
    data = loadmat(filename)
    
    posx = np.array(data["posx"][:,0])
    posy = np.array(data["posy"][:,0])
    direction = np.array(data["headDirection"][:,0])

    try:
        
        spikes1 = np.array(data["spiketrain1"][:,0])
        spikes2 = np.array(data["spiketrain2"][:,0])
        return posx, posy, spikes1, spikes2, direction

        
    except KeyError:
        
        spikes = np.array(data["spiketrain"][:,0])
        return posx, posy, spikes, direction


def find_spikes(posx, posy, spikes, isZeros = False):   # returns spikes spatial pattern
    N = 599979
    if isZeros:
        spikes_loc = np.zeros((N, 2))
    else:
        spikes_loc = np.zeros((1,2))
        
    for i in range(len(spikes)):
        if i<len(posx) and i<len(posy):
            if spikes[i] == 1 and not isZeros:
                temp = np.array( [posx[i], posy[i]] )
                spikes_loc = np.vstack((spikes_loc, temp))

            elif spikes[i] == 1 and isZeros:
                spikes_loc[i][0], spikes_loc[i][1] = posx[i], posy[i]

    return spikes_loc


# In[3]:


def firerate_heatmap(posx, posy, spikes):   # returns a matrix with firerate values for each 2.33X2.33 [cm^2]
    N = 599979
    bins = 43
    length = 100/bins
    res = np.zeros((bins, bins))
    
    for i in range(1, bins+1):
        for j in range(1, bins+1):
            index_x1= np.where(length*(j-1) < posx ) 
            index_x2= np.where( posx < length*j)
            index_x = np.intersect1d(index_x1, index_x2)

            index_y1= np.where( posy < length*i)
            index_y2= np.where( length*(i-1) < posy )
            index_y = np.intersect1d(index_y1, index_y2)

            indexes = np.intersect1d(index_x, index_y)   # Now indexes is an array with the indexes of all of the point in the corresponding square. 
            summ = 0
            count = len(indexes)
            
            for k in range(count):
                if spikes[indexes[k]] == 1:
                    summ += 1
            
            
            if count != 0:
                res[i-1][j-1] = (summ*1000)/count
            else:
                res[i-1][j-1] = 0
        
    return res


# In[4]:


def headDir_fire(head_dir, spikes):   # returns an array with the firerate values in each 10 degrees set 
    head_dir *= 360/(2*np.pi)
    num_bins = 36
    bins = np.zeros(num_bins)
    
    for i in range(1, num_bins+1):
        indexes1 = np.where(head_dir < i*10)
        indexes2 = np.where(head_dir > (i-1)*10)
        indexes = np.intersect1d(indexes1, indexes2)
        
        summ = 0
        for index in indexes:
            if spikes[index] == 1:
                summ += 1
                
        if len(indexes) != 0:
            bins[i-1] = summ*1000/len(indexes)
        else:
            bins[i-1] = 0
    
    return bins


# In[5]:


def get_interspikes(spikes):   # returns the interspikes of the given spikes vec
    indexes = np.nonzero(spikes)[0]
    interspike = [indexes[0]]
    
    for i in range(1, len(indexes)):
        interspike.append( indexes[i] - indexes[i-1] )
        
    interspike = np.array(interspike)
    return interspike   # Notice that the interspikes are returned in miliseconds


def chi_square(expected, observed):
    N = len(expected)
    summ = 0
    
    for i in range(N):
        if expected[i] != 0:
            summ += ((observed[i] - expected[i])**2) / expected[i]
    
    return summ



# In[7]:


def get_random_spikes_square(data_interspikes, amount = 599979):   # returns random generated spikes and 
                                                                   # interspikes according to the mean
                                                                   # and std of the given interspikes
    N = len(data_interspikes)
    avg = np.mean(data_interspikes)
    std = np.std(data_interspikes)#/np.sqrt(N)
    interspikes = np.square(np.rint(norm.rvs(loc = avg, scale = std, size = N)))
    
    spikes = np.zeros(amount)
    index = 0
    for i in range(N):
        try:
            index += int(np.sqrt(interspikes[i]))
            spikes[index] = 1
            
        except IndexError:
            continue
            
    return spikes, interspikes    


def analyze_square(spikes, bins, num = 30):   # data analysis for the given spike. This combines the
                                              # previous functions to give an overall chi_red and p-val 
                                              # after averaging multiple iterations
    p = 0
    chi_red = 0
    
    for i in range(num):
        interspikes = get_interspikes(spikes)
        new_spikes, new_interspikes = get_random_spikes_square(interspikes)
        interspikes = np.square(interspikes)
        
        nu = len(bins)
        observed = np.histogram(interspikes, bins)[0]
        expected = np.histogram(new_interspikes, bins)[0]
        chi = chi_square(expected, observed)

        p += gammaincc (nu/2, chi/2)
        chi_red += chi/(nu-1)
        
    chi_red /= num
    p /= num
    return chi_red, p



# In[6]: These functions are pretty much leftovers. I used the square ones for the analysis.

def analyze(spikes, nu = 1000, num = 15):
    
    interspikes = get_interspikes(spikes)
    p = 0
    chi_red = 0
    
    for i in range(num):
        
        new_spikes, new_interspikes = get_random_spikes(interspikes)
        bins = np.arange(nu)
        
        observed = np.histogram(interspikes, bins)[0]
        expected = np.histogram(new_interspikes, bins)[0]
        chi = chi_square(expected, observed)

        p += gammaincc (nu/2, chi/2)
        chi_red += chi/(nu-1)
    
    chi_red /= num
    p /= num
    return chi_red, p

def get_random_spikes(data_interspikes, amount = 599979):
    N = len(data_interspikes)
    avg = np.mean(data_interspikes)
    std = np.std(data_interspikes)#/np.sqrt(N)
    interspikes = np.abs(np.rint(norm.rvs(loc = avg, scale = std, size = N-1)))
    
    spikes = np.zeros(amount)
    index = 0
    for i in range(N):
        try:
            index += int(interspikes[i])
            spikes[index] = 1
            
        except IndexError:
            continue
            
    return spikes, interspikes    