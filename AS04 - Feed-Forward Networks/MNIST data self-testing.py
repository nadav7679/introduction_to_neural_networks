"""

Test yourself on your ability to recognise MNIST digits.
Run the program, look at the image showed to you and enter a number from 0 - 9.
Do it a hundred times and get a histogram with your results.

"""


exec(open("FF Network.py").read())

# In[2]:


file0 = "MNIST files/train-images.idx3-ubyte"
file1 = "MNIST files/train-labels.idx1-ubyte"
file2 = "MNIST files/t10k-images.idx3-ubyte"
file3 = "MNIST files/t10k-labels.idx1-ubyte"
data = idx2numpy.convert_from_file(file0) / 255      # The devison by 255 is in order to normalize the values to be
test_data = idx2numpy.convert_from_file(file2) / 255 # between 0 and 1
labels = idx2numpy.convert_from_file(file1)
test_labels = idx2numpy.convert_from_file(file3)


# In[3]:


def error(output, answer):
    res = np.zeros(10)
    res[answer] = 1
    
    error = np.max ( np.abs(res - output ) )
    
    return error


# In[4]:


N = 100
errors = np.zeros(N)
for i in range(N):

    num = np.random.randint(len(test_labels))
    plt.imshow(test_data[num])
    plt.show()
    output = input()
    arr = np.zeros(10)
    arr[int(output)] = 1
    err = error(arr, test_labels[num])
    errors[i] = err


# In[10]:


plt.hist(errors, bins = 50, color = 'm', edgecolor = 'black')
plt.title("Testing Myself - Error histogram")
plt.xlabel("Error")
plt.ylabel("Number of incidents")
plt.grid()
#plt.savefig("literally myHistogram")

