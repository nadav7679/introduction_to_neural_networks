
exec(open("network.py").read())



# Question 1 - Test Network with 100 cells
N = 100
net_hundred = Network(N)



memories = []
num_of_mems = np.zeros(N)
overlap_vals = np.zeros(N)

# -------------- Plotting success (overlap values) vs number of stored memories --------------#

net_hundred.clear()
for i in range(100):
    mem0 = np.random.randint(-50,50,size=N)
    mem = arrayToCell(mem0)
    
    memories.append(mem)
    net_hundred.insertMemory(mem)
    
    num = np.random.randint(len(memories))
    corrupt = noiseItUp(memories[num], 0.35)   # Corrupting some random memory by 35%.
    net_hundred.setNet(corrupt)   # Setting the corrupted memory as the initial state of the Networ.
    
    for j in range(6):
        net_hundred.quickUpdate()
        
    
    
    num_of_mems[i] = net_hundred.numOfMem
    overlap_vals[i] = overlap(net_hundred.net, memories[num])
    




plt.plot(num_of_mems,overlap_vals,"r-o",markersize=4,linewidth=0.9)
plt.grid()
plt.xlabel("Number of memories")
plt.ylabel("Success (%)")
net_hundred.maxMem
plt.savefig(fname = "success rate 100-net2")
plt.show()



# ------------- Plotting Iterations vs number of memories -------------- #
iterations = []
memories = []
num_of_mems = []

net_hundred.clear()

for i in range(50):
    count = 0
    mem0 = np.random.randint(-50,50,size=N)
    mem = arrayToCell(mem0)
    
    memories.append(mem)
    net_hundred.insertMemory(mem)
    
    num = np.random.randint(len(memories))
    corrupt = noiseItUp(memories[num], 0.35)   # Corrupting some random memory by 35%.
    net_hundred.setNet(corrupt)   # Setting the corrupted memory as the initial state of the Networ.
    
    while (net_hundred.net != memories[num]):
        net_hundred.update()
        count +=1
        
        if count > 2000:
            break
        
    
    
    num_of_mems.append(net_hundred.numOfMem)
    iterations.append(count)




plt.plot(num_of_mems, iterations)
plt.xlim((0,18))
plt.xlabel("Number of memories")
plt.ylabel("Time-steps needed")
plt.grid()
plt.savefig(fname = "Time vs patterns")
plt.show()



# ------------- Plotting Success vs Noise -------------- #

net_hundred.clear()
percentage = np.arange(0,1,0.01)
memories = []
overlap_vals = np.zeros(len(percentage))



for i in range(15):
    mem0 = np.random.randint(-50,50,size=N)
    mem = arrayToCell(mem0)
    
    memories.append(mem)
    net_hundred.insertMemory(mem)

num = np.random.randint(len(memories))   #Choosing one random memory to corrupt and check

for i in range(len(percentage)):
    
    corrupt = noiseItUp(memories[num], percentage[i])   # Corrupting some random memory by varing degrees of noise.
    net_hundred.setNet(corrupt)   # Setting the corrupted memory as the initial state of the Networ.

    for j in range(7):
        net_hundred.quickUpdate()

    overlap_vals[i] = overlap(net_hundred.net, memories[num])


    



plt.plot(percentage*100,overlap_vals*100)
plt.grid()
plt.xlabel("Noise Percentage (%)")
plt.ylabel("Restoration success (%)")
plt.title("Net with 15 memories")
plt.savefig(fname = "noise dependece 15")
plt.show()




# ------------- Testing determenistic changes using my makeVid function -------------- #

# This patch of code creates videos without any effort!
net_hundred.clear()
memories = []


for i in range(10):
    mem0 = np.random.randint(-50,50,size=N)
    mem = arrayToCell(mem0)
    
    memories.append(mem)
    net_hundred.insertMemory(mem)
    
num = np.random.randint(len(memories))
corrupt_mat = arrayToMatrix(np.copy(memories[num]),10,10)

for i in [3,6]:
    for j in range(10):
        corrupt_mat[i][j] *= -1
        if i != j:
            corrupt_mat[j][i] *= -1
net_hundred.setNet(matrixToArray(corrupt_mat))
makeVid(net_hundred, test = memories[num], name = "zigzag", title = "Shti Va'erev")
    

    


#####  Question 2 - MNIST



file0 = "MNIST files/t10k-images.idx3-ubyte"
file1 = "MNIST files/t10k-labels.idx1-ubyte"
data = idx2numpy.convert_from_file(file0)
labels = idx2numpy.convert_from_file(file1)
N = 28

mnist_net = Network(N*N)


# -------------- Plotting success (overlap values) vs number of stored memories --------------#


mnist_net.clear()
memories = []
num_of_mems = np.zeros(10)
overlap_vals = np.zeros(10)
count = 0

while count <10:
   
    num = np.random.randint(len(labels))
    if labels[num] == count:
        loadMatToNet(mnist_net,data[num])
        memories.append(arrayToCell(matrixToArray(data[num])))
        
        num = np.random.randint(len(memories))
        corrupt = noiseItUp(memories[num], 0.2)   # Corrupting some random memory by 35%.
        mnist_net.setNet(corrupt)   # Setting the corrupted memory as the initial state of the Networ.
    
        for j in range(6):
            mnist_net.quickUpdate()
        
    
    
        num_of_mems[count] = mnist_net.numOfMem
        overlap_vals[count] = overlap(mnist_net.net, memories[num])        
        count += 1

    
    




plt.plot(num_of_mems,overlap_vals,"r-o",markersize=4,linewidth=0.9)
plt.grid()
plt.xlabel("Number of memories")
plt.ylabel("Success (%)")
plt.savefig(fname = "success rate mnist_net1")
plt.show()


###### Question 4


# --------  Testing digits from different handwritings than the one that's stored in the net  ---------#


overlap_vals = []

                  
mnist_net.clear()   
numbers = np.random.randint(0,10,3)
data_index = [0,0,0]

for j in range(len(numbers)): # This loop picks three random digit patterns and inserts them to the Net. Then, it randomly picks a different
                              # pattern of one of stored digits, and cheks if the Net can retrive the initial digit.
    
    data_index[j] = np.random.randint(len(labels))
    while labels[data_index[j]] != numbers[j]:
        data_index[j] = np.random.randint(len(labels))
        
    loadMatToNet(mnist_net,data[data_index[j]])
    

# At this point I have 3 random digits in the network, their data-indexes are saved in data_index



digit = numbers[1]   #Arbitrary choise.
pattern_index = 0
while labels[pattern_index] != labels[data_index[1]] or pattern_index == data_index[1]:
    pattern_index = np.random.randint(len(labels))

mnist_net.setNet(arrayToCell(matrixToArray(data[pattern_index])))

makeVid(mnist_net, name = "restore6", test = arrayToCell(matrixToArray(data[data_index[1]])), updates = int(3.5*N*N) ,title = "3 mem's") 




###### Question 3

# -------------- Testing retrival of shifted digits --------------#


N = 32*32
digits_net = Network(N)
digits_net.clear()
numbers = [
jpgToMatrix("digits/0.jpg"),
jpgToMatrix("digits/1.jpg"),
#jpgToMatrix("digits/2.jpg"),
#jpgToMatrix("digits/3.jpg"),
#jpgToMatrix("digits/4.jpg"),
#jpgToMatrix("digits/5.jpg"),
#jpgToMatrix("digits/6.jpg"),
jpgToMatrix("digits/7.jpg"),
#jpgToMatrix("digits/8.jpg"),
#jpgToMatrix("digits/9.jpg"),
]






shifted_one = np.copy(numbers[1])
for i in range(0,32):
    for j in range(32):
        if i <5: 
            shifted_one[i][j] = 1
        else:
            shifted_one[i][j]= numbers[1][i-5][j]
        
for i in range(len(numbers)):
    loadMatToNet(digits_net, numbers[i])
    
digits_net.setNet(matrixToArray(shifted_one))
makeVid(digits_net , name = "shifted one 3 mems", title = "shifted one - 3 mem's")






