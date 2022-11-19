"""

All of the soultions to the given questions are in this file.

By: Nadav Porat

"""
exec(open("FF Network.py").read())





#---------- Question 1 - Star Picture ----------#


star = jpgToMatrix("star.jpg")
star = np.delete(star, [592, 593] , axis = 1 )  # Slight adjustment

star_net = Network([2, 6, 1], step = 0.5) 
star_preceptron = Network([2, 1], step = 0.5)




def test_star(network ,N = 50**4):

    tests = np.zeros(1)
    difference = np.zeros(1)
    
    training_set = get_training_set(star, N)
    training_set[:, 0] /= 592  # Normalizing the training_set coordinates to be between 0 and 1                  
    training_set[:, 1] /= 592  #(in respecet to the bottem left corner) 
    
    for i in range(N):
        data = training_set[i]
        network.backProp([data[0], data[1]], [data[2]])
        
        if i%100 == 0:
            
            point = np.random.randint(592, size = 2)
            difference = np.append(difference, abs(network.feedForward(point/592)[0] - isInStar(star, point)) )
            tests = np.append(tests, i)
    
    return tests, difference 
    




#The following loop takes about 20min to run so beware.
iGotTime = False
if iGotTime:

    num = 20
    for i in range(num):

        star_preceptron.clear()
        star_net.clear()

        if i == 0:

            tests, avg_precp_diff = test_star(star_preceptron)
            tests, avg_net_diff = test_star(star_net)

            avg_precp_diff /= num
            avg_net_diff /= num
        else:

            avg_precp_diff += test_star(star_preceptron)[1] /num
            avg_net_diff += test_star(star_net)[1] /num

    # save numpy array as csv file
    from numpy import asarray
    from numpy import savetxt
    """ save to csv file
    savetxt('tests_good.csv',  avg_precp_diff, delimiter=',')
    savetxt('avg_net_diff.csv', avg_net_diff, delimiter=',')
    savetxt('tests_good.csv', tests, delimiter=',')
    """


# --- Plotting the data --- #  

from numpy import genfromtxt

precp_data = genfromtxt("avg_precp_diff_good.csv", delimiter = ",")
layers_data = genfromtxt("avg_net_diff_good.csv", delimiter = ",")




plt.plot(precp_data[:,0], precp_data[:,1], label = "Preceptron")
plt.plot(layers_data[:,0], layers_data[:,1], label = "Multi-layered")
plt.grid()
plt.legend()
plt.title("Star Networks - Error vs training iterations (20 avg)")
plt.xlabel("Number of training Iterations")
plt.ylabel("Error")
#plt.savefig("Star Networks error 20 avg")


# --- Getting those nice "point in star" pictures  --- # 

star = jpgToMatrix("star.jpg")
N = 20**4
tests = np.arange(N)

training_set = get_training_set(star, N)
training_set[:, 0] /= 592  # Normalizing the training_set coordinates to be between 0 and 1                  
training_set[:, 1] /= 592  #(in respecet to the bottem left corner) 

for i in tests:  # Training the networks
    data = training_set[i]
    star_net.backProp([data[0], data[1]], [data[2]])
    star_preceptron.backProp([data[0], data[1]], [data[2]])





star = jpgToMatrix("star.jpg")
point = np.random.randint(592, size = 2)

for i in range (-10 ,10):
    try:
        star[point[0]][point[1]+i] *= -1
        star[point[0] + i][point[1]] *= -1
    except ValueError:
        continue


print("Training Iterations: "+ str(star_net.days_in_gym))
plt.figure(figsize = (10,10))
plt.imshow(star, cmap="gray")
plt.axis(False)
print("Precptron answer: " + str(star_preceptron.feedForward(point/592)))
print("Multi-layered net answer: " +str(star_net.feedForward(point/592)[0]))

plt.close()


#---------- Question 1c - examining hidden layers  ----------#
star = jpgToMatrix("star.jpg")
hid_neurons = np.arange(star_net.sizes[1]+1) 
hid_values = np.zeros(star_net.sizes[1]+1) 

size = 50


def f(point):
    a = np.zeros(star_net.sizes[1]+1) 
    num = 0
    
    for i in range (-size ,size):

        for j in range(-size, size):

            try:

                star[point[0]+i][point[1]+j] *= -1
                star_net.feedForward( [[(point[0]+i)/592],[(point[1]+j)/592]] )
                a += star_net.network[1]
                num += 1
                

            except IndexError:
                 return a/num

    return a/num

count = 0
for i in range (-5,5):
    star = jpgToMatrix("star.jpg")
    count += 1
    hid_values += f( [-70+i,-70+i])  #np.random.randint(592, size = 2)  
hid_values /= count
            
# --- Plotting the data --- #  


plt.bar(hid_neurons, hid_values)
plt.title("Hidden layer response")
plt.xlabel("Neuron index")
plt.ylabel("Neuron value")
plt.savefig("hidden_layer")
plt.figure()
plt.imshow(star, cmap="gray")
plt.axis(False)
plt.close()

# --- Plotting the weights --- #  


plt.bar(hid_neurons, star_net.weights[1][:,0], color = ["m","m","m","m","m","m","r"])
plt.title("Second layer to last layer Weights")
plt.xlabel("Weight index")
plt.ylabel("Value")
plt.grid()
plt.savefig("weights")
plt.close()



#---------- Question 2 - MNIST net  ----------#

file0 = "MNIST files/train-images.idx3-ubyte"
file1 = "MNIST files/train-labels.idx1-ubyte"
file2 = "MNIST files/t10k-images.idx3-ubyte"
file3 = "MNIST files/t10k-labels.idx1-ubyte"
data = idx2numpy.convert_from_file(file0) / 255      # The devison by 255 is in order to normalize the values to be
test_data = idx2numpy.convert_from_file(file2) / 255 # between 0 and 1
labels = idx2numpy.convert_from_file(file1)
test_labels = idx2numpy.convert_from_file(file3)
mnist_net = Network([28*28, 16, 16, 10], step = 0.5)






def error(output, answer):
    res = np.zeros(10)
    res[answer] = 1
    
    error = np.max ( np.abs(res - output ) )
    
    return error
    
    




# --- Training montage --- #

def training_montage():
    mnist_net.clear()
    N = 3*10**4
    deviations = np.zeros(1)
    numbers = np.zeros(1)

    for i in range(3*10**4):

        num = np.random.randint(60000-1)
        res = np.zeros(10)
        res[labels[num]] = 1
        mnist_net.backProp(matrixToArray(data[num]), res)

        if i % 20 == 0:

            test_num = np.random.randint(len(test_labels))    
            output = mnist_net.feedForward(matrixToArray(test_data[test_num]))
            err = error(output, test_labels[test_num])
            deviations = np.append(deviations, err )
            numbers = np.append(numbers, i)
    return deviations, numbers
    



iGotTime_revengance = False
if iGotTime_revengance:

    num = 20
    for i in range(num):
        mnist_net.clear()

        if i == 0:

            deviations, numbers  = training_montage()

            deviations /= num
        else:
            deviations += training_montage()[0] /num
            
            
    # save numpy array as csv file
    from numpy import asarray
    from numpy import savetxt
    #save to csv file
    savetxt('numbers.csv',  numbers, delimiter=',')
    savetxt('deviations.csv', deviations, delimiter=',')
    
    plt.plot(numbers, deviations)
    plt.grid()
    plt.title("MNIST Networks - Error vs training iterations (20 avg)")
    plt.xlabel("Number of training Iterations")
    plt.ylabel("Error")
    plt.savefig("MNIST errors 20 avg")

plt.close()
    


N = 3*10**4

for i in range(3*10**4):   # Training those network biceps (just training)

    num = np.random.randint(60000-1)
    res = np.zeros(10)
    res[labels[num]] = 1
    mnist_net.backProp(matrixToArray(data[num]), res)


# --- Error analysis --- #  


N = 10**4
errors = np.zeros(N)
for i in range(10**4):

    num = np.random.randint(len(test_labels))
    output = mnist_net.feedForward(matrixToArray(test_data[num]))
    err = error(output, test_labels[num])
    errors[i] = err

print(np.mean(errors))


# --- Plotting the data --- #  


plt.hist(errors, bins = 70, color = 'm', edgecolor = 'black')
plt.title("Testing MNIST Net - Error histogram")
plt.xlabel("Error")
plt.ylabel("Number of incidents")
plt.grid()
#plt.savefig("MNIST Error histogram good")



