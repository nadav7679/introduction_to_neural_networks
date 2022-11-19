"""
Neural Networks AS05 - Data analysis
By: Nadav Porat ; 207825506

"""

exec(open("Functions.py").read())



#------------------  Q1 - spatial pattern ------------------#

posx, posy, spikes, spikes2, head_dir = load_da_data("data/3.mat")

plt.figure(figsize = (8,8))
plt.plot(posx, posy, color = "black")
loc = find_spikes(posx, posy, spikes)
plt.plot(loc[:,0], loc[:, 1] ,linewidth = 0, marker = "+", color = "magenta")
plt.xlabel("X")
plt.ylabel("Y")

plt.title("Cell 3.1 spatial pattern")
plt.show()
#plt.savefig("cell 3_2")




#------------------  Q2 - FireRate heatmaps ------------------#

plt.close()
fig = plt.figure(figsize = (10,8))

heatmap = firerate_heatmap(posx, posy, spikes)
ax = sns.heatmap(heatmap, center = 0, xticklabels = 4, yticklabels = 4 )
plt.gca().invert_yaxis()
plt.title("Cell 3.1 - Fire Rate distribution (Hz)")
plt.xlabel("X  [2.33 cm] ")
plt.ylabel("Y  [2.33 cm] ")

plt.show()
#fig.savefig("uniform heatmap 2_2")




#------------------  Q3 - head direction ------------------#

plt.close()
a = headDir_fire(head_dir, spikes2)
x = np.arange(0, 360, 10)
plt.figure(figsize = (10,8))
plt.grid()
plt.xlabel("Head Direction $(degrees ^\circ )$")
plt.ylabel("Fire Rate ($Hz$)")
plt.title("Cell 3.2 - Head direction FireRate")
plt.plot(x, a, "m-o", markersize = 8)
plt.show()

#plt.savefig("Cell 3_2 Headdir firerate ")




#------------------  Q4 - statistical analysis ------------------#
plt.close()
graphIt = False
posx, posy, spikes, head_dir = load_da_data("data/2_2.mat")
bins =  np.concatenate((np.arange(0, 50000, 2), # The choice of bins is very important! It is basically resolution of calculation.
                np.arange(50000, 400000, 700))) 

if graphIt:
    interspikes = get_interspikes(spikes2) 
    mu = np.mean(interspikes)
    length = len(interspikes)
    std = np.std(interspikes)

    a = np.square(np.rint(norm.rvs(loc = mu, size = length, scale = std)))
    interspikes_square = np.square(interspikes)

    print(np.mean(interspikes_square))
    print(np.mean(a))


    observed = plt.hist(interspikes_square, bins = bins,
                        label = "interspike square", color = "r")
    expected = plt.hist(a, bins = bins,
                        label = "model dist square", alpha = 0.7, color = "b")

    
    plt.title("Cell 2.2 interspike distribution")
    plt.xlabel("Interspike (msec)")
    plt.ylabel("pdf")
    plt.ylim((0,120))
    plt.legend()
    plt.show()

    plt.savefig("interspike dist 2_2")

parameters = analyze_square(spikes2, bins, num = 500)   # Gettig chi squared reduced and p-value    
print(parameters)



#------------------ plotting the random distribution  ------------------#

# nothing new in the following code. I simply copy and pasted the above code and apppplied it to
# a random spikes distribution generetad from cell 2.1

plt.close()

posx, posy, spikes, head_dir = load_da_data("data/2_1.mat")

rand_spikes, rand_interspikes = get_random_spikes_square(get_interspikes(spikes))
loc = find_spikes(posx, posy, rand_spikes)

plt.figure(figsize = (8,8))
plt.plot(posx, posy, color = "black")

plt.plot(loc[:,0], loc[:, 1] ,linewidth = 0, marker = "+", color = "magenta")
plt.xlabel("X")
plt.ylabel("Y")

plt.title("Uniform dist spatial pattern based on cell 2.1")
#plt.savefig("unifrom dist 2_1")

plt.close()
fig = plt.figure(figsize = (10,8))

heatmap = firerate_heatmap(posx, posy, rand_spikes)
ax = sns.heatmap(heatmap, center = 0, xticklabels = 4, yticklabels = 4 )
plt.gca().invert_yaxis()
plt.title("Cell 2.1 - Fire Rate distribution (Hz)")
plt.xlabel("X  [2.33 cm] ")
plt.ylabel("Y  [2.33 cm] ")

plt.show()
#fig.savefig("uniform heatmap 2_1")

