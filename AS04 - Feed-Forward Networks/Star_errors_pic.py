#!/usr/bin/env python
# coding: utf-8

# In[1]:


exec(open("FF Network.py").read())


# In[2]:


star = jpgToMatrix("star.jpg")
star = np.delete(star, [592, 593] , axis = 1 )  # Slight adjustment

star_net = Network([2, 6, 1], step = 0.5) 


# In[3]:


N = 5*10**4
tests = np.arange(N)

training_set = get_training_set(star, N)
training_set[:, 0] /= 592  # Normalizing the training_set coordinates to be between 0 and 1                  
training_set[:, 1] /= 592  #(in respecet to the bottem left corner) 

for i in tests:  # Training the networks
    data = training_set[i]
    star_net.backProp([data[0], data[1]], [data[2]])


# In[8]:


N = 10**5
star_err= np.zeros(np.shape(star))
for i in range(N):
    x = np.random.randint(592)
    y = np.random.randint(592)
    star_err[x, y] = np.abs(star_net.feedForward([x/592, y/592])[0] - isInStar(star, [x, y]))
    


# In[12]:


plt.figure()
im1 = plt.imshow(star, cmap = "gray")
im2 = plt.imshow(star_err, cmap ="Reds", alpha = 0.8 )
plt.axis("off")
plt.savefig("star_errors_pic")
plt.show()


# In[ ]:




