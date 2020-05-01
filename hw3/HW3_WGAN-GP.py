
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds


# In[2]:


# Parameters
batch_size=64
learning_rate= 1e-3
num_epochs=50


# In[3]:


# In[4]:


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

db = unpickle("/home/fjoshi/HW3/cifar-100-python/train")

def convert(i):
    return np.transpose(np.reshape(i, (32, 32, 3), 'F'), (1, 0, 2))

data = []

for img in db[b'data']:
    image = convert (img)
    image = image/255
    image = image * 2 - 1
    data.append(image)


# In[72]:


#Generator  
# input:Noise ; output: Fake image
def generator(g, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        g = tf.layers.dense(g, units=4 * 4 * 256, kernel_initializer=None)
        g = tf.reshape(g, shape=[-1, 4, 4, 256])
        #g = tf.layers.batch_normalization(g)
        g = tf.nn.leaky_relu(g)
       
       #1st: Conv2d,batch_norm,relu
        g = tf.layers.conv2d_transpose(g, 128, 5, strides=2, padding='same')
        #g = tf.layers.batch_normalization(g, momentum=0.99)
        g = tf.nn.leaky_relu(g)
       
       #2nd: Conv2d,batch_norm,relu
        g = tf.layers.conv2d_transpose(g, 64, 5, strides=2, padding='same')
        #g = tf.layers.batch_normalization(g, momentum=0.99)
        g = tf.nn.leaky_relu(g)
        #3rd: Conv2d,batch_norm,relu
        g = tf.layers.conv2d_transpose(g, 32, 5, strides=2, padding='same')
        #g = tf.layers.batch_normalization(g, momentum=0.99)
        g = tf.nn.leaky_relu(g)
        
       
       
        # Last conv2d and tanh
        g = tf.layers.conv2d_transpose(g, 3, 5, strides=1, padding='same')
        
        image = tf.nn.tanh(g)
        
        return image


# In[73]:


#Discrimiantor 
# input: Image ; output: Prediction fake/real?

def discriminator(d, reuse=False):
    initializer = None
    with tf.variable_scope('Discriminator', reuse=reuse):
        d = tf.layers.conv2d(d, 16, 5, strides=2, padding='same',
                                       kernel_initializer=initializer)
        d = tf.nn.leaky_relu(d)
        d = tf.layers.conv2d(d, 32, 5, strides=2, padding='same',
                                       kernel_initializer=initializer)
        d = tf.nn.leaky_relu(d)
        d = tf.layers.conv2d(d, 64, 5, strides=2, padding='same',
                                       kernel_initializer=initializer)
        #d = tf.layers.batch_normalization(d, momentum=0.99)
        d = tf.nn.leaky_relu(d)
        print(d)
        #2nd: Conv2d,batch_norm,relu
        d = tf.layers.conv2d(d, 128, 5, strides=2, padding='same',
                                       kernel_initializer=initializer)
        d = tf.nn.leaky_relu(d)    
        d = tf.layers.flatten(d)
        d = tf.layers.dense(d, 1)
        return d


# In[74]:


tf.reset_default_graph()
generator_lr = tf.placeholder(tf.float32, shape=[])
critic_lr = tf.placeholder(tf.float32, shape=[])
#Noise input to generator
noise_input = tf.placeholder(dtype=tf.float32,shape= [None, 100], name="NoiseInputGenerator")
#Image input to discriminator
real_Image_input = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3], name="ImageInputDiscriminator")

generated_image=generator(noise_input)
d_real_logit=discriminator(real_Image_input)
d_fake_logit=discriminator(generated_image,reuse=True)


# In[75]:


#loss Calculations

#discriminator Loss
ip_size = tf.shape(real_Image_input)[0]
epsilon = tf.random_uniform([ip_size, 1, 1, 1], 0, 1)
x_linear_interpolate = epsilon * real_Image_input + (1 - epsilon) * generated_image
d_interpolate = discriminator(x_linear_interpolate, reuse=True)
gradient = tf.gradients(d_interpolate, [x_linear_interpolate])[0]
norm = tf.sqrt(tf.reduce_sum(tf.square(gradient), reduction_indices=[1, 2, 3]))
gradient_penalty = 10 * tf.reduce_mean((norm-1)**2)


d_loss_real = tf.reduce_mean(d_real_logit)

d_loss_fake = tf.reduce_mean(d_fake_logit)

discriminator_loss = d_loss_fake - d_loss_fake + gradient_penalty

#generator loss
generator_loss = - d_loss_fake



#generator Variables
g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
#discriminator Variables
d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')


generator_optimizer = tf.train.AdamOptimizer(learning_rate=generator_lr, beta1=0.5).minimize(generator_loss,var_list=g_vars)
discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=critic_lr, beta1=0.5).minimize(discriminator_loss, var_list=d_vars)


# In[76]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())
dataset = tfds.load(name="cifar10", split="train").repeat()
saver = tf.train.Saver()
cifar_data = dataset.batch(batch_size).prefetch(10)


# In[77]:


try:
    # Training
    fake_acc = 0
    glr = 1e-5
    clr = 1e-4
    avg_disc_loss = 0
    avg_gen_loss = 0
    intervals = 100
    idx = 0
    total_data = len(data)
        
    for i in range(50000):
        for j in range(5):
            noise = np.random.uniform(-1,1, size=[batch_size, 100]).astype(np.float32)
            if (idx + batch_size >= total_data):
                idx = 0
            examples_image = data[idx : idx + batch_size]
            idx += batch_size
            # Discriminator Training
            _, loss_disc = sess.run([discriminator_optimizer, discriminator_loss], 
                                feed_dict={real_Image_input: examples_image, noise_input: noise, critic_lr: clr})
        avg_disc_loss += loss_disc
        _, loss_gen = sess.run([generator_optimizer,generator_loss], 
                                   feed_dict={noise_input: noise, generator_lr: glr})
        avg_gen_loss += loss_gen

        if (i) % intervals == 0:
            print("Epoch: {}, Generator Loss:{}, Discriminator Loss: {}".format(i,avg_gen_loss/intervals, avg_disc_loss/intervals))
            saver.save(sess, "model_WGANGP/checkpoint.chkpt")
            avg_gen_loss = 0
            avg_disc_loss = 0
except KeyboardInterrupt:
    pass


# In[ ]:


#saver = tf.train.Saver()
#saver.save(sess, "model_WGAN_uniformnoise_thegoodone2/checkpoint.chkpt")


# In[79]:


iii = sess.run(generated_image, 
                           feed_dict={noise_input: np.random.uniform(-1,1, size=[batch_size, 100]).astype(np.float32)})
iii = (iii + 1)/2


# In[80]:


plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(iii[i])

