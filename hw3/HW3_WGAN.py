
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Parameters
batch_size=64
learning_rate= 1e-3
num_epochs=50


# In[3]:


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


# In[4]:


def showImage(image, label):
    plt.imshow(image)
    plt.title(label_ids[label])


# In[7]:


#Generator  
# input:Noise ; output: Fake image
def generator(g, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        g = tf.layers.dense(g, units=4 * 4 * 512, kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        g = tf.reshape(g, shape=[-1, 4, 4, 512])
        #g = tf.layers.batch_normalization(g)
        g = tf.nn.relu(g)
       
       #1st: Conv2d,batch_norm,relu
        g = tf.layers.conv2d_transpose(g, 256, 5, strides=2, padding='same',
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        #g = tf.layers.batch_normalization(g, momentum=0.99)
        g = tf.nn.relu(g)
        print(g)
       
       #2nd: Conv2d,batch_norm,relu
        g = tf.layers.conv2d_transpose(g, 128, 5, strides=2, padding='same',
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        #g = tf.layers.batch_normalization(g, momentum=0.99)
        g = tf.nn.relu(g)
        print(g)
        #3rd: Conv2d,batch_norm,relu
        g = tf.layers.conv2d_transpose(g, 64, 5, strides=2, padding='same',
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        #g = tf.layers.batch_normalization(g, momentum=0.99)
        g = tf.nn.relu(g)
        print(g)
        
       
       
        # Last conv2d and tanh
        g = tf.layers.conv2d_transpose(g, 3, 5, strides=1, padding='same',
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        print(g)
        
        image = tf.nn.tanh(g)
        
        return image


# In[12]:


#Discrimiantor 
# input: Image ; output: Prediction fake/real?

def discriminator(d, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        d = tf.layers.conv2d(d, 64, 5, strides=2, padding='same',
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        d = tf.nn.leaky_relu(d)
        print(d)
        #1st: Conv2d,batch_norm,relu
        d = tf.layers.conv2d(d, 128, 5, strides=2, padding='same',
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        #d = tf.layers.batch_normalization(d, momentum=0.99)
        d = tf.nn.leaky_relu(d)
        print(d)
        #2nd: Conv2d,batch_norm,relu
        d = tf.layers.conv2d(d, 256, 5, strides=2, padding='same',
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        #d = tf.layers.batch_normalization(d, momentum=0.99,
        #                              kernel_initializer=tf.random_normal_initializer(stddev=0.05))
        d = tf.nn.leaky_relu(d)    
        print(d)
        #3rd: Conv2d,batch_norm,relu
        #d = tf.layers.conv2d(d, 512, 5, strides=2, padding='same',
        #                               kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        #d = tf.layers.batch_normalization(d, momentum=0.99)
        #d = tf.nn.leaky_relu(d)
        #d = tf.layers.conv2d(d, 512, 1, strides=(1,1), padding='same')
        d = tf.layers.flatten(d)
        d = tf.layers.dense(d, 1, kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        return d


# In[13]:


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

#lr = tf.placeholder(tf.float32, shape = [], name = 'lr')
#beta = tf.placeholder(tf.float32, shape = [], name = 'beta')
#training_mode = tf.placeholder(tf.bool)


# In[14]:


#loss Calculations

#discriminator Loss
d_loss_real = tf.reduce_mean(d_real_logit)

d_loss_fake = tf.reduce_mean(d_fake_logit)

discriminator_loss = (d_loss_real - d_loss_fake)

#generator loss
generator_loss = - d_loss_fake

#generator Variables
g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
#discriminator Variables
d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')


#clipping the generator
discriminator_clip=[c.assign(tf.clip_by_value(c,-0.01,0.01)) for c in d_vars]

generator_optimizer = tf.train.RMSPropOptimizer(learning_rate=generator_lr).minimize(generator_loss,var_list=g_vars)
discriminator_optimizer = tf.train.RMSPropOptimizer(learning_rate=critic_lr).minimize(-discriminator_loss, var_list=d_vars)


# In[15]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()


# In[ ]:


try:
    # Training
    fake_acc = 0
    glr = 5e-4
    clr = 5e-4
    avg_disc_loss = 0
    avg_gen_loss = 0
    intervals = 100
    total_data = len(data)
    idx = 0
    for i in range(50000):
        for critic in range(6):
            if (idx + batch_size >= total_data):
                idx = 0
            examples_image = data[idx : idx + batch_size]
            idx += batch_size
            noise = np.random.uniform(-1,1, size=[batch_size, 100]).astype(np.float32)
        
            # Discriminator Training
            _, loss_disc = sess.run([discriminator_optimizer, discriminator_loss], 
                                feed_dict={real_Image_input: examples_image, noise_input: noise, critic_lr: clr})
            sess.run(discriminator_clip)
        avg_disc_loss += loss_disc
        if (idx + batch_size >= total_data):
            idx = 0
        examples_image = data[idx : idx + batch_size]
        idx += batch_size
        noise = np.random.uniform(-1,1, size=[batch_size, 100]).astype(np.float32)


        _, loss_gen = sess.run([generator_optimizer,generator_loss], 
                                   feed_dict={noise_input: noise, generator_lr: glr})
        avg_gen_loss += loss_gen

        if i % intervals == 0 or i == 1:
            print("Epoch: {}, Generator Loss:{}, Discriminator Loss: {}".format(i,avg_gen_loss/intervals, avg_disc_loss/intervals))
            #saver.save(sess, "model_WGAN/checkpoint.chkpt")
            avg_gen_loss = 0
            avg_disc_loss = 0
except KeyboardInterrupt:
    pass


# In[17]:


iii = sess.run(generated_image, 
                           feed_dict={noise_input: np.random.uniform(-1,1, size=[batch_size, 100]).astype(np.float32)})


# In[18]:


plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow((iii[i] + 1)/2)

