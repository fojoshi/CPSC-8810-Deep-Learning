
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Parameters
batch_size=64


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


#Generator function: 
# input:Noise ; output: Fake image
def generator(g, training=True,reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        g = tf.layers.dense(g, units= 4 * 4 * 512, kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        g = tf.reshape(g, shape=[-1, 4, 4, 512])
        g = tf.layers.batch_normalization(g)
        g = tf.nn.relu(g)
       
       #1st: Conv2d,batch_norm,relu
        g = tf.layers.conv2d_transpose(g, 256, 5, strides=2, padding='same',
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        g = tf.layers.batch_normalization(g, momentum=0.99, training=training)
        g = tf.nn.relu(g)
       
       #2nd: Conv2d,batch_norm,relu
        g = tf.layers.conv2d_transpose(g, 128, 5, strides=2, padding='same',
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        g = tf.layers.batch_normalization(g, momentum=0.99, training=training)
        g = tf.nn.relu(g)
       
        #3rd: Conv2d,batch_norm,relu
        g = tf.layers.conv2d_transpose(g, 64, 5, strides=2, padding='same',
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        g = tf.layers.batch_normalization(g, momentum=0.99, training=training)
        g = tf.nn.relu(g)
       
       
       
        # Last conv2d and tanh
        g = tf.layers.conv2d_transpose(g, 3, 5, strides=1, padding='same',
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        image = tf.nn.tanh(g)
        
        return image


# In[5]:


#Discrimiantor function:
# input: Image ; output: Prediction fake/real?

def discriminator(d, training=True, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        d = tf.layers.conv2d(d, 64, 5, strides=2, padding='same',
                                      kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        d = tf.nn.leaky_relu(d)
        #1st: Conv2d,batch_norm,relu
        d = tf.layers.conv2d(d, 128, 5, strides=2, padding='same',
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        d = tf.layers.batch_normalization(d, momentum=0.99, training=training)
        d = tf.nn.leaky_relu(d)
        #2nd: Conv2d,batch_norm,relu
        d = tf.layers.conv2d(d, 256, 5, strides=2, padding='same',
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        d = tf.layers.batch_normalization(d, momentum=0.99, training=training)
        d = tf.nn.leaky_relu(d)    
        #3rd: Conv2d,batch_norm,relu
        d = tf.layers.conv2d(d, 512, 1, strides=2, padding='same',
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        d = tf.layers.batch_normalization(d, momentum=0.99, training=training)
        d = tf.nn.leaky_relu(d)

        #d = tf.layers.conv2d(d, 3, 5, strides=(1,1), padding='same')
        d = tf.layers.flatten(d)
        d = tf.layers.dense(d, 1, kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        return d


# In[6]:


tf.reset_default_graph()

#Noise input to generator
noise_input = tf.placeholder(dtype=tf.float32,shape= [None, 100], name="NoiseInputGenerator")
#Image input to discriminator
real_Image_input = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3], name="ImageInputDiscriminator")

generated_image=generator(noise_input)
d_real_logit=discriminator(real_Image_input)
d_fake_logit=discriminator(generated_image,reuse=True)

lr = tf.placeholder(tf.float32, shape = [], name = 'lr')
training_mode = tf.placeholder(tf.bool)


# In[7]:



d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=d_real_logit, labels=tf.ones_like(d_real_logit)))

d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=d_fake_logit, labels=tf.zeros_like(d_fake_logit)))
discriminator_loss = d_loss_real + d_loss_fake

#generator loss
generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=d_fake_logit, labels=tf.ones_like(d_fake_logit)))

#generator Variables
g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
#discriminator Variables
d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')
g_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = 'Generator')
d_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = 'Discriminator')

with tf.control_dependencies(g_ops):
    generator_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(generator_loss,var_list=g_vars)
with tf.control_dependencies(d_ops):
    discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(discriminator_loss, var_list=d_vars)


# In[9]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[ ]:


#saver = tf.train.Saver()
#saver.restore(sess, "models6_dc_1/checkpoint.chkpt")
#saver.restore(sess, "model_WGAN_uniformnoise_thegoodone2")


# In[39]:


try:
    # Training
    idx = 0
    total_data = len(data)
    batch_size = 50
    fake_acc = 0
    learning_rate = 5e-5
    for i in range(50000):
        for j in range(3):
            noise = np.random.uniform(-1,1, size=[batch_size, 100]).astype(np.float32)
            if (idx + batch_size >= total_data):
                idx = 0
            examples_image = data[idx : idx + batch_size]
            idx += batch_size
            _, loss_disc = sess.run([discriminator_optimizer, discriminator_loss], 
                                        feed_dict={real_Image_input: examples_image, noise_input: noise, 
                                                   training_mode:True, lr: learning_rate})
        noise = np.random.uniform(-1,1, size=[batch_size, 100]).astype(np.float32)
        if (idx + batch_size >= total_data):
            idx = 0
        examples_image = data[idx : idx + batch_size]
        idx += batch_size
            
        _, loss_gen = sess.run([generator_optimizer,generator_loss], 
                               feed_dict={noise_input: noise,
                                         training_mode:True, lr: learning_rate})
        fake_acc += sess.run(tf.reduce_mean(tf.round(tf.nn.sigmoid(d_fake_logit))), {noise_input: noise})
        if i % 100==0 or i==1:
            print("Epoch: {}, Generator Loss:{}, Discriminator Loss: {}, Fake Acc: {}".format(i,loss_gen, loss_disc, fake_acc/100))
            #saver.save(sess, "models6_dc_8/checkpoint.chkpt")
            fake_acc = 0
except KeyboardInterrupt:
    pass


# In[48]:


imgs = sess.run(generated_image, 
                           feed_dict={noise_input: np.random.uniform(-1, 1, size=[64, 100]).astype(np.float32)})
imgs = (imgs + 1)/2


# In[49]:


plt.figure(figsize=(18, 18))
for i in range(64):
    plt.subplot(8, 8, i + 1)
    plt.imshow(imgs[i])
    plt.title(i)


# In[52]:


# calculate inception score for cifar-10 in Keras
'''
from math import floor
from numpy import ones
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp
from numpy.random import shuffle
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.datasets import cifar10
from skimage.transform import resize
from numpy import asarray

# scale an array of images to a new size
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return asarray(images_list)

# assumes images have any shape and pixels in [0,255]
def calculate_inception_score(images, n_split=2, eps=1E-16):
    # load inception v3 model
    model = InceptionV3()
    # enumerate splits of images/predictions
    scores = list()
    n_part = floor(images.shape[0] / n_split)
    for i in range(n_split):
        # retrieve images
        ix_start, ix_end = i * n_part, (i+1) * n_part
        subset = images[ix_start:ix_end]
        # convert from uint8 to float32
        subset = subset.astype('float32')
        # scale images to the required size
        subset = scale_images(subset, (299,299,3))
        # pre-process images, scale to [-1,1]
        subset = preprocess_input(subset)
        # predict p(y|x)
        p_yx = model.predict(subset)
        # calculate p(y)
        p_y = expand_dims(p_yx.mean(axis=0), 0)
        # calculate KL divergence using log probabilities
        kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
        # sum over classes
        sum_kl_d = kl_d.sum(axis=1)
        # average over images
        avg_kl_d = mean(sum_kl_d)
        # undo the log
        is_score = exp(avg_kl_d)
        # store
        scores.append(is_score)
        print(mean(scores))
    # average across images
    is_avg, is_std = mean(scores), std(scores)
    return is_avg, is_std

'''



# In[53]:


'''images = sess.run(generated_image, 
                           feed_dict={noise_input: np.random.uniform(-1, 1, size=[1000, 100]).astype(np.float32)})
images = images * 255
shuffle(images)
print('loaded', images.shape)
# calculate inception score
is_avg, is_std = calculate_inception_score(images)
print('score', is_avg, is_std)'''

