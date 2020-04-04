
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json as js
import bleu_eval


# In[40]:


BATCH_SIZE = 128
seq_length = 42
f = open("/home/fjoshi/HW2/MLDS_hw2_1_data/training_label.json") 
data_train = js.load(f)
f_test = open("/home/fjoshi/HW2/MLDS_hw2_1_data/testing_label.json") 
data_test = js.load(f_test)
N = len(data_train)
N_test = len(data_test)


# In[3]:


# Gives you the features from JSON file
def getFeatures(filename, directory="training_data"):
    return np.load("/home/fjoshi/HW2/MLDS_hw2_1_data/{}/feat/{}.npy".format(directory, filename))
# Gives you the Label to corresponding feature given the index of it from JSON file
def getLabel(index, data = data_train):
    # Return label sentence string
    return np.random.choice(data[index]['caption'])
def getVideoId(index, data = data_train):
    return data[index]['id']


# In[4]:


def getIndicesFromSentence(sentence):
    # If input is "<bos> A man is going <eos>". 
    # indices will be : [2, 4, 6, 42, 125, 3, 0, 0, 0, 0, 0, 0]
    # If word is not in word_to_index_dict, replace with <unk>
    # word_to_index_dict[word]
    

    indices=[]
    word=sentence.split()
    # indices.append(word_to_index_dict['<bos>'])
    for w in word:
        w=''.join(e for e in w if e.isalnum())
        w=w.lower()

        if w in word_to_index_dict:

            indices.append(word_to_index_dict[w])
        else:
            indices.append(word_to_index_dict['<unk>'])
    indices.append(word_to_index_dict['<eos>'])    
    
    # Check current size of indices --> len function
    # while loop until len(indices) < seq_length... add padding.  
    # return indices must be of size [seq_length]
    len(indices)
    while len(indices) < seq_length:
        indices.append(0)
    
    return indices


# In[5]:


def getSentencesFromIndices(batch):
    sentences = []
    for indices in batch:
        sentence = ""
        for index in indices:
            if index not in [0, 2, 3]:
                sentence += index_to_word_dict[index] +" "
            if index == 3:
                break
        sentences.append(sentence.strip())
    return sentences


# In[6]:


#This function makes mini-batch
def sample_minibatch(count=BATCH_SIZE):
    # Step 1: Sample random indices from N of size = batch_size
    batch_ids = np.random.choice(N, [count], replace=False)
    batch_X = []
    batch_Y = []
    # Step 2: Get Features of all batch IDS, store in batch_X ==> (batch_size, 80, 4096)
    # Step 3: Get a random label of each corresponding batch ids and store it in batch_Y ==> (batch_size, 1)
    for i in batch_ids:
        batch_X.append(getFeatures(data_train[i]['id']))
        batch_Y.append(getIndicesFromSentence(getLabel(i)))
        
    return batch_X, batch_Y

def test_batch():
   
    batch_testX = []
    video_ids = []
    for i in range(N_test):
        batch_testX.append(getFeatures(data_test[i]['id'], "testing_data"))
        # batch_testY.append(getIndicesFromSentence(getLabel(i, data_test)))
        video_ids.append(getVideoId(i, data_test))
        
    return batch_testX, video_ids

batch_testX, video_ids = test_batch()


# In[7]:


default_ouput_file = "output.txt"
def write_output_file(output_file=default_ouput_file):
    prediction = sess.run(fetches=pred_sentence, feed_dict={X:batch_testX})
    predicted_sentences = getSentencesFromIndices(prediction)
    with open(output_file, 'w') as f:
        for vids, preds in zip(video_ids, predicted_sentences):
            f.write(vids+","+preds+"\n")


# In[8]:


word_to_count_dict = {}
word_to_index_dict = {}
index_to_word_dict = {}
count=0

# word_to_index_dict maps all words in training labels into an index..
# For example, a ==> 1, the ==> 2, woman ==> 3

# index_to_word_dict maps indexes to words...
# 1 ==> a, 2 ==> the, etc


# Loop through all videos in data_train
# Loop through all captions in videos
# Split every caption into words using .split() method
# Insert into dictionary using syntax: word_to_index_dict[key] = value

for videos in data_train: # Getting the videos from data train
    for captions in videos['caption']:# Getting the captions from videos(in data train)
        word=captions.split() 
        for w in word:
            w=''.join(e for e in w if e.isalnum())
            w=w.lower()
            # w.replace() ==> Remove ',.
            # w.lower() ==> lower case
            
            # Beyond this point, w must be preprocessed
            if w in word_to_count_dict:
                word_to_count_dict[w]=word_to_count_dict[w]+1
            else:    
                word_to_count_dict[w]=1


# Loop through all words in word_to_count_dict
# If word_count > 3, insert word into word_index_dict with a unique id 
# Every insert, do a unique_id += 1


# ADD <pad>, <unk>, <bos>, <eos>

word_to_index_dict['<pad>'] = 0
word_to_index_dict['<unk>'] = 1
word_to_index_dict['<bos>'] = 2
word_to_index_dict['<eos>'] = 3
index_to_word_dict[0] = '<pad>'
index_to_word_dict[1] = '<unk>'
index_to_word_dict[2] = '<bos>'
index_to_word_dict[3] = '<eos>'
unique_id = 4
for w in word_to_count_dict:
    if word_to_count_dict[w]>=3:
        word_to_index_dict[w]= unique_id
        index_to_word_dict[unique_id]= w
        unique_id += 1


# In[41]:


tf.reset_default_graph()
hidden = 128
X = tf.placeholder(tf.float32, shape = [None, 80,4096], name = 'Input')
Y = tf.placeholder(tf.int32, shape = [None, seq_length], name = 'Output')
batch_size = tf.shape(X)[0]
padding = tf.zeros(shape=[batch_size, hidden])

lstm1= tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=hidden)
lstm2= tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=hidden)
state1=lstm1.zero_state(batch_size,dtype=tf.float32)
state2=lstm2.zero_state(batch_size,dtype=tf.float32)

vocab_size= len(word_to_index_dict)
embedding_size =hidden #it is equal to hidden units
word_embed=tf.Variable(tf.random.truncated_normal([vocab_size,embedding_size]))

for i in range(80):
    input_lstm1_enc= X[:,i,:]
    output_1_enc, state1= lstm1(input_lstm1_enc,state1)
    
    input_lstm2_enc = tf.concat([output_1_enc,padding],1)
    output_2_enc, state2= lstm2(input_lstm2_enc,state2)

pad=tf.zeros([batch_size,4096])
bos = tf.fill([batch_size], 2)
word_predicted_embedding = tf.nn.embedding_lookup(word_embed, bos)
loss = tf.zeros(batch_size)
pred_sentence = tf.fill([1, batch_size], 2)
for i in range(seq_length):
    output_lstm1_dec, state1= lstm1(pad,state1)    
    input_lstm2_dec=tf.concat([output_lstm1_dec,word_predicted_embedding],1)
    output_lstm2_dec,state2=lstm2(input_lstm2_dec,state2)
    
    # predicted_word_index=tf.argmax(tf.matmul(output_lstm2_dec,tf.transpose(word_embed)),axis=1)
    
    y_at_time_i = Y[:, i]  # size = BS,
    y_oneHotEnc = tf.one_hot(indices=y_at_time_i, depth=vocab_size) # BS * VS
    logits = tf.matmul(output_lstm2_dec, tf.transpose(word_embed)) # BS * VS
    predicted_word_index=tf.argmax(logits, axis=1, output_type=tf.int32)
    word_predicted_embedding=tf.nn.embedding_lookup(word_embed,predicted_word_index)
    pred_sentence = tf.concat([pred_sentence, tf.expand_dims(predicted_word_index, axis=0)], axis=0)
    #pred_sentence.append(predicted_word_index)
    loss_of_this_word = tf.nn.softmax_cross_entropy_with_logits(labels=y_oneHotEnc, logits=logits)
    loss += loss_of_this_word
pred_sentence = tf.transpose(pred_sentence, name="Prediction")
loss = tf.reduce_mean(loss)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)


# In[42]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())



# In[ ]:


last_losses = 0
intervals = 5
log_intervals = 50
best_bleu_score = 0.5
saver = tf.train.Saver()
for i in range(15001):
    batch_X, batch_Y = sample_minibatch()
    l, test= sess.run(fetches=[loss, optimizer], feed_dict={X:batch_X, Y:batch_Y})
    last_losses += l
    if i % intervals == 0:
        write_output_file(default_ouput_file)
        current_bleu_score = bleu_eval.getBlueScore(default_ouput_file)
        if (current_bleu_score > best_bleu_score):
            best_bleu_score = current_bleu_score
            write_output_file("best_ouput.txt")
            saver.save(sess, "Foram_testing4/final.chkpt")
            print("Updated Best Score: ", best_bleu_score)
            
    if i % log_intervals == 0:
        print(i, last_losses/log_intervals, best_bleu_score)
        last_losses = 0


# In[36]:


prediction = sess.run(fetches=pred_sentence, feed_dict={X:batch_X, Y:batch_Y})


# In[ ]:


np.shape(prediction)


# In[ ]:


batch_X, batch_Y = sample_minibatch()


# In[40]:


batch_Y


# In[37]:


getSentencesFromIndices(batch_Y)


# In[38]:


getSentencesFromIndices(prediction)


# In[ ]:


getSentencesFromIndices(prediction)


# In[ ]:


saver = tf.train.Saver()


# In[ ]:


saver.save(sess, "training/checkpoint.chkpt")


# In[ ]:


saver.save(sess, "training2/checkpoint.chkpt")


# In[28]:


bleu_eval.getBlueScore(default_ouput_file)

