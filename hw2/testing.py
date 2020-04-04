
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import bleu_eval
import json as js
import os
import sys

data_dir = sys.argv[1]
output_test = sys.argv[2]

# In[2]:

BATCH_SIZE = 32
seq_length = 42
f = open("/home/fjoshi/HW2/MLDS_hw2_1_data/training_label.json") 
data_train = js.load(f)
f_test = open("/home/fjoshi/HW2/MLDS_hw2_1_data/testing_label.json") 
data_test = js.load(f_test)
N = len(data_train)
N_test = len(data_test)

# Gives you the features from JSON file
def getFeatures(filename, directory="training_data"):
    return np.load("/home/fjoshi/HW2/MLDS_hw2_1_data/{}/feat/{}.npy".format(directory, filename))
# Gives you the Label to corresponding feature given the index of it from JSON file
def getLabel(index, data = data_train):
    # Return label sentence string
    return np.random.choice(data[index]['caption'])
def getVideoId(index, data = data_train):
    return data[index]['id']


# In[3]:


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

default_ouput_file = "output.txt"
def write_output_file(output_file=default_ouput_file):
    prediction = sess.run(fetches=pred_sentence, feed_dict={X:batch_testX})
    predicted_sentences = getSentencesFromIndices(prediction)
    with open(output_file, 'w') as f:
        for vids, preds in zip(video_ids, predicted_sentences):
            f.write(vids+","+preds+"\n")
            
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


# In[4]:


sess = tf.Session()
saver = tf.train.import_meta_graph('final_model/final.chkpt.meta')
saver.restore(sess, 'final_model/final.chkpt')

batchX = []
vids = []
for _, _, files in os.walk(data_dir):
    pass

for numpyfile in files:
    nump = np.load("{}/{}".format(data_dir, numpyfile))
    vids.append(numpyfile.split(".npy")[0])
    batchX.append(nump)

# In[5]:

graph = tf.get_default_graph()
X = graph.get_tensor_by_name("Input:0")
feed_dict ={X:batchX}
pred = graph.get_tensor_by_name("Prediction:0")


# In[6]:


predicted_sentence = sess.run(pred, feed_dict)


# In[7]:


predicted_sentences =getSentencesFromIndices(predicted_sentence)


with open(output_test, "w") as f:
    for vid, sentence in zip(vids, predicted_sentences):
       f.write(vid + "," + sentence + "\n")

print("Generated Output File")
