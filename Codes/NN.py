# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 13:27:17 2021

@author: aniru
"""
import nltk
import os
import string
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import collections

with open('train/train/dialogues_train.txt', 'r', encoding='utf-8') as f:
    train_sent = f.readlines()

train_sent = [x.strip() for x in train_sent]
train_sent = [x.split(" __eou__") for x in train_sent]

with open('test/test/dialogues_test.txt', 'r', encoding='utf-8') as f:
    test_sent = f.readlines()

test_sent = [x.strip() for x in test_sent]
test_sent = [x.split(" __eou__") for x in test_sent]


with open('train/train/dialogues_emotion_train.txt', 'r', encoding='utf-8') as f:
    train_labels = f.readlines()

train_labels = [x.strip() for x in train_labels]
train_labels = [x.split(" ") for x in train_labels]

with open('test/test/dialogues_emotion_test.txt', 'r', encoding='utf-8') as f:
    test_labels = f.readlines()

test_labels = [x.strip() for x in test_labels]
test_labels = [x.split(" ") for x in test_labels]


def flatten(x):
    result = []
    for el in x:
        if isinstance(x, collections.Iterable) and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

train_sent = flatten(train_sent)
train_labels = flatten(train_labels)
print(train_labels[0:10])

test_sent = flatten(test_sent)
test_labels = flatten(test_labels)
print(test_labels[0:12])

# To remove trailing blank element
while('' in train_sent) :
    train_sent.remove('')
    
print(train_sent[0:10])

while('' in test_sent) :
    test_sent.remove('')
    
print(test_sent[0:12])

def full_remove(x, removal_list):
    for w in removal_list:
        x = x.replace(w, ' ')
    return x

# Remove digits 
digits = [str(x) for x in range(10)]
remove_digits = [full_remove(x, digits) for x in train_sent]

# Remove punctuation
remove_punc = [full_remove(x, list(string.punctuation)) for x in remove_digits]

# Make everything lower-case and remove any white space
sents_lower = [x.lower() for x in remove_punc]
sents_lower = [x.strip() for x in sents_lower]

# Remove stop words 
stop_set = ['the', 'a', 'an', 'i', 'he', 'she', 'they', 'to', 'of', 'it', 'from']
def removeStopWords(stopWords, txt):
    newtxt = ' '.join([word for word in txt.split() if word not in stopWords])
    return newtxt

train_sents_processed = [removeStopWords(stop_set,x) for x in sents_lower]

print(train_sents_processed[0:10])

# Remove digits 
digits1 = [str(x) for x in range(10)]
remove_digits1 = [full_remove(x, digits) for x in test_sent]

# Remove punctuation
remove_punc1 = [full_remove(x, list(string.punctuation)) for x in remove_digits1]

# Make everything lower-case and remove any white space
sents_lower1 = [x.lower() for x in remove_punc1]
sents_lower1 = [x.strip() for x in sents_lower1]

test_sents_processed = [removeStopWords(stop_set,x) for x in sents_lower1]

print(test_sents_processed[:12])

from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Flatten, Dense

max_review_length = 200
tokenizer = Tokenizer(num_words=10000,  #max no. of unique words to keep
                      filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', 
                      lower=True #convert to lower case
                     )
tokenizer.fit_on_texts(train_sents_processed)
tokenizer.fit_on_texts(test_sents_processed)

x_train = tokenizer.texts_to_sequences(train_sents_processed)
x_train = sequence.pad_sequences(x_train, maxlen= max_review_length)
y_train = np.array(train_labels, dtype='int8')
print('Shape of training data tensor:', x_train.shape)

x_test = tokenizer.texts_to_sequences(test_sents_processed)
x_test = sequence.pad_sequences(x_test, maxlen= max_review_length)
y_test = np.array(test_labels, dtype='int8')
print('Shape of testing data tensor:', x_test.shape)

import pandas as pd
y_train = pd.get_dummies(y_train).values
y_test = pd.get_dummies(y_test).values


EMBEDDING_DIM = 200
model = Sequential()
model.add(Embedding(10000, EMBEDDING_DIM, input_length=x_train.shape[1]))
model.add(Flatten())
model.add(Dense(7, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

print(model.summary())
history = model.fit(x_train, y_train, batch_size=256, epochs=6, verbose=1, validation_split=0.2)
score = model.evaluate(x_test, y_test, verbose=1)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()