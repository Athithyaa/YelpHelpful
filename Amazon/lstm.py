from __future__ import print_function


import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import pandas as pd
import gzip
import numpy as np


from keras.preprocessing import sequence
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb


max_features = 20000
max_words = 5000
maxlen = 200
batch_size = 32

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

print('Loading data...')
df = getDF('reviews_mi.json.gz')
reviewText = df['reviewText'].tolist()
helpfulData = df['helpful'].tolist()
helpfulRating = []
for item in helpfulData:
	if item[0] == 0:
		helpfulRating.append(float(item[0]))
	else:
		helpfulRating.append(float(item[0])/float(item[1]))

helpfulOrNot = []
for item in helpfulRating:
	if item == 0.0:
		helpfulOrNot.append(0)
	else:
		helpfulOrNot.append(1)

tok = Tokenizer(num_words=max_words,lower=True, split=" ")
tok.fit_on_texts(reviewText)
reviewText = tok.texts_to_sequences(reviewText)

[x_train, x_test, y_train, y_test] = train_test_split(reviewText, helpfulOrNot, test_size=0.20)


print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)