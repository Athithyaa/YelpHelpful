import csv
import numpy
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
import gzip


numpy.random.seed(7)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
df = getDF('reviews_Health_and_Personal_Care_5.json.gz')

reviewText = df['reviewText'].tolist()
helpfulData = df['helpful'].tolist()
helpfulRating = []

for item in helpfulData:
    helpfulRating.append((float(item[0] +1) )/(float(item[1] + 2) ))

helpfulOrNot = []

for item in helpfulRating:
    if item <= 0.5:
        helpfulOrNot.append(0)
    else:
        helpfulOrNot.append(1)


frqt_words = 8000
review_len = 200

tok = Tokenizer(num_words=frqt_words, lower=True, split=" ")
tok.fit_on_texts(reviewText)
reviewText = tok.texts_to_sequences(reviewText)

[x_train, x_test, y_train, y_test] = train_test_split(reviewText, helpfulOrNot, test_size=0.20)


X_train = sequence.pad_sequences(x_train, maxlen=review_len)
X_test = sequence.pad_sequences(x_test, maxlen=review_len)

word_embeddings_length = 200
model = Sequential()
model.add(Embedding(frqt_words, word_embeddings_length, input_length=review_len))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=128)
scores = model.evaluate(X_test, y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))
