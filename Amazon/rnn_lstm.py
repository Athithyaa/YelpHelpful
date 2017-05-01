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

numpy.random.seed(7)

with open('../reviews_data.csv', 'rb+') as inp_file:
    review_reader = csv.reader(inp_file, delimiter=',')
    next(review_reader)
    data = list(review_reader)
    X = [row[0] for row in data]
    y = [row[1] for row in data]
    [X_train, X_test, y_train, y_test] = train_test_split(X[1:400000], y[1:400000], test_size=0.20)
    frqt_words = 8000
    review_len = 200

    tok = Tokenizer(num_words=frqt_words, lower=True, split=" ")
    tok.fit_on_texts(X_train)
    X_train = tok.texts_to_sequences(X_train)
    X_test = tok.texts_to_sequences(X_test)

    X_train = sequence.pad_sequences(X_train, maxlen=review_len)
    X_test = sequence.pad_sequences(X_test, maxlen=review_len)

    word_embeddings_length = 200
    model = Sequential()
    model.add(Embedding(frqt_words, word_embeddings_length, input_length=review_len))
    model.add(LSTM(128))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=128)
    scores = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy: %.2f%%" % (scores[1]*100))




