import csv
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

with open('reviews_data.csv', 'rb+') as inp_file:
    review_reader = csv.reader(inp_file, delimiter=',')
    next(review_reader)
    data = list(review_reader)
    X = [row[0] for row in data]
    y = [row[1] for row in data]
    [X_train, X_test, y_train, y_test] = train_test_split(X[1:400000], y[1:400000], test_size=0.20)
    cV = CountVectorizer(stop_words='english').fit(X_train)
    X_train_counts = cV.transform(X_train)
    X_train_tf = TfidfTransformer(use_idf=False).fit_transform(X_train_counts)
    mnb = MultinomialNB()
    clf = mnb.fit(X_train_tf, np.array(y_train))
    X_test_counts = cV.transform(X_test)
    X_test_tf = TfidfTransformer(use_idf=False).fit_transform(X_test_counts)
    y_res = clf.predict(X_test_tf)
    score = accuracy_score(y_test, y_res)
    print score


