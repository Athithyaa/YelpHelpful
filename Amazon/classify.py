'''
Data Citation
Image-based recommendations on styles and substitutes
J. McAuley, C. Targett, J. Shi, A. van den Hengel
SIGIR, 2015

'''

import pandas as pd
import gzip
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

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
		helpfulOrNot.append(-1)
	else:
		helpfulOrNot.append(1)

[X_train, X_test, y_train, y_test] = train_test_split(reviewText, helpfulOrNot, test_size=0.20)
cV = CountVectorizer(stop_words='english').fit(X_train)
X_train_counts = cV.transform(X_train)
X_train_tf = TfidfTransformer(use_idf=False).fit_transform(X_train_counts)
sgd = SGDClassifier()
print X_train_tf.shape, np.asarray(y_train).shape
clf = sgd.fit(X_train_tf, np.asarray(y_train))
X_test_counts = cV.transform(X_test)
X_test_tf = TfidfTransformer(use_idf=False).fit_transform(X_test_counts)
y_res = clf.predict(X_test_tf)
score = accuracy_score(y_test, y_res)
print score