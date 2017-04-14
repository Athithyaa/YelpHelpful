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
from sklearn.neural_network import MLPClassifier

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

df = getDF('reviews_Video_Games.json.gz')
reviewText = df['reviewText'].tolist()
helpfulData = df['helpful'].tolist()
helpfulRating = []
for item in helpfulData:

    #smoothed the counts. for every review, added 1 count for helpful and 1 for unhelpful. gives a 3% improvement.
    helpfulRating.append((float(item[0] +1) )/(float(item[1] + 2) ))

helpfulOrNot = []
for item in helpfulRating:
	if item <= 0.5:
		helpfulOrNot.append(-1)
	else:
		helpfulOrNot.append(1)

[X_train, X_test, y_train, y_test] = train_test_split(reviewText, helpfulOrNot, test_size=0.20)
cV = CountVectorizer(stop_words='english').fit(X_train)
X_train_counts = cV.transform(X_train)
X_train_tf = TfidfTransformer(use_idf=False).fit_transform(X_train_counts)

#clf_nn = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
#clf_nn.fit(X_train_tf, np.asarray(y_train))

clf_nb  = MultinomialNB()
clf_nb.fit(X_train_tf, np.asarray(y_train))

#sgd = SGDClassifier()
#clf = sgd.fit(X_train_tf, np.asarray(y_train))
X_test_counts = cV.transform(X_test)
X_test_tf = TfidfTransformer(use_idf=False).fit_transform(X_test_counts)
#y_res = clf.predict(X_test_tf)

#y_res_nn = clf_nn.predict(X_test_tf)
y_res_nb = clf_nb.predict(X_test_tf)

#score = accuracy_score(y_test, y_res)
#score_nn = accuracy_score(y_test, y_res_nn)
score_nb = accuracy_score(y_test, y_res_nb)
print "Multinomial Naive Bayes: ", score_nb
#print "Neural network score:", score_nn