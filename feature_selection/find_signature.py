#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10) mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )


### test_size is the percentage of events assigned to the test set (the remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number of data points and a large number of features;
### train on only 150 events to put ourselves in this regime

features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]


from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)
accuracy = accuracy_score(pred, labels_test)
print 'accuracy: {}\n'.format(accuracy)

# get a list of the relative importance of all the features being used.
v = clf.n_features_ 
importance = clf.feature_importances_
print 'number of features: {}'.format(v)
print 'list of relative impotrtance: {}\n'.format(importance)

maxx = 0
most_important = 0
count = 0
for i,j in enumerate(importance):
	if j > maxx:
		maxx = j
		most_important = i
	if j > 0.2:
		count += 1	
		
print 'max important feature: {}'.format(maxx)
print 'most important features index: {}'.format(most_important)
print 'count of feature importance > 0.2: {}'.format(count)


feature_names = vectorizer.get_feature_names()
print feature_names[most_important]

