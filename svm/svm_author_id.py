#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")

from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np


features_train, features_test, labels_train, labels_test = preprocess()

if False:
	### What is the accuracy of the classifier?
	### How do the training and prediction times compare to Naive Bayes? (slower)
	clf = SVC(kernel="linear")

	# fitting/training
	t0 = time()
	clf.fit(features_train, labels_train) 
	print "training time:", round(time()-t0, 3), "s"

	# prediction
	t0 = time()
	pred = clf.predict(features_test)
	print "predict time:", round(time()-t0, 3), "s"

	# accuracy
	t0 = time()
	acc = accuracy_score(pred, labels_test)
	print "accuracy time:", round(time()-t0, 3), "s"
	print "accuracy:", acc

if False:
	### One way to speed up an algorithm is to train it on a smaller training dataset. 
	### What is the accuracy now?
	clf = SVC(kernel="linear")

	features_train = features_train[:len(features_train)/100] 
	labels_train = labels_train[:len(labels_train)/100] 

	# fitting/training
	t0 = time()
	clf.fit(features_train, labels_train) 
	print "training time:", round(time()-t0, 3), "s"

	# prediction
	t0 = time()
	pred = clf.predict(features_test)
	print "predict time:", round(time()-t0, 3), "s"

	# accuracy
	t0 = time()
	acc = accuracy_score(pred, labels_test)
	print "accuracy time:", round(time()-t0, 3), "s"
	print "accuracy:", acc

if False:
	### Change the kernel of your SVM to 'rbf'
	### What is  the accuracy now?
	clf = SVC(kernel="rbf")

	features_train = features_train[:len(features_train)/100] 
	labels_train = labels_train[:len(labels_train)/100] 

	# fitting/training
	t0 = time()
	clf.fit(features_train, labels_train) 
	print "training time:", round(time()-t0, 3), "s"

	# prediction
	t0 = time()
	pred = clf.predict(features_test)
	print "predict time:", round(time()-t0, 3), "s"

	# accuracy
	t0 = time()
	acc = accuracy_score(pred, labels_test)
	print "accuracy time:", round(time()-t0, 3), "s"
	print "accuracy:", acc

if False:
	### Keep the training set size and rbf kernel from the last quiz, but try several values of C (say, 10.0, 100., 1000., and 10000.). 
	### Which one gives the best accuracy?
	### Once you've optimized the C value for your RBF kernel, what accuracy does it give?
	### Does this C value correspond to a simpler or more complex decision boundary?

	clf = SVC(kernel="rbf",C=10000)

	features_train = features_train[:len(features_train)/100] 
	labels_train = labels_train[:len(labels_train)/100] 

	# fitting/training
	t0 = time()
	clf.fit(features_train, labels_train) 
	print "training time:", round(time()-t0, 3), "s"

	# prediction
	t0 = time()
	pred = clf.predict(features_test)
	print "predict time:", round(time()-t0, 3), "s"

	# accuracy
	t0 = time()
	acc = accuracy_score(pred, labels_test)
	print "accuracy time:", round(time()-t0, 3), "s"
	print "accuracy:", acc

if False:
	### go back to using the full training set.
	### What is the accuracy of the optimized SVM?

	clf = SVC(kernel="rbf",C=10000)

	# fitting/training
	t0 = time()
	clf.fit(features_train, labels_train) 
	print "training time:", round(time()-t0, 3), "s"

	# prediction
	t0 = time()
	pred = clf.predict(features_test)

	# accuracy
	t0 = time()
	acc = accuracy_score(pred, labels_test)
	print "accuracy time:", round(time()-t0, 3), "s"
	print "accuracy:", acc

if False:
	### What class does your SVM (0 or 1, corresponding to Sara and Chris respectively) predict for
	### element 10 of the test set? The 26th? The 50th?
	### Use the RBF kernel, C=10000, and 1% of the training set

	clf = SVC(kernel="rbf",C=10000)

	features_train = features_train[:len(features_train)/100] 
	labels_train = labels_train[:len(labels_train)/100] 

	# fitting/training
	t0 = time()
	clf.fit(features_train, labels_train) 
	print "training time:", round(time()-t0, 3), "s"

	# prediction
	t0 = time()
	pred = clf.predict(features_test)
	print "predict time:", round(time()-t0, 3), "s"

	# accuracy
	t0 = time()
	acc = accuracy_score(pred, labels_test)
	print "accuracy time:", round(time()-t0, 3), "s"
	print "accuracy:", acc

if False:
	### There are over 1700 test events--how many are predicted to be in the 'Chris' (1) class?
	### (Use the RBF kernel, C=10000., and the full training set.)

	clf = SVC(kernel="rbf",C=10000)

	# fitting/training
	t0 = time()
	clf.fit(features_train, labels_train) 
	print "training time:", round(time()-t0, 3), "s"

	# prediction
	t0 = time()
	pred = clf.predict(features_test)
	print "predict time:", round(time()-t0, 3), "s"

	print 'total events:',pred.size
	print 'predicted events in "Chris" (1) class', np.count_nonzero(pred)

	# accuracy
	t0 = time()
	acc = accuracy_score(pred, labels_test)
	print "accuracy time:", round(time()-t0, 3), "s"
	print "accuracy:", acc
