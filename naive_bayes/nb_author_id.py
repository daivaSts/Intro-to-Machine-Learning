#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train are features for the training
### features_test are features for testing datasets
### labels_train are the train labels 
### labels_test are the test labels

features_train, features_test, labels_train, labels_test = preprocess()

def NBAccuracy(features_train, labels_train, features_test, labels_test):
    """ compute the accuracy of your Naive Bayes classifier """
    ### import the sklearn module for GaussianNB
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score

    ### create classifier
    clf = GaussianNB()

    ### fit the classifier on the training features and labels. 
    t0 = time()
    fit_classifier = clf.fit(features_train, labels_train)
    print "training time:", round(time()-t0, 3), "s"

    ### use the trained classifier to predict labels for the test features
    t0 = time()
    pred = clf.predict(features_test)
    print "predict time:", round(time()-t0, 3), "s"

    ### calculate and return the accuracy on the test data

    ## V1
    #accuracy = clf.score(features_test, labels_test)

    ## V2
    accuracy = accuracy_score(labels_test, pred)

    return accuracy


NBAccuracy(features_train, labels_train, features_test, labels_test)



