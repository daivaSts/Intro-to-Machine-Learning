#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training and testing datasets, 
### respectively labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

### Set min_samples_split=40. It will probably take a while to train. 
def classify(features_train, labels_train,n):
    clf = DecisionTreeClassifier(min_samples_split=n)

    t0 = time()
    clf.fit(features_train, labels_train) 
    print "training time:", round(time()-t0, 3), "s"

    t0 = time()
    pred = clf.predict(features_test)
    print "prediction time:", round(time()-t0, 3), "s"

    t0 = time()
    acc = accuracy_score(pred, labels_test)
    print "accuracy time:", round(time()-t0, 3), "s"

    return acc
    
acc_min_samples_split = classify(features_train, labels_train,40)

### What is the accuracy?
print 'accuracy: ', acc_min_samples_split

### What is the number of features in your data?
print 'number of features', len(features_train[0])

### go into ../tools/email_preprocess.py, and find the line of code: selector = SelectPercentile(f_classif, percentile=10). 
### Change percentile from 10 to 1, and rerun dt_author_id.py. What is the number of features now?
print 'number of features with percentile 1', len(features_train[0])

### Would a large value for percentile lead to a more complex or less complex decision tree, all other things being equal?

### What's the accuracy of your decision tree when you use only 1% of your available features (i.e. percentile=1)?
print 'accuracy with 1%  of features: ', acc_min_samples_split
