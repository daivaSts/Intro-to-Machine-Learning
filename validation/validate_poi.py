#!/usr/bin/python
"""
    Starter code for the validation mini-project. The first step toward building your POI identifier!
    Start by loading/formatting the data
    After that, it's not our code anymore--it's yours!
"""
import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

### first element is our labels, any added elements are predictor features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)
# print len(labels)
# print len (features)
# print len(data)


### Create a decision tree classifier, train it on all the data, and print out the accuracy
def classify_all(features, labels):
    clf = DecisionTreeClassifier()
    clf.fit(features, labels) 
    pred = clf.predict(features)
    acc = accuracy_score(pred, labels)
    return acc
    
dt_accuracy_all = classify_all(features, labels)
print "DecisionTreeClassifier accuracy all data: {}\n".format(dt_accuracy_all)

### add in training and testing, to get a trustworthy accuracy number. random_state controls which points go into the training set and 
### which are used for testing; setting it to 42 means we know exactly which events are in which set, and can check the results
feature_train, feature_test, label_train, label_test = train_test_split(features, labels, test_size=0.3, random_state=42)

def classify(feature_train, feature_test, label_train, label_tes):
    clf = DecisionTreeClassifier()
    clf.fit(feature_train, label_train) 
    pred = clf.predict(feature_test)
    acc = accuracy_score(pred, label_test)
    return acc
    
dt_accuracy = classify(feature_train, feature_test, label_train, label_test)
print "DecisionTreeClassifier accuracy data: {}\n".format(dt_accuracy)
