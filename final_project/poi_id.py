#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import matplotlib.pyplot
import cPickle
import numpy
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.decomposition import RandomizedPCA
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from time import time


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred',\
						 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive',\
						  'restricted_stock', 'director_fees','to_messages', 'from_poi_to_this_person', 'from_messages',\
						   'from_this_person_to_poi', \
					  'shared_receipt_with_poi']

features_list_financial = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred',\
						 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive',\
						  'restricted_stock', 'director_fees'] 
features_list_email = ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'poi',\
					  'shared_receipt_with_poi'] 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


#########################################################
## Task 2: Remove outliers
#########################################################
label_x = features_list[1]
label_y = features_list[2]


#print len(data_dict)
data_dict.pop( 'TOTAL', 0 )
#data_dict.pop( 'LAY KENNETH L', 0)
#print len(data_dict)
data = featureFormat(data_dict, features_list)	 


for point in data:
    salary = point[1]
    bonus = point[2]

    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel(label_x)
matplotlib.pyplot.ylabel(label_y)
#matplotlib.pyplot.show()

#########################################################
### Task 3: Create new feature(s)
#########################################################
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> START

def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """
    ### you fill in this code, so that it returns either
    ###     the fraction of all messages to this person that come from POIs
    ###     or
    ###     the fraction of all messages from this person that are sent to POIs
    ### the same code can be used to compute either quantity

    ### beware of "NaN" when there is no known email address (and so
    ### no filled email features), and integer division!
    ### in case of poi_messages or all_messages having "NaN" value, return 0.
    fraction = 0.
    if poi_messages == 'NaN' or all_messages == 'NaN':
        fraction = 0.
    else:
        fraction = float(poi_messages) / all_messages  
    return fraction


for name in data_dict:

    data_point = data_dict[name]

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    data_point["fraction_from_poi"] = fraction_from_poi

    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    data_point["fraction_to_poi"] = fraction_to_poi


#print data_dict['COLWELL WESLEY']   
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> END
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
features_list.append("fraction_from_poi")
features_list.append("fraction_to_poi")
#print 'features_list: {}'.format(features_list)

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> START

def preprocess(labels, features):
    """ returned:
            -- training/testing features
            -- training/testing labels
    """
    ### test_size is the percentage of events assigned to the test set (remainder go into training)
    features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.25, random_state=42)
    return  features_train, features_test, labels_train, labels_test

features_train, features_test, labels_train, labels_test = preprocess(labels, features)

if False:
	""" compute the accuracy of Naive Bayes classifier """
	clf = GaussianNB()
	t0 = time()
	fit_classifier = clf.fit(features_train, labels_train)
	pred = clf.predict(features_test)

	accuracy = accuracy_score(labels_test, pred)
	recall_sc = recall_score(labels_test, pred)
	precision_sc = precision_score(labels_test, pred)
	f1_sc = f1_score(labels_test, pred)

	print "GaussianNB running time:", round(time()-t0, 3), "s"
	print 'GaussianNB accuracy: {}\n'.format( accuracy)
	print 'GaussianNB recall_score: {}'.format(recall_sc)
	print 'GaussianNB precision_score: {}'.format(precision_sc)
	print 'GaussianNB f1_score: {}\n'.format(f1_sc)

if True:
	""" compute the accuracy of Decision Tree classifier """
	clf = DecisionTreeClassifier(min_samples_split=10)
	t0 = time()
	clf.fit(features_train, labels_train) 
	pred = clf.predict(features_test)

	acc = accuracy_score(pred, labels_test)
	recall_sc = recall_score(labels_test, pred)
	precision_sc = precision_score(labels_test, pred)
	f1_sc = f1_score(labels_test, pred)

	print "Decision Tree running time:", round(time()-t0, 3), "s"
	print 'Decision Tree accuracy: {}\n'.format(acc)
	print 'Decision Tree recall_score: {}'.format(recall_sc)
	print 'Decision Tree precision_score: {}'.format(precision_sc)
	print 'Decision Tree f1_score: {}\n'.format(f1_sc)

if False:
	""" compute the accuracy of AdaBoostClassifier """
	clf = AdaBoostClassifier(algorithm='SAMME.R',n_estimators=300,random_state=0)
	t0 = time()
	clf.fit(features_train, labels_train)
	pred = clf.predict(features_test)

	acc = accuracy_score(pred, labels_test)
	recall_sc = recall_score(labels_test, pred)
	precision_sc = precision_score(labels_test, pred)
	f1_sc = f1_score(labels_test, pred)
	decision = clf.decision_function(features_train)

	print "AdaBoostClassifier running time:", round(time()-t0, 3), "s"
	print 'AdaBoostClassifier accuracy: {}\n'.format(acc)
	print 'AdaBoostClassifier recall_score: {}'.format(recall_sc)
	print 'AdaBoostClassifier precision_score: {}'.format(precision_sc)
	print 'AdaBoostClassifier f1_score: {}\n'.format(f1_sc)


if False:
	""" compute the accuracy of RandomForestClassifier """
	clf = RandomForestClassifier(n_estimators=50,min_samples_split=10,max_depth=20,min_samples_leaf=5)
	t0 = time()
	clf.fit(features_train, labels_train)
	pred = clf.predict(features_test)

	acc = accuracy_score(pred, labels_test)
	precision_sc = precision_score(labels_test, pred)
	f1_sc = f1_score(labels_test, pred)
	decision = clf.decision_function(features_train)
	get_params = clf.get_params()

	print "RandomForestClassifier running time:", round(time()-t0, 3), "s"
	print 'RandomForestClassifier accuracy: {}\n'.format(acc)
	print 'RandomForestClassifier recall_score: {}'.format(recall_sc)
	print 'RandomForestClassifier precision_score: {}'.format(precision_sc)
	print 'RandomForestClassifier f1_score: {}\n'.format(f1_sc)

if True:
	""" compute the accuracy of Support Vector Classification"""

	# min_max_scaler = preprocessing.MinMaxScaler()
	# finance_features_scaled = min_max_scaler.fit_transform(features)

	# features_train, features_test, labels_train, labels_test = preprocess(labels, finance_features_scaled)

	#parameters = {'kernel':('linear', 'rbf'), 'C':[1, 1000], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}
	# svr = SVC()
	# clf = GridSearchCV(svr, parameters)

	clf = SVC(kernel='rbf')
	t0 = time()
	clf.fit(features_train, labels_train) 
	pred = clf.predict(features_test)
	acc = accuracy_score(pred, labels_test)
	recall_sc = recall_score(labels_test, pred)
	precision_sc = precision_score(labels_test, pred)
	f1_sc = f1_score(labels_test, pred)

	print "SVC running time:", round(time()-t0, 3), "s"
	print "SVC accuracy: {}".format( acc)
	print 'SVC recall_score: {}'.format(recall_sc)	
	print 'SVC precision_score: {}'.format(precision_sc)
	print 'SVC f1_score: {}\n'.format(f1_sc)
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> END

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

#dump_classifier_and_data(clf, my_dataset, features_list)