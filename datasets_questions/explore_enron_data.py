#!/usr/bin/python
""" 
    Starter code for exploring the Enron dataset (emails + finances); 
    loads up the dataset (pickled dict of dicts).

    The dataset has the form: enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project, but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""
import sys
sys.path.append("../final_project/")
sys.path.append("../tools/")
import pickle
from poi_email_addresses import poiEmails
from feature_format import featureFormat
from feature_format import targetFeatureSplit



#enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
enron_data_handler = open("../final_project/final_project_dataset.pkl", "r")
enron_data = pickle.load(enron_data_handler)
enron_data_handler.close()

poi_data_handler = open("../final_project/poi_names.txt", "r")

### sample
print enron_data['LAY KENNETH L']

if False:
	### How many data points (people) are in the Enron dataset?
	### For each person, how many features are available?

	print 'Dataset has {} people'.format(len(enron_data))	
	for key in enron_data.keys():
		name = key
	
	print 'There are {} features available for each person'.format(len(enron_data[name]))
	

if False:
	### The 'poi' feature records whether the person is a person of interest. How many POIs are there in the E+F dataset?
	### We compiled a list of all POI names (in ../final_project/poi_names.txt) and associated email addresses
	### How many POI's were there total?
	count = 0
	for i in poi_data_handler:
		if i.startswith('('):	
			count += 1

	print 'There are {} POIs'.format(count)	
	poi_data_handler.close()

if False:
	### The 'poi' feature records whether the person is a person of interest. How many POIs are there in the E+F dataset?
	count = 0
	for value in enron_data.values():
		if value['poi'] == True:
			count += 1
	print 'There are {} "poi" feature records'.format(count)	

if False:
	
	for key in enron_data.keys():
		if 'Prentice'.upper() in key:
			name1 = key
		if 'Colwell'.upper() in key:
			name2 = key	
		if 'Skilling'.upper() in key:
			name3 = key	

	print enron_data[name1]

	### What is the total value of the stock belonging to James Prentice?		
	print 'the total value of the stock belonging to James Prentice is {}'.format(enron_data[name1]['total_stock_value'])

	### How many email messages do we have from Wesley Colwell to persons of interest?
	print 'there are {} email messages from Wesley Colwell to persons of interest'.format(enron_data[name2]['from_this_person_to_poi'])

	### What is the value of stock options exercised by Jeffrey Skilling?
	print 'the value of stock options exercised by Jeffrey Skilling is {}'.format(enron_data[name3]['exercised_stock_options'])

	### Of these three individuals (Lay, Skilling and Fastow), who took home the most money (largest value of 'total_payments' feature)? 
	### How much money did that person get?
	print '*******'
	max_earning = 0
	max_eraner = 0
	for j in ['Lay', 'Skilling', 'Fastow']:
		for i in enron_data.keys():
			index = i.find(j.upper())
			if index > -1:
				print '{} received ${}'.format(i.title(), enron_data[i]['total_payments'])	
				if enron_data[i]['total_payments'] > max_earning:
					max_earning = enron_data[i]['total_payments']
					max_eraner = i

	print '{} received largest the amount'.format(max_eraner.title())					
	
if False:	
	### How many folks in this dataset have a quantified salary? What about a known email address?

	count_salary = 0
	count_email = 0 
	for values in enron_data.values():
		if values['salary'] != 'NaN':
			count_salary += 1
		if values['email_address'] != 'NaN':
			count_email += 1

	print 'count of quantified salary is {}'.format(count_salary)
	print 'count of email address is {}'.format(count_email)


if False:	
	### We've written some helper functions (featureFormat() and targetFeatureSplit() in tools/feature_format.py) that can 
	### take a list of feature names and the data dictionary, and return a numpy array.
	### How many POIs in the E+F dataset have 'NaN' for their total payments? What percentage of POI's as a whole is this?

	count_payment_NaN = 0
	count_poi = 0
	count_poi_NaN = 0

	for i in enron_data.keys():		
		if enron_data[i]['total_payments'] != 'NaN':
			count_payment_NaN += 1

		if enron_data[i]['poi'] == True:
			count_poi += 1	
		if enron_data[i]['total_payments'] == 'NaN' and enron_data[i]['poi'] == True:
			count_poi_NaN += 1

	### How many people in the E+F dataset (as it currently exists) have 'NaN' for their total payments? What percentage of 
	### people in the dataset as a whole is this?		
	print '{} people have "NaN" for their total payments'.format(len(enron_data) - count_payment_NaN)
	print 'percentage of all people  - {}%'.format((1 - float(count_payment_NaN)/len(enron_data))*100)

	### How many POIs in the E+F dataset have "NaN" for their total payments? What percentage of POI's as a whole is this?
	print '{} POIs have "NaN" for their total payments'.format(count_poi_NaN)
	print 'percentage of all POIs  - {}%'.format(count_poi_NaN/count_poi)

	### If you added in, say, 10 more data points which were all POI's, and put "NaN" for the total payments for those folks,
	### the numbers you just calculated would change. 
	### What is the new number of people of the dataset? What is the new number of folks with "NaN" for total payments?
	print 'there are total {} people, and {} people with "NaN" total payments'.format(len(enron_data) + 10,len(enron_data) - count_payment_NaN+10)

	### What is the new number of POI's in the dataset? What percentage of them have "NaN" for their total payments?
	print 'there are {} POIs, {} have "NaN" for total payments'.format(count_poi+10, 10)
	

