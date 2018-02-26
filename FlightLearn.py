import numpy as np
import csv
from datetime import date
import random
from sklearn import linear_model
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv('BOS_CUN_trips1M.csv')

def add_features():

	'''
	This section is a list of helper functions for further optimizing code
	-------------------------------------------------------------------------------------------------------------------------------------------------
	'''
	
	def create_dict(id_list):
		#creates a dictionary for relevant one-hot categorical vectors
		id_dict={}
		for i in range(len(id_list)):
			id_dict[id_list[i]]=i
		return id_dict

	def total_days(departure, order):
		#calculates the total days between order date and departure date and changes the raw value to categorical data
		total_days=departure.sub(order)
		total_days.rename(columns={0:'total days'}, axis='columns')
		total_days.astype('timedelta64[D]')
		total_days=total_days.apply(lambda x: x.days)
		total_days=pd.cut(total_days, bins=12)
		return pd.get_dummies(total_days)

	def one_hot(features, feature_list, prefixes):
		#creates one-hot vectors for all the categorical data
		for i in range(len(feature_list)):
			if type(feature_list[i])==str:
				feature_vector=pd.get_dummies(data[feature_list[i]], prefix=prefixes[i])
			else:
				feature_vector=pd.get_dummies(feature_list[i], prefix=prefixes[i])
			features=pd.concat([features,feature_vector], axis=1)
		return features

	'''
	-------------------------------------------------------------------------------------------------------------------------------------------------

	This initializes many of the labels for the data frames and certain dates into date time as well as lists to help shorten and optimize code length

    ------------------------------------------------------------------------------------------------------------------------------------------------------

	'''

	monthsDepart=['Depart January', 'Depart February', 'Depart March', 'Depart April', 'Depart May', 'Depart June', 'Depart July', 'Depart August', 'Depart September', 'Depart October', 'Depart November', 'Depart December']
	monthsReturn=['Return January', 'Return February', 'Return March', 'Return April', 'Return May', 'Return June', 'Return July', 'Return August', 'Return September', 'Return October', 'Return November', 'Return December']
	days_of_weekD=['Depart Monday', 'Depart Tuesday', 'Depart Wednesday', 'Depart Thursday', 'Depart Friday', 'Depart Saturday','Depart Sunday']
	days_of_weekR=['Return Monday', 'Return Tuesday', 'Return Wednesday', 'Return Thursday', 'Return Friday', 'Return Saturday','Return Sunday']

	#creates dictionary of carrier ids
	carrier_ids=create_dict(data.majorcarrierid.unique())

	#creates dictionary of cabin classes
	cabin_ids=create_dict(data.cabinclass.unique())

	#creates dictionary of sources
	source_ids=create_dict(data.source.unique())

	#converting dates to date_time
	order_date=pd.to_datetime(data['received_odate'])
	departure_date=pd.to_datetime(data['departure_odate'])
	return_date=pd.to_datetime(data['return_ddate'])

	#getting the month of departure and return
	departure_month=pd.DatetimeIndex(departure_date).month
	return_month=pd.DatetimeIndex(return_date).month

	#categorical features that will be transferred to one-hot vectors
	one_hot_feature_set=['majorcarrierid', 'cabinclass', 'source', 'departure_dow', 'return_dow', departure_month, return_month]

	prefixes=[carrier_ids, cabin_ids, source_ids, days_of_weekD, days_of_weekR, monthsDepart, monthsReturn]

	#features of continuous data
	feature_list=['outbounddurationminutes', 'returndurationminutes', 'outboundstops', 'los2', 'refundable', 'includes_sns', 'total']


	'''
    ------------------------------------------------------------------------------------------------------------------------------------------------
	'''


	features=total_days(return_date, order_date)

	features=one_hot(features, one_hot_feature_set, prefixes)

	for feature in feature_list:
		features=pd.concat([features, data[feature]], axis=1)

	return features

def train(visualize=False):

	print('Adding Features...')

	features=add_features()

	print('Features Added!')

	print('Building training and test sets...')

	train, test = train_test_split(features, test_size=0.2)

	train, _ = train_test_split(train, test_size=0.01)

	test, _ = train_test_split(test, test_size=0.01)

	train_labels= train[train.columns[-1]]
	train.drop(labels=train.columns[0], axis=1, inplace=True)
	train.drop(labels=train.columns[-1], axis=1, inplace=True)

	test_labels= test[test.columns[-1]]
	test.drop(labels=train.columns[0], axis=1, inplace=True)
	test.drop(labels=test.columns[-1], axis=1, inplace=True)

	scaler=StandardScaler()

	train_scale=scaler.fit(train)
	train=train_scale.fit_transform(train)

	test_scale=scaler.fit(test)
	test=test_scale.fit_transform(test)

	print('Training and test sets built!')

	print('Training model...')


	linreg=linear_model.Ridge(alpha=0.000001)

	print('Model Trained!')

	print('Running test sets!')

	linreg.fit(train, train_labels)

	print('Test sets run, here is your score:')

	if visualize:
		train_graph, validation_graph=validation_curve(linreg, train, train_labels, param_name='alpha', param_range=(0.1, 1.0, 10.0))
		plt.plot(train_graph)
		plt.plot(validation_graph)
		plt.show()

	test_prices=linreg.predict(test)
	sum1=0
	for i in range(test_prices.shape[0]):
		sum1+=test_prices[0]
	expected_price=sum1/test_prices.shape[0]

	return (linreg.score(test, test_labels),expected_price)

