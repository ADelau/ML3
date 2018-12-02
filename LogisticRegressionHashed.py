# ! /usr/bin/env python
# -*- coding: latin_1 -*-

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time

NB_USER_FEATURES = 100

def load_from_csv(path, delimiter=','):
    """
    Load csv file and return a NumPy array of its data

    Parameters
    ----------
    path: str
        The path to the csv file to load
    delimiter: str (default: ',')
        The csv field delimiter

    Return
    ------
    D: array
        The NumPy array of the data contained in the file
    """
    return pd.read_csv(path, delimiter=delimiter, encoding = "latin_1")

def encode(data):
	data = data.astype("category")
	"""
	le = LabelEncoder()
	print(data.index)
	for i in range(len(data.columns)):
		print(data.values[:,i])
		print(type(data.values[0,i]))
		data.values[:,i] = le.fit_transform(data.values[:,i])
	"""
	data = pd.get_dummies(data, sparse = True)

	return data

def clean(data):
	nanProportionToDrop = 1/len(data.columns)
	data = data.dropna(axis = 1, thresh = nanProportionToDrop*len(data.index))
	data = data.dropna(axis = 0, how = "any")

	return data

def make_dataset(userMoviePairPath, includeY):
	userMoviePair = load_from_csv(userMoviePairPath)
	userFeatures = clean(load_from_csv("data/data_user.csv"))
	movieFeatures = clean(load_from_csv("data/data_movie.csv"))	

	dataset = userMoviePair

	if(includeY):
		y = load_from_csv("data/output_train.csv")
		dataset = pd.concat([dataset, y], axis = 1, sort = False)

	dataset = clean(dataset)
	dataset = pd.merge(dataset, userFeatures, on = "user_id")
	dataset = pd.merge(dataset, movieFeatures, on = "movie_id")
	dataset["user_id"] = dataset["user_id"].map(lambda x : x % NB_USER_FEATURES)
	
	return dataset

def make_train_set():
	dataset = make_dataset("data/data_train.csv", True)

	y = dataset["rating"].to_frame()
	x = dataset.drop("rating", axis = 1)

	return x, y

def make_predict_set():
	return make_dataset("data/data_test.csv", False)

def make_prediction(trainX, trainY, predictX, complexity = 1.0):
	print("encoding ...")
	trainLength = len(trainX)
	
	toEncode = pd.concat(objs = [trainX, predictX], axis = 0, sort = False, ignore_index = True)
	encoded = encode(toEncode)
	
	trainX = encoded[:trainLength]
	predictX = encoded[trainLength:]

	trainY = trainY

	print("fitting ...")
	#classifier = LogisticRegressionCV(cv = 3, max_iter = 10, tol = 0.1, multi_class = "auto", n_jobs = -1)
	classifier = LogisticRegression(max_iter = 1000, tol = 0.0001, multi_class = "auto", n_jobs = -1, C = complexity, solver = "lbfgs")
	classifier.fit(trainX, trainY)

	print("predicting ...")
	predictY = classifier.predict(predictX)
	return predictY

def make_submission(y_predict, file_name='submission',
                    date=True):
	
	print(y_predict)
	"""
	Write a submission file for the Kaggle platform

	Parameters
	----------
	y_predict: array [n_predictions]
	    The predictions to write in the file. `y_predict[i]` refer to the
	    user `user_ids[i]` and movie `movie_ids[i]`
	user_movie_ids: array [n_predictions, 2]
	    if `u, m = user_movie_ids[i]` then `y_predict[i]` is the prediction
	    for user `u` and movie `m`
	file_name: str or None (default: 'submission')
	    The path to the submission file to create (or override). If none is
	    provided, a default one will be used. Also note that the file extension
	    (.txt) will be appended to the file.
	date: boolean (default: True)
	    Whether to append the date in the file name

	Return
	------
	file_name: path
	    The final path to the submission file
	"""

	# Naming the file

	"""
	user_movie_ids = load_from_csv("data/data_test.csv")

	if date:
	    file_name = '{}_{}'.format(file_name, time.strftime('%d-%m-%Y_%Hh%M'))

	file_name = '{}.txt'.format(file_name)

	# Writing into the file
	with open(file_name, 'w') as handle:
	    handle.write('"USER_ID_MOVIE_ID","PREDICTED_RATING"\n')
	    for (user_id, movie_id), prediction in zip(user_movie_ids,
	                                             y_predict):

	        if np.isnan(prediction):
	            raise ValueError('The prediction cannot be NaN')
	        line = '{:d}_{:d},{}\n'.format(user_id, movie_id, prediction)
	        handle.write(line)
	return file_name
	"""

def predict_matrix():
	print("building train set...")
	trainX, trainY = make_train_set();

	print("building test set...")
	predictX = make_predict_set();

	predictY = make_prediction(trainX, trainY, predictX)
	
	print("making submission ...")
	make_submission(predictY)

def compute_accuracy():
	print("building dataset...")
	x, y = make_train_set()
	trainX, testX, trainY, testY = train_test_split(x, y)

	#C_PARAM_RANGE = [0.001, 0.01, 0.1, 1, 10, 100]
	C_PARAM_RANGE = [0.01]
	for C in C_PARAM_RANGE:	
		predictY = make_prediction(trainX, trainY, testX, complexity = C)
		print("accuracy for c = {} is {}".format(C, mean_squared_error(testY, predictY)))

if __name__ == "__main__":
	compute_accuracy()