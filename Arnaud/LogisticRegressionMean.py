# ! /usr/bin/env python
# -*- coding: latin_1 -*-

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time
import math
from sklearn.utils import shuffle


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

def myToordinal(date):
	return date.toordinal()

def make_dataset(toSplit = False, splitRatio = 0.2):
	userMovieTrainPair = load_from_csv("data/data_train.csv")
	userFeatures = load_from_csv("data/data_user.csv")
	movieFeatures = load_from_csv("data/data_movie.csv")

	userMovieTrainPair = userMovieTrainPair.astype("category")
	userFeatures = userFeatures.astype("category")
	movieFeatures = movieFeatures.astype("category")

	userFeatures = userFeatures.drop("zip_code", axis = 1)
	userFeatures["age"] = userFeatures["age"].astype("int64")
	
	movieFeatures = movieFeatures.drop("movie_title", axis = 1)
	movieFeatures = movieFeatures.drop("IMDb_URL", axis = 1)
	movieFeatures = movieFeatures.drop("video_release_date", axis = 1)

	movieFeatures = clean(movieFeatures)
	movieFeatures["release_date"] = movieFeatures["release_date"].astype("datetime64")
	movieFeatures["release_date"] = movieFeatures["release_date"].map(myToordinal, na_action = "ignore")

	trainDataset = userMovieTrainPair
	
	y = load_from_csv("data/output_train.csv")
	trainDataset = pd.concat([trainDataset, y], axis = 1, sort = False)
	trainDataset = pd.merge(trainDataset, userFeatures, on = "user_id")
	trainDataset = pd.merge(trainDataset, movieFeatures, on = "movie_id")

	if toSplit:
		trainLength = math.floor(splitRatio * len(trainDataset))
		trainDataset = shuffle(trainDataset)
		testDataset = trainDataset[trainLength:]
		trainDataset = trainDataset[:trainLength]

	else:
		testDataset = load_from_csv("data/data_test.csv")
		testDataset = testDataset.astype("category")
		testDataset = pd.merge(testDataset, userFeatures, on = "user_id")
		testDataset = pd.merge(testDataset, movieFeatures, on = "movie_id")

	userMeanRating = trainDataset.groupby("user_id", as_index = False).rating.agg("mean")
	userMeanRating.rename(index = str, columns = {"rating": "userMeanRating"}, inplace = True)
	movieMeanRating = trainDataset.groupby("movie_id", as_index = False).rating.agg("mean")
	movieMeanRating.rename(index = str, columns = {"rating": "movieMeanRating"}, inplace = True)

	trainDataset = pd.merge(trainDataset, userMeanRating, on = "user_id")
	trainDataset = pd.merge(trainDataset, movieMeanRating, on = "movie_id")
	trainDataset = trainDataset.drop("user_id", axis = 1)
	trainDataset = trainDataset.drop("movie_id", axis = 1)

	testDataset = pd.merge(testDataset, userMeanRating, on = "user_id")
	testDataset = pd.merge(testDataset, movieMeanRating, on = "movie_id")
	testDataset = testDataset.drop("user_id", axis = 1)
	testDataset = testDataset.drop("movie_id", axis = 1)

	trainDataset = clean(trainDataset)
	testDataset.fillna(method = "ffill", inplace = True)

	trainY = trainDataset["rating"].to_frame()
	trainX = trainDataset.drop("rating", axis = 1)
	
	if toSplit:
		testY = testDataset["rating"].to_frame()
		testX = testDataset.drop("rating", axis = 1)

		return trainX, trainY, testX, testY

	return trainX, trainY, testDataset

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

	trainX.info()
	trainY.info()
	predictX.info()

	print("fitting ...")
	classifier = LogisticRegression(max_iter = 1000, tol = 0.0001, multi_class = "multinomial", n_jobs = -1, C = complexity, solver = "lbfgs") #changer et mettre auto
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
	print("building dataset...")
	trainX, trainY, predictX = make_dataset();

	predictY = make_prediction(trainX, trainY, predictX)
	
	print("making submission ...")
	make_submission(predictY)

def compute_accuracy():
	print("building dataset...")
	trainX, trainY, testX, testY = make_dataset(True, 0.2)

	C_PARAM_RANGE = [1]
	for C in C_PARAM_RANGE:	
		predictY = make_prediction(trainX, trainY, testX, complexity = C)
		print("mean_squared_error for c = {} is {}".format(C, mean_squared_error(testY, predictY)))

if __name__ == "__main__":
	compute_accuracy()