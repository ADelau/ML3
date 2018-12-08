# ! /usr/bin/env python
# -*- coding: latin_1 -*-

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

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

	return dataset

def make_train_set():
	dataset = make_dataset("data/data_train.csv", True)

	y = dataset["rating"].to_frame()
	x = dataset.drop("rating", axis = 1)

	return x, y

def make_predict_set():
	return make_dataset("data/data_test.csv", False)

def make_prediction(trainX, trainY, predictX):
    trainX = trainX
    trainY = trainY
    predictX = predictX

    trainX["gender"] = trainX["gender"].astype("category")
    trainX["occupation"] = trainX["occupation"].astype("category")
    trainX["zip_code"] = trainX["zip_code"].astype("category")
    trainX["movie_title"] = trainX["movie_title"].astype("category")
    trainX["release_date"] = trainX["release_date"].astype("category")
    trainX["IMDb_URL"] = trainX["IMDb_URL"].astype("category")

    """
    trainX = trainX.drop("gender", axis = 1)
    trainX = trainX.drop("occupation", axis = 1)
    trainX = trainX.drop("zip_code", axis = 1)
    trainX = trainX.drop("movie_title", axis = 1)
    trainX = trainX.drop("release_date", axis = 1)
    trainX = trainX.drop("IMDb_URL", axis = 1)
    """

    trainX.info()
    trainY.info()
    
    print("fitting ...")
    max_depth= [8, 10, 20, 30, 40, 50]
    n_estimators= [10, 50, 100, 150, 300]
    max_features = [2, 5 ,10]
    param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth}
    classifier = RandomForestClassifier()
    classifier_search = RandomizedSearchCV(estimator=classifier, param_distributions=param_grid, n_iter=50, n_jobs=-1, cv=3, random_state=11)
    classifier_search.fit(trainX, trainY)
    
    print(classifier_search.best_params_)
    
    print("predicting ...")
    predictY = classifier.predict(predictX)
	
    feature_importances = pd.DataFrame(classifier.feature_importances_, index = trainX.columns, columns=["Importance"])
    print(feature_importances)
    
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
	predictY = make_prediction(trainX, trainY, testX)
	return accuracy_score(testY, predictY)

if __name__ == "__main__":
	print(compute_accuracy())