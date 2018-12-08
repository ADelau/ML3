# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import datetime
from contextlib import contextmanager


import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression, mutual_info_regression
import utils
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import json

@contextmanager
def measure_time(label):
    """
    Context manager to measure time of computation.
    >>> with measure_time('Heavy computation'):
    >>>     do_heavy_computation()
    'Duration of [Heavy computation]: 0:04:07.765971'

    Parameters
    ----------
    label: str
        The label by which the computation will be referred
    """
    start = time.time()
    yield
    end = time.time()
    print('Duration of [{}]: {}'.format(label,
                                        datetime.timedelta(seconds=end-start)))


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

	return dataset

def make_train_set():
    dataset = make_dataset("data/data_train.csv", True)

    y = dataset["rating"].to_frame()
    x = dataset.drop("rating", axis = 1)

    x["release_date"] = [utils.dateConverter(date) for date in x["release_date"]]
    x["gender"] = [utils.genderConverter(item) for item in x["gender"]]

    one_hot = pd.get_dummies(x['occupation'])
    x = x.drop("occupation", axis = 1)
    x = x.join(one_hot)

    x = x.drop("IMDb_URL", axis = 1)
    x = x.drop("movie_title", axis = 1)
    x = x.drop("zip_code", axis = 1)

    return x, y


def evaluate(model, x, y):
    predictions = model.predict(x)
    mse = (predictions - y)**2
    
    return np.mean(mse)

if __name__ == '__main__':
    prefix = 'data/'
    plotFolder = "graphs/"
    fileName = '{}_{}.txt'.format("forestParamTuningSelectGrid", time.strftime('%d-%m-%Y_%Hh%M'))
    file  = open(fileName, "w")


    # ------------------------------- Learning ------------------------------- #
    # Build the learning matrix
    X_ls, y_ls = make_train_set()
    X_ls = X_ls.drop("user_id", axis = 1)
    X_ls = X_ls.drop("movie_id", axis = 1)
    X_ls = X_ls.drop("unknown", axis = 1)

    trainX, testX, trainY, testY = train_test_split(X_ls, y_ls, random_state=42)
    trainY = np.ravel(trainY)
    testY = np.ravel(testY)

    with measure_time('Simplest Forest'):
        rf = RandomForestRegressor()
        rf.fit(trainX, trainY)
        simpleScore = evaluate(rf, testX, testY)
        file.write("No parameters score : {} \n".format(simpleScore))
        

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 5)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt', 4, 8]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 5)]
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10, 20, 50]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    param_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'bootstrap': bootstrap}

    with measure_time('Random Search'):

        # Use the random grid to search for best hyperparameters
        # First create the base model to tune
        rf = RandomForestRegressor()
        grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = 4, verbose = 2)
        # Fit the random search model
        grid_search.fit(trainX, trainY)
        score = evaluate(grid_search, testX, testY)

    print("Grid score : ", score)
    file.write("Grid score : {} \n".format(score))
    file.write(json.dumps(grid_search.best_params_))
    file.write("\n")

    file.close()
