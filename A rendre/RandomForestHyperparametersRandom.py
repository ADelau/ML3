# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import os
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split

import json

import utils


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
    """
    Fill NaN in data with 0

    Parameters
    ----------
    data: a data Dataframe

    Return
    ------
    the DataFrame with its NaN filled with 0
    """

    nanProportionToDrop = 1/len(data.columns)
    data = data.dropna(axis = 1, thresh = nanProportionToDrop*len(data.index))
    data = data.dropna(axis = 0, how = "any")

    return data 

def make_dataset(userMoviePairPath, includeY):
    """
    Create the dataset from csv files

    Parameters
    ----------
    userMoviePairPath: the path to the csv file containg pairs of user/movie id's
    includeY: True to inclue rating Y, false otherwise

    Return
    ------
    the dataset in a DataFrame
    """

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
    """
    Create the train set

    Return
    ------
    the training dataset in a DataFrame
    """

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
    """
    Evaluate a model with the MSE

    Parameters
    ----------
    model: a trained ML model with the "predict" function
    x: the features to predict
    y: the real results of the predictions

    Return
    ------
    the MSE
    """

    predictions = model.predict(x)
    mse = (predictions - y)**2
    
    return np.mean(mse)

if __name__ == '__main__':
    """
    Script used to find best hyperparameters of RandomForest with RandomSearch
    """

    prefix = 'data/'
    plotFolder = "graphs/"
    fileName = '{}_{}.txt'.format("forestParamTuningSelectRandom", time.strftime('%d-%m-%Y_%Hh%M'))
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

    # Test without parameters tuning
    with measure_time('Simplest Forest'):
        rf = RandomForestRegressor()
        rf.fit(trainX, trainY)
        simpleScore = evaluate(rf, testX, testY)
        file.write("No parameters score : {} \n".format(simpleScore))
        

    # Number of trees in random forest, not to optimize as increasing it can only increase accuracy 
    n_estimators = [100]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 5)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10, 20, 50]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'bootstrap': bootstrap}
    print(random_grid)

    with measure_time('Random Search'):

        # Use the random grid to search for best hyperparameters
        # First create the base model to tune
        rf = RandomForestRegressor()
        # Random search of parameters, using 3 fold cross validation, 
        # search across 100 different combinations, and use all available cores
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter =100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
        # Fit the random search model
        rf_random.fit(trainX, trainY)
        score = evaluate(rf_random, testX, testY)
        
    print("Random score : ", score)
    file.write("Random score : {} \n".format(score))
    file.write(json.dumps(rf_random.best_params_))
    file.write("\n")


    file.close()
