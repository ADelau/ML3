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
from scipy import sparse
#import utils
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

def make_dataset(userMoviePairPath):
    userMoviePair = load_from_csv(userMoviePairPath)
    ratings = load_from_csv("data/output_train.csv")
    userFeatures = load_from_csv("data/data_user.csv")
    movieFeatures = load_from_csv("data/data_movie.csv")
    
    uMTriplets = pd.concat([userMoviePair, ratings], axis = 1, sort = False)
    uMTriplets = clean(uMTriplets)
    uMTriplets = uMTriplets.values
    
    user_id = uMTriplets[:,0]
    movie_id = uMTriplets[:,1]
    ratings = uMTriplets[:,2]
    #userRatingsMatrix=sparse.csc_matrix((ratings,(movie_id,user_id)))
    #movieRatingsMatrix=sparse.csc_matrix((ratings,(user_id,movie_id)))
    #userRatings=pd.DataFrame(userRatings_matrix)
    #movieRatings=pd.DataFrame(movieRatings_matrix)

    userRatings = []
    for _ in range(len(userFeatures)):
        userRatings.append([])

    movieRatings = []
    for _ in range(len(movieFeatures)):
        movieRatings.append([])
   
    for i in range(len(ratings)):
        userRatings[user_id[i] - 1].append(ratings[i])
        movieRatings[movie_id[i] - 1].append(ratings[i])
        
    userRatings=pd.DataFrame(userRatings).set_index(userFeatures["user_id"])
    movieRatings=pd.DataFrame(movieRatings).set_index(movieFeatures["movie_id"])

    userFeatures = clean(userFeatures)
    movieFeatures = clean(movieFeatures)

    dataset = pd.merge(userMoviePair, userRatings, on ="user_id")
    dataset = pd.merge(dataset, movieRatings, on = "movie_id")

    dataset = clean(dataset)
    dataset = pd.merge(dataset, userFeatures, on = "user_id")
    dataset = pd.merge(dataset, movieFeatures, on = "movie_id")
    
    return userRatings

def make_train_set():
    dataset = make_dataset("data/data_train.csv")

    y = dataset["rating"].to_frame()
    x = dataset.drop("rating", axis = 1)
    x = x.drop("IMDb_URL", axis = 1)
    x = x.drop("zip_code", axis = 1)

    x["release_date"] = [utils.dateConverter(date) for date in x["release_date"]]
    x["gender"] = x["gender"].astype("category")
    x["occupation"] = x["occupation"].astype("category")

    polarity, subjectivity = utils.stringAnalyzer(x["movie_title"])
    x = x.drop("movie_title", axis = 1)
    x = pd.concat(objs = [x, pd.Series(data = polarity, name = "polarity")], axis = 1, sort = False)
    x = pd.concat(objs = [x, pd.Series(data = subjectivity, name = "subjectivy")], axis = 1, sort = False)

    return x, y

if __name__ == '__main__':
    prefix = 'data/'

    dataset = make_dataset("data/data_train.csv")
    print(dataset)
#    # ------------------------------- Learning ------------------------------- #
#    # Build the learning matrix
#    X_ls, y_ls = make_train_set()
#    X_ls = X_ls.drop("user_id", axis = 1)
#    X_ls = X_ls.drop("movie_id", axis = 1)
#    X_ls = encode(X_ls)
#
#    trainX, testX, trainY, testY = train_test_split(X_ls, y_ls, random_state=42)
#
#    scores = []
#    kCross = 3
#    nFeatures = range(1, 25, 2)
#    # nFeatures = [1, 5, 9, 13, 17, 21, 24]
#
#    # Build the model
#    start = time.time()
#
#    with measure_time('Training'):
#        print('Training...')
#        
#        y = np.ravel(trainY)
#        model = RandomForestRegressor(n_estimators=10, max_depth=20)
#        model.fit(trainX, y)