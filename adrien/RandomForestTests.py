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
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression, mutual_info_regression
import utils
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import warnings

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

def make_submission(y_predict, user_movie_ids, file_name='submission',
                date=True):
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
	# Replace NaN with mean of the feature
    n = data.isna().any()
    col = data.columns[n]
    
    for item in col:
        mean = data[item].mean()

        if pd.isnull(mean):
            data.drop(item, axis=1, inplace=True)
        else:
            data[item] = data[item].fillna(mean)
        
    # data = data.fillna(0)
    return data

def make_dataset(userMoviePairPath, includeY):
    userMoviePair = load_from_csv(userMoviePairPath)

    userFeatures = load_from_csv("data/data_user.csv")
    movieFeatures = load_from_csv("data/data_movie.csv")


    dataset = userMoviePair

    if(includeY):
        y = load_from_csv("data/output_train.csv")
        dataset = pd.concat([dataset, y], axis = 1, sort = False)

    dataset = pd.merge(dataset, userFeatures, on = "user_id")
    dataset = pd.merge(dataset, movieFeatures, on = "movie_id")

    return dataset

def make_train_set():
    dataset = make_dataset("data/data_train.csv", True)

    y = dataset["rating"].to_frame()
    x = dataset.drop("rating", axis = 1)

    x = x.drop("IMDb_URL", axis = 1)
    x = x.drop("movie_title", axis = 1)
    x = x.drop("zip_code", axis = 1)

    x["release_date"] = [utils.dateConverter(date) for date in x["release_date"]]
    x["gender"] = [utils.genderConverter(item) for item in x["gender"]]

    one_hot = pd.get_dummies(x['occupation'])
    x = x.drop("occupation", axis = 1)
    x = x.join(one_hot)

    return clean(x), y

def make_test_set():
    dataset = make_dataset("data/data_test.csv", False)

    dataset["release_date"] = [utils.dateConverter(date) for date in dataset["release_date"]]
    dataset["gender"] = [utils.genderConverter(item) for item in dataset["gender"]]

    dataset = dataset.drop("IMDb_URL", axis = 1)
    dataset = dataset.drop("movie_title", axis = 1)
    dataset = dataset.drop("zip_code", axis = 1)

    one_hot = pd.get_dummies(dataset['occupation'])
    dataset = dataset.drop("occupation", axis = 1)
    dataset = dataset.join(one_hot)

    return clean(dataset)


if __name__ == '__main__':
    prefix = 'data/'
    plotFolder = "graphs/"


    # ------------------------------- Learning ------------------------------- #
    # Build the learning matrix
    X_ls, y_ls = make_train_set()

    X_ls2 = X_ls.drop("movie_id", axis = 1)
    X_ls2 = X_ls2.drop("user_id", axis = 1)

    trainX, testX, trainY, testY = train_test_split(X_ls2, y_ls)

    # Build the model
    start = time.time()

    with measure_time('Training'):
        print('Training...')

        trainY = np.ravel(trainY)
        model = RandomForestRegressor(n_estimators=100)
        model.fit(trainX, trainY)

    # ------------------------------ Prediction ------------------------------ #
    # Predict
    testY = np.ravel(testY)
    score = -1 * np.mean(cross_val_score(model, testX, testY, n_jobs=-1, cv=10, scoring='neg_mean_squared_error'))
    print(score)
    