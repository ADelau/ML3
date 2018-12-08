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
    x = x.drop("IMDb_URL", axis = 1)
    x = x.drop("movie_title", axis = 1)
    x = x.drop("zip_code", axis = 1)
    x = x.drop("occupation", axis = 1)

    x["release_date"] = [utils.dateConverter(date) for date in x["release_date"]]
    x["gender"] = [utils.genderConverter(item) for item in x["gender"]]

    return x, y


if __name__ == '__main__':
    prefix = 'data/'
    plotFolder = "graphs/"


    # ------------------------------- Learning ------------------------------- #
    # Build the learning matrix
    X_ls, y_ls = make_train_set()
    # print(X_ls.shape)
    # print(X_ls.columns.values)

    scores = []
    kCross = 3
    nFeatures = range(1, 25, 2)
    # nFeatures = [1, 5, 9, 13, 17, 21, 24]

    # Build the model
    start = time.time()

    file  = open(plotFolder + "featuresSelec.txt", "w")

    with measure_time('Training'):
        print('Training...')
        for n in nFeatures:
            trainX, testX, trainY, testY = train_test_split(X_ls, y_ls)

            y = np.ravel(trainY)
            selector = SelectKBest(f_regression, k=n)
            selector.fit(trainX, y)

            # Get idxs of columns to keep
            cols = selector.get_support(indices=True)
            # Create new dataframe with only desired column
            testX = testX[trainX.columns[cols]]

            model = DecisionTreeRegressor(max_depth = 10)
            s = -1 * np.mean(cross_val_score(model, testX, testY, n_jobs=-1, cv=kCross, scoring='neg_mean_squared_error'))
            print("number = ", n, "score = ", s)
            scores.append(s)

            file.write("N Features = {}, cols = {} \n".format(n, testX.columns.values))

    file.close()

    plt.plot(nFeatures, scores, label="Mean Squared Error")
    plt.title("Cross Validation Score")
    plt.legend()

    plt.savefig(plotFolder + "crossValScore.pdf")


    