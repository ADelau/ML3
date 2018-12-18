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


if __name__ == '__main__':
    prefix = 'data/'
    plotFolder = "graphs/"


    # ------------------------------- Learning ------------------------------- #
    # Build the learning matrix
    X_ls, y_ls = make_train_set()
    X_ls = X_ls.drop("user_id", axis = 1)
    X_ls = X_ls.drop("movie_id", axis = 1)

    trainX, testX, trainY, testY = train_test_split(X_ls, y_ls, random_state=42)

    scores = []
    kCross = 3
    nFeatures = range(1, 25, 2)
    # nFeatures = [1, 5, 9, 13, 17, 21, 24]

    # Build the model
    start = time.time()

    file  = open(plotFolder + "featuresSelec.txt", "w")

    with measure_time('Training'):
        print('Training...')
        
        y = np.ravel(trainY)
        model = RandomForestRegressor(n_estimators=10, max_depth=20)
        model.fit(trainX, y)

    # scores = [round (x,3) for x in model.feature_importances_]
    scores = [x for x in model.feature_importances_]
    featuresNames = trainX.columns

    for i in range(len(scores)):
        scores[i] = [scores[i], featuresNames[i]]

    # Je sais pas coder en python...
    occupationScore = 0
    tmp = []
    for s in scores:
        score = s[0]
        name = s[1]

        if utils.occupationConverter(name) != -1:
            occupationScore += score
        else:
            tmp.append(s)

    tmp.append([occupationScore, "Occupat°"])
    scores = sorted(tmp)

    x = []
    y = []
    for i in range(len(scores)):
        y.append(scores[i][0])
        name = scores[i][1]
        if i%3 == 0:
            x.append(name)
        elif i%3 == 1: 
            x.append('\n' +name)
        else:
            x.append( '\n' + '\n' + name)


    plt.bar(x, y)
    plt.xticks(rotation=90)
    plt.rc('xtick', labelsize=15)
    plt.title("Feature importance")

    plt.savefig(plotFolder + "randomForestFeaturesScores.pdf")

    cumsum = 0
    for score in scores:
        cumsum += score[0]
        score[0] = cumsum
    
    crossVal = []
    x = []
    times = []
    testY = np.ravel(testY)
    for i in range(-1, len(scores)):
        if i >= 0:
            if scores[i][0] > 0.5:
                break
            if scores[i][1] == "occupation":
                break
            testX = testX.drop(scores[i][1], axis = 1)
        start = time.time()
        cScore = -1 * np.mean(cross_val_score(model, testX, testY, n_jobs=-1, cv=10, scoring='neg_mean_squared_error'))
        end = time.time()
        times.append(datetime.timedelta(seconds=end-start).total_seconds())

        crossVal.append(cScore)
        x.append(i+1)


    plt.figure()
    plt.plot(x, crossVal)
    plt.title("Scores with regards to number of dropped features")
    plt.savefig(plotFolder + "randomForestFeaturesDroppedScore.pdf")

    plt.figure()
    plt.plot(x, times)
    plt.title("Time with regards to number of dropped features")
    plt.savefig(plotFolder + "randomForestFeaturesDroppedTime.pdf")
