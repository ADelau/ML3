# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin

class NormalizedEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def get_params(self, deep = True):
        return self.estimator.get_params(deep)

    def set_params(self, **params):
        self.estimator.set_params(**params)

    def fit(self, X, y):
        self.trainX = X.copy(deep = True)
        self.trainY = y.copy(deep = True)
        self.trainUserMovieDataset = pd.DataFrame.from_dict(data = {"userID" : self.trainX["user_id"], "movieID" : self.trainX["movie_id"], "rating" : self.trainY})
        self.trainX = self.trainX.drop("user_id", axis = 1)
        self.trainX = self.trainX.drop("movie_id", axis = 1)

    def _fit(self):

        self.globalMean = np.mean(self.trainY)
        self.globalSTD = np.std(self.trainY)

        self.userMeanRating = self.trainUserMovieDataset.groupby("userID", as_index = False).rating.agg("mean")
        self.userMeanRating.rename(index = str, columns = {"rating": "userMeanRating"}, inplace = True)

        self.movieMeanRating = self.trainUserMovieDataset.groupby("movieID", as_index = False).rating.agg("mean")
        self.movieMeanRating.rename(index = str, columns = {"rating": "movieMeanRating"}, inplace = True)

        self.userStdRating = self.trainUserMovieDataset.groupby("userID", as_index = False).rating.agg("var")
        self.userStdRating.rename(index = str, columns = {"rating": "userStdRating"}, inplace = True)
        self.userStdRating["userStdRating"] = self.userStdRating["userStdRating"].apply(lambda x : np.sqrt(x))

        self.movieStdRating = self.trainUserMovieDataset.groupby("movieID", as_index = False).rating.agg("var")
        self.movieStdRating.rename(index = str, columns = {"rating": "movieStdRating"}, inplace = True)
        self.movieStdRating["movieStdRating"] = self.movieStdRating["movieStdRating"].apply(lambda x : np.sqrt(x))

        self.userDict = {}
        self.movieDict = {}

        for index, sample in self.userMeanRating.iterrows():
            userID = sample["userID"]
            self.userDict[userID]=int(index)

        for index, sample in self.movieMeanRating.iterrows():
            movieID = sample["movieID"]
            self.movieDict[movieID]=int(index)

        normalizedRating = []

        for index, sample in self.userMovieDataset.iterrows():
            userID = sample["userID"]
            movieID = sample["movieID"]
            rating = sample["rating"]

            userIndex = self.userDict[userID]
            movieIndex = self.movieDict[movieID]

            userMean = self.userMeanRating["userMeanRating"][userIndex]
            movieMean = self.movieMeanRating["movieMeanRating"][movieIndex]
            userStd = self.userStdRating["userStdRating"][userIndex]
            movieStd = self.movieStdRating["movieStdRating"][movieIndex]

            mean = ((userMean + movieMean)/2)

            if userStd == 0 or np.isnan(userSTD):
                std = movieSTD

            elif movieSTD == 0 or np.isnan(movieSTD):
                std = userSTD

            else:
                std = ((userStd + movieStd)/2)

            if std == 0 or np.isnan(std):
                normalizedRating.append(rating-mean)
            else:
                normalizedRating.append((rating-mean)/std)

        self.estimator.fit(X, normalizedRating)

    def predict(X):
        userMovieDataset = pd.DataFrame(data = (X["user_id"], X["movie_id"], normalizedRating), columns = ["userID", "movieID", "normalizedRating"])
        X = X.drop("user_id", axis = 1)
        X = X.drop("movie_id", axis = 1)
        trainLength = len(self.trainX)

        toEncode = pd.concat(objs = [self.trainX, X], axis = 0, sort = False, ignore_index = True)
        encoded = pd.get_dummies(toEncode, sparse = True)

        self.trainX = encoded[:trainLength]
        X = encoded[trainLength:]

        self._fit()
        normalizedRating = self.estimator.predict(X)
        rating = []

        for index, sample in userMovieDataset.iterrows():
            userID = sample['userID']
            movieID = sample['movieID']

            if userID in userDict:
                userIndex = self.userDict[userID]
                userMean = self.userMeanRating["userMeanRating"][userIndex]
                userStd = self.userStdRating["userStdRating"][userIndex]
            else:
                userMean = np.nan
                userStd = np.nan

            if movieID in movieDict:
                movieIndex = self.movieDict[movieID]
                movieMean = self.movieMeanRating["movieMeanRating"][movieIndex]
                movieStd = self.movieStdRating["movieStdRating"][movieIndex]
            else:
                movieMean = np.nan
                movieStd = np.nan

            if np.isnan(userMean) and np.isnan(movieMean):
                mean = self.globalMean

            elif np.isnan(userMean):
                mean = movieMean

            elif np.isnan(movieMean):
                mean = userMean

            else:
                mean = (userMean + movieMean)/2


            if np.isnan(userSTD) and np.isnan(movieSTD):
                std = self.globalSTD

            elif np.isnan(userSTD):
                std = movieSTD

            elif np.isnan(movieSTD):
                std = userSTD

            else:
                std = (userSTD + movieSTD)/2

            rating.append(normalizedRating[index] * std + mean)

        return rating