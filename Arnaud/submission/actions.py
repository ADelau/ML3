# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import dataset
from sklearn.metrics import mean_squared_error
import random
import math
from itertools import product

def cross_val_score(estimator, X, y, cv = 5):
	
	score = 0

	foldSize = len(X)//cv

	for fold in range(cv):

		trainX = pd.concat([X.iloc[:fold * foldSize], X.iloc[(fold + 1) * foldSize:]], axis = 0, sort = False, ignore_index = True)
		trainY = pd.concat([y.iloc[:fold * foldSize], y.iloc[(fold + 1) * foldSize:]], axis = 0, sort = False, ignore_index = True)
		
		testX = X.iloc[fold * foldSize : (fold + 1) * foldSize]
		testY = y.iloc[fold * foldSize : (fold + 1) * foldSize]

		estimator.fit(trainX, trainY)
		predictY = estimator.predict(testX)

		score += mean_squared_error(testY, predictY)

	score /= cv

	return score


class RandomizedSearchCV():

	def __init__(self, estimator, param_distribution, n_iter = 10, cv = 3):

		self.estimator = estimator
		self.param_distribution = param_distribution
		self.n_iter = n_iter
		self.cv = cv
		self.best_param_ = None
		self.best_score_ = math.inf


	def compute_param(self, X, y, filename):

		self.best_param_ = None
		self.best_score_ = math.inf

		file = open(filename, "w")

		for iteration in range(self.n_iter):

			print("iteration {}".format(iteration))

			testedParam = {name : random.choice(distribution) for name, distribution in self.param_distribution.items()}
			self.estimator.set_params(**testedParam)

			score = cross_val_score(self.estimator, X, y, self.cv)

			for key, value in testedParam.items():
				file.write("{} = {}\n".format(key, value))

			file.write("score = {}\n \n".format(score))

			if score < self.best_score_:
				self.best_score_ = score
				self.best_param_ = testedParam

		for key, value in self.best_param_.items():
				file.write("best {} = {}\n".format(key, value))

		file.write("best score = {}\n \n".format(self.best_score_))

		file.close()

def _my_product(inp):
    return (dict(zip(inp.keys(), values)) for values in product(*inp.values()))


class GridSearchCV():

	def __init__(self, estimator, param_grid, cv = 3):

		self.estimator = estimator
		self.param_grid = param_grid
		self.cv = cv
		self.best_param_ = None
		self.best_score_ = math.inf


	def fit(self, X, y, filename):

		self.best_param_ = None
		self.best_score_ = math.inf

		file = open(filename, "w")

		iteration = 0

		for testedParam in _my_product(param_grid):

			print("iteration {}".format(iteration))
			iteration += 1

			self.estimator.set_params(**testedParam)

			score = cross_val_score(self.estimator, X, y, self.cv)

			for key, value in testedParam.items():
				file.write("{} = {}\n".format(key, value))

			file.write("score = {}\n".format(score))

			if score < self.best_score_:
				self.best_score_ = score
				self.best_param_ = testedParam

		for key, value in self.best_param_.items():
				file.write("best {} = {}\n".format(key, value))

		file.write("best score = {}\n".format(self.best_score_))

		file.close()

	def predict(self, x):
		return self.estimator.predict(x)

def make_submission(estimator, fileName):
	trainX, trainY, predictX = dataset.make_dataset()

	estimator.fit(trainX, trainY)
	predictY = estimator.predict(predictX)

	userMovieID = dataset.get_test_user_movie_pair()
	userMovieID = userMovieID.values

	with open(fileName, 'w') as handle:
		handle.write('"USER_ID_MOVIE_ID","PREDICTED_RATING"\n')
		for (userID, movieID), prediction in zip(userMovieID, predictY):
		    if np.isnan(prediction):
		        raise ValueError('The prediction cannot be NaN')
		    line = '{:d}_{:d},{}\n'.format(userID, movieID, prediction)
		    handle.write(line)