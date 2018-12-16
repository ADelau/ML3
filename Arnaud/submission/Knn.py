# ! /usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.neighbors import KNeighborsRegressor
from NormalizedEstimator import NormalizedEstimator
import dataset
import actions

def compute_accuracy():
	X, y, _ = dataset.make_dataset()

	estimator = KNeighborsRegressor(n_neighbors = 200)
	normalizedEstimator = NormalizedEstimator(estimator)

	score = actions.cross_val_score(normalizedEstimator, X, y, cv = 2)
	print("score = {}".format(score))

if __name__ == "__main__":
	compute_accuracy()