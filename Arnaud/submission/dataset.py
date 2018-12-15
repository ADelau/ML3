# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from textblob.en.sentiments import PatternAnalyzer

def stringAnalyzer(string):
    polarity = []
    subjectivity = []

    analyzer = PatternAnalyzer()

    for sentence in string:
        analysis = analyzer.analyze(sentence)
        polarity.append(analysis[0])
        subjectivity.append(analysis[1])

    return polarity, subjectivity

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

def fill(data):
	data.fillna(method = "ffill", inplace = True)
	data.fillna(method = "bfill", inplace = True)

	return data

def clean(data):
	return data.dropna(axis = 0, how = "any")

def make_base_dataset(userMoviePairPath, user_path, includeY):
	userMoviePair = load_from_csv(userMoviePairPath)
	userFeatures = load_from_csv(user_path)
	movieFeatures = load_from_csv("data/data_movie.csv")

	userFeatures = userFeatures.drop("zip_code", axis = 1)

	userFeatures["gender"] = userFeatures["gender"].astype("category")
	userFeatures["occupation"] = userFeatures["occupation"].astype("category")

	movieFeatures = movieFeatures.drop("IMDb_URL", axis = 1)
	movieFeatures = movieFeatures.drop("video_release_date", axis = 1)

	movieFeatures["release_date"] = movieFeatures["release_date"].astype("category")

	polarity, subjectivity = stringAnalyzer(movieFeatures["movie_title"])
	movieFeatures = pd.concat(objs = [movieFeatures, pd.Series(data = polarity, name = "polarity")], axis = 1, sort = False)
	movieFeatures = pd.concat(objs = [movieFeatures, pd.Series(data = subjectivity, name = "subjectivy")], axis = 1, sort = False)
	movieFeatures = movieFeatures.drop("movie_title", axis = 1)

	dataset = userMoviePair

	if(includeY):
		y = load_from_csv("data/output_train.csv")
		dataset = pd.concat([dataset, y], axis = 1, sort = False)

	dataset = pd.merge(dataset, userFeatures, on = "user_id")
	dataset = pd.merge(dataset, movieFeatures, on = "movie_id")

	return dataset

def make_train_dataset():
	dataset = clean(make_base_dataset("data/data_train.csv", "data/data_user_3.csv", True))

	y = dataset["rating"].to_frame()
	x = dataset.drop("rating", axis = 1)

	return x,y

def make_test_dataset():
	return fill(make_base_dataset("data/data_test.csv", "data/data_user_3.csv", True))

def make_dataset():
	trainX, trainY = make_train_dataset()
	testX = make_test_dataset()

	return trainX, trainY, testX

def get_test_user_movie_pair():
	return load_from_csv("data/data_test.csv")