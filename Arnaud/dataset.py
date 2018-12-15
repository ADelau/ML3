# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import utils
import pandas as pd

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

def clean(data):
	nanProportionToDrop = 1/len(data.columns)
	data = data.dropna(axis = 1, thresh = nanProportionToDrop*len(data.index))
	data = data.dropna(axis = 0, how = "any")

	return data

def make_base_dataset(userMoviePairPath, user_path, includeY):
	userMoviePair = load_from_csv(userMoviePairPath)
	userFeatures = clean(load_from_csv(user_path))
	movieFeatures = clean(load_from_csv("data/data_movie.csv"))	

	dataset = userMoviePair

	if(includeY):
		y = load_from_csv("data/output_train.csv")
		dataset = pd.concat([dataset, y], axis = 1, sort = False)

	dataset = pd.merge(dataset, userFeatures, on = "user_id")
	dataset = pd.merge(dataset, movieFeatures, on = "movie_id")
	dataset = clean(dataset)

	return dataset

def make_dataset_1():
	dataset = make_base_dataset("data/data_train.csv", "data/data_user_3.csv", True)

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

	

	return x,y