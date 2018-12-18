# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import datetime
from contextlib import contextmanager
import pandas as pd
import numpy as np
import utils
from sklearn.ensemble import RandomForestRegressor
import math
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

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
    """
    Perform One-hot encoding on the data

    Parameters
    ----------
    data: the dataFrame to encode

    Return
    ------
    data: the encoded dataFrame
    """
    data = pd.get_dummies(data, sparse = True)
    
    return data

def clean(data):
    """
    Clean the data with missing values

    Parameters
    ----------
    data: the dataFrame to clean

    Return
    ------
    data: the cleaned dataFrame
    """
    nanProportionToDrop = 1/len(data.columns)
    data = data.dropna(axis = 1, thresh = nanProportionToDrop*len(data.index))
    data = data.dropna(axis = 0, how = "any")

    return data

def make_dataset(toSplit = False, splitRatio = 0.2):
    """
    Make the dataset

    Parameters
    ----------
    toSplit: boolean indicating if the train dataset has to be splitted between train and test split(if True)
            or if it should output the full train set as train set and the test set as test set(is False)
    splitRatio: the ratio of the dataset to be used as test set if the dataset is splitted

    Return
    ------
    trainX: the train dataset inputs
    trainY: the train dataset outputs
    testX: if toSplit = True, the test dataset inputs
    testY: if toSplit = True, the test dataset outputs
    testDataset : if toSplit = False, the test dataset inputs
    userMeanRating: the mean rating per user based on the train set
    movieMeanRating: the mean rating per movie based on the train set
    userStdRating: the mean standard deviation of the user based on the train set
    movieStdRating: the mean standard deviation of the movie based on the train set
    userDict: a dictionnary mapping user id to index in the user mean and std ratings
    movieDict: a dictionnary mapping movie id to index in the movie mean and std ratings
    """
    
    # Load datasets
    userMoviePair = load_from_csv("data/data_train.csv")
    userFeatures = load_from_csv("data/data_user_3.csv")
    movieFeatures = load_from_csv("data/data_movie.csv")

    # Drop unused features
    movieFeatures = movieFeatures.drop("IMDb_URL", axis = 1)
    movieFeatures = movieFeatures.drop("video_release_date", axis = 1)
    userFeatures = userFeatures.drop("zip_code", axis = 1)
    
    # Convert date into an integer to induce an order between the movies in our model
    movieFeatures["release_date"] = [utils.dateConverter(date) for date in movieFeatures["release_date"]]
    
    # Set the type of gender and occupation features as category so that they can be encoded 
    userFeatures["gender"] = userFeatures["gender"].astype("category")
    userFeatures["occupation"] = userFeatures["occupation"].astype("category")

    # Compute the polarity and subjectivity of the movie names
    polarity, subjectivity = utils.stringAnalyzer(movieFeatures["movie_title"])
    movieFeatures = pd.concat(objs = [movieFeatures, pd.Series(data = polarity, name = "polarity")], axis = 1, sort = False)
    movieFeatures = pd.concat(objs = [movieFeatures, pd.Series(data = subjectivity, name = "subjectivy")], axis = 1, sort = False)
    movieFeatures = movieFeatures.drop("movie_title", axis = 1)
    
    trainDataset = userMoviePair

    # Load train outputs 
    y = load_from_csv("data/output_train.csv")
    trainDataset = pd.concat([trainDataset, y], axis = 1, sort = False)
    
    # Clean dataset
    trainDataset = clean(trainDataset)
	
    #Split in train and test set
    if toSplit:
        trainLength = math.floor((1 - splitRatio) * len(trainDataset))
        trainDataset = shuffle(trainDataset)
        testDataset = trainDataset[trainLength:]
        trainDataset = trainDataset[:trainLength]
    
    #Load test set
    else:
        testDataset = load_from_csv("data/data_test.csv")

    # Compute aggregated values on the ratings for users and movies 
    userMeanRating = trainDataset.groupby("user_id", as_index = False).rating.agg("mean")
    userMeanRating.rename(index = str, columns = {"rating": "userMeanRating"}, inplace = True)
    movieMeanRating = trainDataset.groupby("movie_id", as_index = False).rating.agg("mean")
    movieMeanRating.rename(index = str, columns = {"rating": "movieMeanRating"}, inplace = True)

    userStdRating = trainDataset.groupby("user_id", as_index = False).rating.agg("var")
    userStdRating.rename(index = str, columns = {"rating": "userStdRating"}, inplace = True)
    userStdRating["userStdRating"] = userStdRating["userStdRating"].apply(lambda x : np.sqrt(x))
    movieStdRating = trainDataset.groupby("movie_id", as_index = False).rating.agg("var")
    movieStdRating.rename(index = str, columns = {"rating": "movieStdRating"}, inplace = True)
    movieStdRating["movieStdRating"] = movieStdRating["movieStdRating"].apply(lambda x : np.sqrt(x))
    
    # Establish the mapping between user and movie id and their index in the aggregated datasets
    userDict = {}
    movieDict = {}
    for index, sample in userMeanRating.iterrows():
        userId = sample["user_id"]
        userDict[userId]=int(index)
    
    for index, sample in movieMeanRating.iterrows():
        movieId = sample["movie_id"]
        movieDict[movieId]=int(index)
    
    # Normalize the ratings
    normalizedRating = []
    for index, sample in trainDataset.iterrows():
        userId = sample["user_id"]
        movieId = sample["movie_id"]
        rating = sample["rating"]
        userIndex = userDict[userId]
        movieIndex = movieDict[movieId]
        userMean = userMeanRating["userMeanRating"][userIndex]
        movieMean = movieMeanRating["movieMeanRating"][movieIndex]
        userStd = userStdRating["userStdRating"][userIndex]
        movieStd = movieStdRating["movieStdRating"][movieIndex]
        mean = ((userMean + movieMean)/2)
        std = ((userStd + movieStd)/2)
        
        # No std if not enough samples 
        if std == 0 or np.isnan(std):
            normalizedRating.append(rating-mean)
        else:
            normalizedRating.append((rating-mean)/std)
    
    normalizedRating = pd.Series(normalizedRating, name="rating")
        
    # Drop old ratings
    trainDataset = trainDataset.drop("rating", axis = 1)
    
    # Merge the features
    trainDataset = pd.merge(trainDataset, userFeatures, on = "user_id")
    trainDataset = pd.merge(trainDataset, movieFeatures, on = "movie_id")
    
    # Add the normalized ratings 
    trainDataset = pd.concat([trainDataset, normalizedRating], axis = 1, sort = False)

    # Merge the features 
    testDataset = pd.merge(testDataset, userFeatures, on = "user_id")
    testDataset = pd.merge(testDataset, movieFeatures, on = "movie_id")
    
    # Drop the ids that are no longer used 
    trainDataset = trainDataset.drop("user_id", axis = 1)
    trainDataset = trainDataset.drop("movie_id", axis = 1)

    # Clean the train dataset
    trainDataset = clean(trainDataset)
    
    # Fill the missing values in the test dataset 
    testDataset.fillna(method = "ffill", inplace = True)

    # Split the train dataset into inputs and outputs 
    trainY = trainDataset["rating"].to_frame()
    trainX = trainDataset.drop("rating", axis = 1)
	
    # Split the test dataset into inputs and outputs if toSplit = True  
    if toSplit:
        testY = testDataset["rating"].to_frame()
        testX = testDataset.drop("rating", axis = 1)
        return trainX, trainY, testX, testY, userMeanRating, movieMeanRating, userStdRating, movieStdRating, userDict, movieDict
  
    return trainX, trainY, testDataset, userMeanRating, movieMeanRating, userStdRating, movieStdRating, userDict, movieDict

def make_prediction(trainX, trainY, testX, userMeanRating, movieMeanRating, userStdRating, movieStdRating, userDict, movieDict):
    """
    Predict the values for the inputs in testX

    Parameters
    ----------
    trainX: the train dataset inputs
    trainY: the train dataset outputs
    testX: the inputs on which to make predictions
    userMeanRating: the mean rating per user based on the train set
    movieMeanRating: the mean rating per movie based on the train set
    userStdRating: the mean standard deviation of the user based on the train set
    movieStdRating: the mean standard deviation of the movie based on the train set
    userDict: a dictionnary mapping user id to index in the user mean and std ratings
    movieDict: a dictionnary mapping movie id to index in the movie mean and std ratings

    Return
    ------
    denormalizedRating: the predicted ratings
    """
    
    # Drop the ids that are not used 
    predictX = testX.drop("user_id", axis = 1)
    predictX = predictX.drop("movie_id", axis = 1)
    
    # Perform One-hot encoding to encode the categorical features 
    print("encoding ...")
    trainLength = len(trainX)
	
    toEncode = pd.concat(objs = [trainX, predictX], axis = 0, sort = False, ignore_index = True)
    encoded = encode(toEncode)
	
    trainX = encoded[:trainLength]
    predictX = encoded[trainLength:]
    
    # Fit the classifier
    print("fitting ...")
    classifier = RandomForestRegressor(max_depth = 30, n_estimators = 400, max_features = 'auto', bootstrap = True, min_samples_split = 10) 
    classifier.fit(trainX, trainY)
    
    # Predict normalized ratings 
    print("predicting ...")
    predictY = classifier.predict(predictX)
    predictY = pd.Series(predictY, name="rating")
    predictionDataset = pd.concat([testX, predictY], axis = 1, sort = False)
    
    # Denormalize the ratings
    denormalizedRating = []
    for index, sample in predictionDataset.iterrows():
        userId = sample['user_id']
        movieId = sample['movie_id']
        rating = sample['rating']
        
        # Get user mean and std 
        if userId in userDict:
            userIndex = userDict[userId]
            userMean = userMeanRating["userMeanRating"][userIndex]
            userStd = userStdRating["userStdRating"][userIndex]
        else:
            userMean = np.nan

        # Get movie mean and std 
        if movieId in movieDict:
            movieIndex = movieDict[movieId]
            movieMean = movieMeanRating["movieMeanRating"][movieIndex]
            movieStd = movieStdRating["movieStdRating"][movieIndex]
        else: 
            movieMean = np.nan
        
        # Compute global mean and std 
        if np.isnan(userMean):
            mean = movieMean
        elif np.isnan(movieMean):
            mean = userMean
        elif np.isnan(userMean) and np.isnan(movieMean):
            userMean = userMeanRating["userMeanRating"].mean()
            movieMean = movieMeanRating["movieMeanRating"].mean()
            mean = ((userMean + movieMean)/2)
        else:
            mean = ((userMean + movieMean)/2)
        
        std = ((userStd + movieStd)/2)
        
        # Denormalize the ratings 
        if std == 0 or np.isnan(std):
            denormalizedRating.append(rating + mean) 
        else:
            denormalizedRating.append(rating * std + mean)

    return denormalizedRating

def submit():
    """
    make a file to submit on kaggle platform
    """
    print("building dataset...")
    trainX, trainY, testX, userMeanRating, movieMeanRating, userStdRating, movieStdRating, userDict, movieDict = make_dataset(False, 0.2)
    
    predictY = make_prediction(trainX, trainY, testX, userMeanRating, movieMeanRating, userStdRating, movieStdRating, userDict, movieDict)

    # Make the submission file 
    user_movie_pair = []
    for index, item in testX.iterrows():
        user = item['user_id']
        movie = item['movie_id']
        user_movie_pair.append([int(user), int(movie)])
        
    fname = make_submission(predictY, user_movie_pair , 'RandomForestNormalizedRating')
    print('Submission file "{}" successfully written'.format(fname))

def compute_accuracy():
    """
    Compute the accuracy of the model

    Parameters
    ----------
    neighbors: the number of neighbors to use for the estimator

    Return
    ------
    error: the mean squared error
    """
    
    trainX, trainY, testX, testY, userMeanRating, movieMeanRating, userStdRating, movieStdRating, userDict, movieDict = make_dataset(True, 0.2)
    
    predictY = make_prediction(trainX, trainY, testX, userMeanRating, movieMeanRating, userStdRating, movieStdRating, userDict, movieDict)
    error = mean_squared_error(testY, predictY) 
    return error
   
if __name__ == '__main__':
    compute_accuracy()
    

   


