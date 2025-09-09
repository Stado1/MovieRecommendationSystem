#!/usr/bin/env python3

import pandas as pd
import sklearn
import torch
import numpy as np
import math

#import functions from other python files
from dataPrepFunctions import graphScoresForOneMovie
from dataPrepFunctions import rateMoviePolpularityRating, rateMoviePolpularityViews
from dataPrepFunctions import MergeMoviesAndRatings
from dataPrepFunctions import removeSparseData
from neuralNetworkRecommendation import recommendNN

movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')
#tags = pd.read_csv('ml-latest-small/tags.csv')

# print general info of data
print("amount of movies:")
print(movies.shape[0])
print("amount of ratings:")
print(ratings.shape[0])
#print("amount of tags:")
#print(tags.shape[0])

# check if data has duplicates
duplicatesRatings = ratings.duplicated(subset=["userId", "movieId"]).sum()
print("amount of duplicate ratings:", duplicatesRatings)
duplicateMovies = movies.duplicated(subset=["movieId"]).sum()
print("amount of duplicate movies:", duplicateMovies)


movieId = 2
# Make a histogram of the ratings for the movie with movieId
graphScoresForOneMovie(movieId, ratings, movies.iloc[movieId-1, 1])
print("Saved image of histogram of the ratings for the movie with movieId = ", movieId)

# Make a panda dataframe ordering all the movies from most reviewed to least reviewed
print("----------------------")
print("Panda dataframe ordering all the movies from most reviewed to least reviewed:   ")
print(rateMoviePolpularityViews(ratings, movies))

# Make a panda dataframe ordering all the movies from best average review to worst average review
print("----------------------")
print("Panda dataframe ordering all the movies from best average review to worst average review: ")
print(rateMoviePolpularityViews(ratings, movies))

# merge movies and the ratings dataframes
dataSet = MergeMoviesAndRatings(movies, ratings)

# remove movies from dataset that have less than 5 raings
print("----------------------")
print("remove movies from dataset that have less than 5 raings: ")
dataSet = removeSparseData(dataSet)

# train the neural network
print("----------------------")
print("Start training neural network:")
recommendNN(dataSet)






