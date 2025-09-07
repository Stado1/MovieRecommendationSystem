#!/usr/bin/env python3

#import libraries
import pandas as pd
import sklearn
import torch
import numpy as np

from scipy.sparse import csr_matrix

from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#import functions from other python files
from dataPrepFunctions import graphScoresForOneMovie
from dataPrepFunctions import rateMoviePolpularityRating, rateMoviePolpularityViews
from dataPrepFunctions import MergeMoviesAndRatings
from dataPrepFunctions import removeSparseData
from knnRecommendation import testHello

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

print(";;;;;;;;;;")

movieId = 2

# Make a histogram of the ratings for the movie with movieId
#graphScoresForOneMovie(movieId, ratings, movies.iloc[movieId-1, 1])

# Make a panda structure ordering all the movies from most reviewed to least reviewed
print(rateMoviePolpularityViews(ratings, movies))

# Make a panda structure ordering all the movies from best average review to worst average review
#print(rateMoviePolpularityViews(ratings, movies))

# numOfUsers = ratings.iloc[-1, 0]
# print("num of users = ", numOfUsers)
#
# userRating = ratings[ratings['movieId'] == movieId]
# numOfRatings = userRating.shape[0]
# print("num of ratings for movie = ", numOfRatings)
# # print(userRating.iloc[:, 2].to_numpy())
#
# averageRating = userRating['rating'].mean()
# print("movieId = ", movieId)
# print("average rating = ", averageRating)
# # print("again = ", np.mean(userRating.iloc[:, 2].to_numpy()))




dataSet = MergeMoviesAndRatings(movies, ratings)
dataSet = removeSparseData(dataSet)



