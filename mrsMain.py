#!/usr/bin/env python3

#import libraries
import pandas as pd
import sklearn
import torch
import numpy as np

#import functions from other python files
from dataPrepFunctions import graphScoresForOneMovie
from dataPrepFunctions import rateMoviePolpularityRating, rateMoviePolpularityViews


movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')
tags = pd.read_csv('ml-latest-small/tags.csv')

print("amount of movies:")
print(movies.shape[0])
print("amount of ratings:")
print(ratings.shape[0])
print("amount of tags:")
print(tags.shape[0])

print(";;;;;;;;;;")

movieId = 325

# Make a histogram of the ratings for the movie with movieId
#graphScoresForOneMovie(movieId, ratings, movies.iloc[movieId-1, 1])

numOfUsers = ratings.iloc[-1, 0]
print("num of users = ", numOfUsers)

userRating = ratings[ratings['movieId'] == movieId]
numOfRatings = userRating.shape[0]
print("num of ratings for movie = ", numOfRatings)
# print(userRating.iloc[:, 2].to_numpy())

averageRating = userRating['rating'].mean()
print("average rating = ", averageRating)
# print("again = ", np.mean(userRating.iloc[:, 2].to_numpy()))



