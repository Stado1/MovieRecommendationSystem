#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# returns a panda dataframe with movie names and average ratings of each movie
# ordered from best rated to worst rated
def rateMoviePolpularityRating(ratings, movies):

    # find the average rating of each movie
    ratingGroups = ratings.groupby("movieId")
    ratingAverages = ratingGroups["rating"].mean()
    ratingAverages = ratingAverages.reset_index(name="ratingAverage")

    # count the amount of reviews of each movie, stores it in a panda structure
    ratingCounts = ratings.groupby("movieId").size()
    ratingCounts = ratingCounts.reset_index(name="amountOfRatings")

    #merge
    ratingAverages = ratingAverages.merge(ratingCounts, on="movieId")

    # add movie names and other useless info
    movieRatingAverages = movies.merge(ratingAverages, on="movieId", how="left")
    # remove useless info
    movieRatingAverages = movieRatingAverages.drop(columns=["movieId"])
    movieRatingAverages = movieRatingAverages.drop(columns=["genres"])

    # sort based on average rating
    movieRatingAverages = movieRatingAverages.sort_values(by="ratingAverage", ascending=False)


    return movieRatingAverages


# returns a panda dataframe with movie names and average ratings of each movie
# ordered from most amount of ratings to least amount of ratings
def rateMoviePolpularityViews(ratings, movies):

    # count the amount of reviews of each movie, stores it in a panda structure
    ratingCounts = ratings.groupby("movieId").size()
    ratingCounts = ratingCounts.reset_index(name="amountOfRatings")

    # find the average rating of each movie
    ratingGroups = ratings.groupby("movieId")
    ratingAverages = ratingGroups["rating"].mean()
    ratingAverages = ratingAverages.reset_index(name="ratingAverage")

    #merge
    ratingCounts = ratingCounts.merge(ratingAverages, on="movieId")

    # add movie names and other useless info
    movieRatingCounts = movies.merge(ratingCounts, on="movieId", how="left")
    # remove useless info
    movieRatingCounts = movieRatingCounts.drop(columns=["movieId"])
    movieRatingCounts = movieRatingCounts.drop(columns=["genres"])

     # sort based on amount of reviews
    movieRatingCounts = movieRatingCounts.sort_values(by="amountOfRatings", ascending=False)


    return movieRatingCounts












