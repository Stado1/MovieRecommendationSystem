#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def graphScoresForOneMovie(movieNum, ratings, movieName):

    #get all the ratings for the movie and store in numpy array
    movieId = movieNum
    userRating = ratings[ratings['movieId'] == movieId]
    userRating = userRating.iloc[:, 2].to_numpy()

    #Make a histogram and store it as "scoreHistogram.png"
    plt.hist(userRating, bins=100)
    plt.xticks(np.arange(0, 5.5, 0.5))
    plt.xlabel("Ratings")
    plt.ylabel("Frequency")
    plt.title("Rating out of 5, for the movie: \n" + movieName)
    plt.savefig("scoreHistogram.png")



