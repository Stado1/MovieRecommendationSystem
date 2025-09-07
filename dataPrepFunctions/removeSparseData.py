#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# this function will remove entries with less than XXX
# amount of reviews to improve the recommendation system
def removeSparseData(data):

    # amount of reviews needed to not be removed
    minAmountOfReviews = 5

    movieCount = data['movieId'].value_counts()
    moviesToKeep = movieCount[movieCount >= minAmountOfReviews].index
    filteredData = data[data['movieId'].isin(moviesToKeep)]

    return filteredData



