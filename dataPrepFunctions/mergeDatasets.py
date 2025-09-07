#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# this function adds the movie titles to the rating
# structure and removes useless data
# It does not modify eisting dataframes but returns
# a new panda dataframe
def MergeMoviesAndRatings(movies, ratings):

    # merge dataframes and store in new dataframe
    mergedData = ratings
    mergedData = mergedData.merge(movies, on="movieId")

    # remove timestamps
    mergedData = mergedData.drop(columns=["timestamp"])

    # remove genres
    mergedData = mergedData.drop(columns=["genres"])

    return mergedData



