#!/usr/bin/env python3

import pandas as pd
import sklearn
import torch



test = pd.read_csv('~/irdeto_env/code/MovieRecommendationSystem/ml-latest-small/movies.csv')

print(type(test))

print(test.tail())



