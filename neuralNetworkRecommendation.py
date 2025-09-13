#!/usr/bin/env python3

import pandas as pd
import sklearn
import torch
import numpy as np
import math

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# The neural network that is used
class RatingPredictionNN(nn.Module):
    def __init__(self, userAmount, movieAmount, embeddingVectorSize):
        super(RatingPredictionNN, self).__init__()
        self.userEmbeding = nn.Embedding(userAmount, embeddingVectorSize)
        self.movieEmbeding = nn.Embedding(movieAmount, embeddingVectorSize)

        self.fc1 = nn.Linear(embeddingVectorSize*2, 32)  # Input layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)     # Hidden layer
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(16, 1)     # Hidden layer
        # self.relu = nn.ReLU()
        # self.fc4 = nn.Linear(32, 1)     # Hidden layer


    def forward(self, x):
        userID = x[:, 0].long()
        movieID = x[:, 1].long()
        userVector = self.userEmbeding(userID)
        movieVector = self.movieEmbeding(movieID)
        x = torch.cat([userVector, movieVector], dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        # x = self.relu(x)
        # x = self.fc4(x)
        return x

# class used for early stopping
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        patience: number of epochs to wait for improvement before stopping
        min_delta: minimum change in the monitored metric to count as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# This function trains train the neural network defined above, using the dataset it is given
def recommendNN(dataSet):
    # convert the dataset as categorial indices for embedding vectors
    dataSet["userId"] = dataSet["userId"].astype("category").cat.codes
    dataSet["movieId"] = dataSet["movieId"].astype("category").cat.codes
    userAmount = dataSet["userId"].nunique()
    movieAmount = dataSet["movieId"].nunique()


    embeddingVectorSize = 32

    # Split the data up into a train eval and test split
    X = dataSet[["userId", "movieId"]].values
    y = dataSet["rating"].values
    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Convert the data into the right format
    X_train_tensor = torch.tensor(X_train, dtype=torch.long)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    datasetTrain = TensorDataset(X_train_tensor, y_train_tensor)
    loaderTrain = DataLoader(datasetTrain, batch_size=256, shuffle=True)

    X_eval_tensor = torch.tensor(X_eval, dtype=torch.long)
    y_eval_tensor = torch.tensor(y_eval, dtype=torch.float32).unsqueeze(1)
    datasetEval = TensorDataset(X_eval_tensor, y_eval_tensor)
    loaderEval = DataLoader(datasetEval, batch_size=256, shuffle=False)

    X_test_tensor = torch.tensor(X_test, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    datasetTest = TensorDataset(X_test_tensor, y_test_tensor)
    loaderTest = DataLoader(datasetTest, batch_size=256, shuffle=False)


    # Initialize model, loss, optimizer
    model = RatingPredictionNN(userAmount, movieAmount, embeddingVectorSize)
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    n_epochs = 1000
    early_stopping = EarlyStopping(patience=5, min_delta=0.0001)
    for epoch in range(n_epochs):
        model.train()
        for xb, yb in loaderTrain:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        # Evaluate on eval set
        model.eval()
        rmse_sum = 0
        n_samples = 0
        with torch.no_grad():
            for xb, yb in loaderEval:
                preds = model(xb)
                mse = ((preds - yb) ** 2).sum().item()
                rmse_sum += mse
                n_samples += yb.size(0)

        val_rmse = math.sqrt(rmse_sum / n_samples)
        print(f"Epoch {epoch+1}, Validation RMSE: {val_rmse:.4f}")

        # Check early stopping
        early_stopping(val_rmse)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Test the model

    model.eval()  # set model to evaluation mode
    rmse_sum = 0
    n_samples = 0

    with torch.no_grad():
        for xb, yb in loaderTest:
            preds = model(xb)
            # Compute squared error for the batch
            mse = ((preds - yb) ** 2).sum().item()
            rmse_sum += mse
            n_samples += yb.size(0)

    rmse = math.sqrt(rmse_sum / n_samples)
    print(f"RMSE on test set: {rmse:.4f}")


