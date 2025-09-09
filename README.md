# Movie Recommendation System
This project uses a neural network recommendation system that predicts movie ratings for users. The datast that will be used is the MovieLens small dataset. The goal is to get an as low as possible **root mean squared error (RMSE)** between the predicted ratings and the actual ratings.

### necessary packages to run this code:
- scikit-learn             1.7.1
- torch                    2.8.0
- pandas                   2.3.2


## Workflow
### 1. Data Preprocessing
Before training the model some data exploration and preprocessing is done:
- Check if there are any duplicate movies or reviews.
- Plot Histograms for given movies.
- Rank movies based on popularity (most to least reviewed).
- Rank movies based on rating (highest to lowest average rating).
- Merge different CVS data files from the MovieLens dataset.
- Filter out movies with too few ratings.

### 2. The Model
The architecture of the model has the following properties:
- A Single embedding vector is used to represent "userId" and "movieId".
- A fully connected neural network is used to find more complex interactions between users and movies than a convolutional neural network could.
- The output layer has one output neuron that will predict a single rating value.
Different value for embedding vector size and different neural network configurations will be used tofigure out the optimal architecture to predict movie ratings.

### Training
