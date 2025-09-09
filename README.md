# Movie Recommendation System
This project uses a neural network recommendation system that predicts movie ratings for users. This prject uses the small MovieLens dataset. The goal is to get an as low as possible **root mean squared error (RMSE)** between the predicted ratings and the actual ratings.

### Necessary packages to run this code:
- scikit-learn             1.7.1
- torch                    2.8.0
- pandas                   2.3.2


## Workflow
### Data Preprocessing
Before training the model some data exploration and preprocessing is done:
- Check if there are any duplicate movies or reviews.
- Plot Histograms of ratings for given movies.
- Rank movies based on popularity (most to least reviewed).
- Rank movies based on rating (highest to lowest average rating).
- Merge different CSV data files from the MovieLens dataset into a single Pandas dataframe.
- Filter out movies with less than 5 ratings.

### Model
The architecture of the model has the following properties:
- Two embedding vectors are used to represent "userId" and "movieId", which are joined into a single input vector.
- A fully connected neural network is used to find more complex interactions between users and movies than a simple linear model could.
- The output layer has one output neuron that will predict a single rating value.

Different values for embedding vector size and different neural network configurations will be used to figure out the optimal architecture to predict movie ratings.

### Training and Evaluation
- The Adam optimizer was used with a learning rate of 0.001.
- For the loss function mean squared error was used but for validation RMSE was used.
- The batch size used was 256.
- Early stopping was used to prevent overfitting.


## Results
The Results for neural networks with an embedding vector size of 32 are:

| Hidden Layer Structure (nodes per layer) | RMSE   |
|------------------------------------------|--------|
| 32                                       | 0.8889 |
| 64                                       | **0.8818** |
| 128                                      | 0.8854 |
| 32-16                                    | 0.8941 |
| 64-32                                    | 0.8967 |
| 128-64                                   | 0.8936 |
| 64-32-16                                 | 0.8911 |
| 128-64-32                                | 0.8831 |


The Results for neural networks with an embedding vector size of 64 are:

| Hidden Layer Structure (nodes per layer) | RMSE   |
|------------------------------------------|--------|
| 32                                       | 0.8949 |
| 64                                       | 0.8959 |
| 128                                      | 0.9010 |
| 32-16                                    | 0.8963 |
| 64-32                                    | 0.8974 |
| 128-64                                   | 0.9075 |
| 64-32-16                                 | 0.8975 |
| 128-64-32                                | 0.9164 |

## Discussion and Future Research
What can be seen in the tables is that there is not a lot of variation between the different neural networks and the embedding vector size. This is probably because a simple neural network is good enough to find all the connections between users and their tastes in movies. A more complex neural network would just find patterns in noise, which is why a smaller embedding vector with a simpler neural network performs slightly better.

A possible way to improve the system is to feed more data into the neural network, information like movie genres and movie tags can be useful for making predictions.

