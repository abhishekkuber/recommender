import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.utils import shuffle

# Getting the rating dataset , consistes of triples of (user, movie, rating)
df = pd.read_csv('rating.csv')
df.head()


'''
Not sure if the users and movies are indexed from 0 to N-1

The categorical type is a process of factorization, each unique value or category is given a incremented integer value
starting from zero.

The data structure consists of a categories array and an integer array of codes which point to the real value in
the categories array.
'''

df.userId = pd.Categorical(df.userId)
df['newUserId'] = df.userId.cat.codes

df.movieId = pd.Categorical(df.movieId)
df['newMovieId'] = df.movieId.cat.codes

# Getting the required values
user_ids = df['newUserId'].values
movie_ids = df['newMovieId'].values
ratings = df['rating'].values

# Getting number of movies and users
total_users = len(set(user_ids))
total_movies = len(set(movie_ids))

# Hyperparameter (Embedding dimension)
EMBEDDING_DIM = 50


'''
The model would have two inputs, one for the user and one for the movie. Both of them will be indices, so the shape is (1, )

Embedding layer turns integers into vectors of a fixed size
Input shape: (batch_size, input_length).
Output shape: (batch_size, input_length, output_dim). # Here, input length is 1

The embeddings are in a 3D array, so we flatten them to a size of (total_samples, EMBEDDING_DIM)
For each row, the feature vector will be a vector of size 2 * EMBEDDING_DIM

'''

users = Input(shape=(1,))      # user input
movies = Input(shape=(1,))      # movie input 

user_embeddings = Embedding(total_users, EMBEDDING_DIM)(users) 
movie_embeddings = Embedding(total_movies, EMBEDDING_DIM)(movies) 

# Flattening
user_embeddings = Flatten()(user_embeddings) 
movie_embeddings = Flatten()(movie_embeddings) 

# Concatenate into a feature vector
x = Concatenate()([user_embeddings, movie_embeddings])     

x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(1)(x)

model = Model(inputs=[users, movies], outputs=x)
optimizer = SGD(lr=0.001, momentum=0.9)
model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])


# Splitting the data
user_ids, movie_ids, ratings = shuffle(user_ids, movie_ids, ratings)

split = int(0.8 * len(ratings))
train_user_ids = user_ids[:split]
train_movie_ids = movie_ids[:split]
train_ratings = ratings[:split]

test_user_ids = user_ids[split:]
test_movie_ids = movie_ids[split:]
test_ratings = ratings[split:]

history = model.fit(x=[train_user_ids, train_movie_ids], y=train_ratings, epochs=10, batch_size=2048, validation_data=([test_user_ids, test_movie_ids], test_ratings))