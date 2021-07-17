# Movie Recommender System - for MovieLens 20M dataset 

Recommender systems are used almost everywhere. Most of them are similar, using cosine similarities to give recommendations. Here, an alternate approach is used.

The dataset consists of triples of *(user_id, movie_id, rating)*. 

Instead of the conventional neural networks having a single input, there are **two inputs** here, the *user_id* and *movie_id*. The inputs are converted to embeddings and then combined together. Since the inputs are now vectors of fixed dimension of (2 * EMBEDDING_DIM), *(check the code for clarity)*, we can consider this as a regression problem. We pass these through an Artificial Neural Network, and the rating is calculated.

Link to the dataset:
[MovieLens 20M dataset](https://www.kaggle.com/grouplens/movielens-20m-dataset)
