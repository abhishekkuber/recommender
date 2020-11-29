import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import heapq

training_set = pd.read_csv('training_set.csv')
training_set = training_set.drop(columns=['Timestamp'])

movies = pd.read_csv('movies.dat', sep='::', header=None)
movies = movies.drop(columns=[2])

training_set['Rating'].value_counts()
'''
4    261651
3    195899
5    169780
2     80679
1     42112
Name: Rating, dtype: int64
'''
list_cosines = []

names = ['5 stars', '4 stars', '3 stars', '2 stars', '1 star']
values = [169780, 261651, 195899, 80679, 42112]
plt.figure(figsize=(12, 8))
plt.bar(names, values)
plt.xlabel('Ratings')
plt.ylabel('Number of users')
plt.title('Users and their Ratings')
plt.show()


sparse_matrix_training = sparse.csr_matrix((training_set.Rating, (training_set.Movie, training_set.User)))
print(sparse_matrix_training)

'''
def calc_recc(int user):
    for i in allusers:
        calculate cosine/euclidean distance between users 
        take distance in array 
        top 50 users distance 
        take the top user id 
        recommend top rated movies 
'''

def calc_recommendation(user):
    for i in range(1, 3953):
        temp = cosine_similarity(sparse_matrix_training[user], sparse_matrix_training[i])
        list_cosines.append(temp)
        
        
taken_user = int(input('Please enter a user '))

calc_recommendation(taken_user)


temp = heapq.nlargest(50, list_cosines)

 
most_similar = []
for i in range(3952) : 
    for j in range(50):
        if (list_cosines[i] == temp[j]):
            most_similar.append(i+1)
            


rating_vector = np.zeros(3883, dtype=int)

hehe = sparse_matrix_training[taken_user, :].toarray()
sparse_matrix_training = sparse_matrix_training.toarray()


for i in range(1, 3884):
    if (sparse_matrix_training[taken_user][i] != 0):
        rating_vector[i] = sparse_matrix_training[taken_user][i]
       
        
recommendation = []

for i in (most_similar):
    for j in range(1, 3884):
        if (sparse_matrix_training[i][j] == 5):
            recommendation.append(j)
        



recommendation_non_duplicates = list(dict.fromkeys(recommendation))
print(recommendation_non_duplicates)


a = np.zeros(851, dtype=int)
b = np.zeros(851, dtype=int)

'''
for i in range(851):
    array_2d[i][1] = recommendation.count( recommendation_non_duplicates[i] )        
    array_2d[0][i] = recommendation_non_duplicates[i]
    
'''
        
for i in range(851):
    a[i] = recommendation_non_duplicates[i]
        
    
for i in range(851):
    movie_temp = a[i]
    b[i] = recommendation.count(movie_temp)
        
    
final_count = np.zeros((851,2), dtype=int)

for i in range(851):
    final_count[i][0] = a[i]
    final_count[i][1] = b[i]
    
from operator import itemgetter
final_count = sorted(final_count,key=itemgetter(1), reverse=True)

top_recommendations = final_count[0:10]
top_recommendations_movies =  [row[0] for row in top_recommendations]


print("Movies you would like are")

for j in (top_recommendations_movies):
    for i in range(3883):
        if(movies.iloc[i][0] == j):
            print(movies.iloc[i][1])
            
