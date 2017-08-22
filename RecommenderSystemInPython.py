
# coding: utf-8

# # Building Recommender Systems in Python using MovieLens data
# 
# Recommender systems are used to personalize your experience on the web, telling you what to buy, where to eat or even who you should be friends with. People's tastes vary, but generally follow patterns. People tend to like things that are similar to other things they like, and they tend to have similar taste as other people they are close with. Recommender systems try to capture these patterns to help predict what else you might like. E-commerce, social media, video and online news platforms have been actively deploying their own recommender systems to help their customers to choose products more efficiently, which serves win-win strategy.
# 
# Two most ubiquitous types of recommender systems are Content-Based and Collaborative Filtering (CF). Collaborative filtering produces recommendations based on the knowledge of users’ attitude to items, that is it uses the “wisdom of the crowd” to recommend items. In contrast, content-based recommender systems focus on the attributes of the items and give you recommendations based on the similarity between them.
# 
# In general, Collaborative filtering (CF) is the workhorse of recommender engines. The algorithm has a very interesting property of being able to do feature learning on its own, which means that it can start to learn for itself what features to use. CF can be divided into Memory-Based Collaborative Filtering and Model-Based Collaborative filtering. In this tutorial, you will implement Model-Based CF by using singular value decomposition (SVD) and Memory-Based CF by computing cosine similarity.
# 
# You will use MovieLens dataset, which is one of the most common datasets used when implementing and testing recommender engines. It contains 100k movie ratings from 943 users and a selection of 1682 movies.
# 
# The data can be downloaded from here : http://files.grouplens.org/datasets/movielens/ml-100k.zip.  Also read more on the data from :  http://files.grouplens.org/datasets/movielens/ml-100k-README.txt

# Lets begin by importing libraries

# In[1]:

import numpy as np
import pandas as pd


# Now read in the u.data file, which contains the full dataset

# ## Reading the data

# In[3]:

header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=header)


# In[4]:

df.head(5)


# Next, let's count the number of unique users and movies.

# In[5]:

n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]


# In[6]:

print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))


# ## Splitting the data
# 
# Now split the data into two datasets according to the percentage of test examples (test_size), which in this case is 0.25.

# In[10]:

from sklearn.cross_validation import train_test_split as cv
train_data, test_data = train_test_split(df, test_size = 0.25, random_state = 0)


# ### Memory-Based Collaborative Filtering

# Memory-Based Collaborative Filtering approaches can be divided into two main sections: 
# 1. user-item filtering - “Users who are similar to you also liked …”
# 2. item-item filtering - “Users who liked this item also liked …”

# In both cases, we create a user-item matrix which we build from the entire dataset. Since we have split the data into testing and training we will need to create two 943 ×× 1682 matrices. The training matrix contains 75% of the ratings and the testing matrix contains 25% of the ratings.

# After we have built the user-item matrix we calculate the similarity and create a similarity matrix. The similarity values between items in Item-Item Collaborative Filtering are measured by observing all the users who have rated both items. For User-Item Collaborative Filtering the similarity values between users are measured by observing all the items that are rated by both users.

# In[11]:

#Create two user-item matrices, one for training and another for testing
train_data_matrix = np.zeros((n_users, n_items))
for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]
    
test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]


# A distance metric commonly used in recommender systems is cosine similarity, where the entities are seen as vectors in nn-dimensional space and the similarity is calculated based on the angle between these vectors. 

# Cosine similiarity for users k and a can be calculated using the formula below, where you take dot product of the user vectors, and divide it by multiplication of the Euclidean lengths of the vectors.

# \begin{equation}
#     s_{u}^{cos} (u_{k},u_{a}) = u_{k}.u_{a}/[||u_{k}||*||u_{a}||]
# \end{equation}
# 

# In Python, we use the pairwise_distances function from sklearn to calculate the cosine similarity. Note, the output will range from 0 to 1 since the ratings are all positive.

# In[12]:

from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')


# Next step is to make predictions. You have already created similarity matrices: user_similarity and  item_similarity and therefore you can make a prediction by applying following formula for user-based CF:

# \begin{equation}
# \hat{x}_{k,m} = \bar{x}_{k} + \frac{\sum\limits_{u_a} sim_u(u_k, u_a) (x_{a,m} - \bar{x}_{u_a})}{\sum\limits_{u_a}|sim_u(u_k, u_a)|}
# \end{equation}

# You can look at the similarity between users k and a as weights that are multiplied by the ratings of a similar user aa (corrected for the average rating of that user). You will need to normalize it so that the ratings stay between 1 and 5 and, as a final step, sum the average ratings for the user that you are trying to predict.
# 
# When making a prediction for item-based CF you don't need to correct for users average rating since query user itself is used to do predictions:

# \begin{equation}
# \hat{x}_{k,m} = \frac{\sum\limits_{i_b} sim_i(i_m, i_b) (x_{k,b}) }{\sum\limits_{i_b}|sim_i(i_m, i_b)|}
# \end{equation}

# ## Making Prediction

# Lets write a predit function based on above, as follows

# In[13]:

def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred


# In[14]:

item_prediction = predict(train_data_matrix, item_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user')


# There are many evaluation metrics but one of the most popular metric used to evaluate accuracy of predicted ratings is Root Mean Squared Error (RMSE). Since we only want to consider predicted ratings that are in the test dataset, we filter out all other elements in the prediction matrix with prediction[ground_truth.nonzero()]

# In[15]:

from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))


# Now print the errors

# In[17]:

print('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
print('Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))


# Memory-based algorithms are easy to implement and produce reasonable prediction quality. The drawback of memory-based CF is that it doesn't scale to real-world scenarios and doesn't address the well-known cold-start problem, that is when new user or new item enters the system. Model-based CF methods are scalable and can deal with higher sparsity level than memory-based models, but also suffer when new users or items that don't have any ratings enter the system.

# ### Model-Based Collaborative Filtering

# Model-based Collaborative Filtering is based on matrix factorization (MF) which has received greater exposure, mainly as an unsupervised learning method for latent variable decomposition and dimensionality reduction. Matrix factorization is widely used for recommender systems where it can deal better with scalability and sparsity than Memory-based CF. A well-known matrix factorization method is Singular value decomposition (SVD). Collaborative Filtering can be formulated by approximating a matrix X by using singular value decomposition. The general equation can be expressed as follows:

# \begin{equation}
#     X=U \times S \times V^T
# \end{equation}

# Given an m×n matrix X:
# 
# U is an m×r orthogonal matrix
# 
# S is an r×r diagonal matrix with non-negative real numbers on the diagonal
# 
# V is an n×r orthogonal matrix
# 
# Elements on the diagnoal in S are known as singular values of X. Matrix X can be factorized to U, S and V. The U matrix represents the feature vectors corresponding to the users in the hidden feature space and the V matrix represents the feature vectors corresponding to the items in the hidden feature space.

# Now you can make a prediction by taking dot product of U, S and transpose of V. The below code makes use of SVD's  from the scipy module

# In[18]:

import scipy.sparse as sp
from scipy.sparse.linalg import svds

#get SVD components from train matrix. Choose k.
u, s, vt = svds(train_data_matrix, k = 20)
s_diag_matrix=np.diag(s)
X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
print('User-based CF MSE: ' + str(rmse(X_pred, test_data_matrix)))


# ## Concluding Remarks

# 1. In this notebook, we have covered how to implement simple Collaborative Filtering methods, both memory-based CF and model-based CF.
# 2. Memory-based models are based on similarity between items or users, where we use cosine-similarity.
# 3. Model-based CF is based on matrix factorization where we use SVD to factorize the matrix.
# 4. Building recommender systems that perform well in cold-start scenarios (where little data is available on new users and items) remains a challenge. The standard collaborative filtering method performs poorly is such settings.
