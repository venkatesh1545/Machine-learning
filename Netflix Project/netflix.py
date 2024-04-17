# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Reading dataset file
dataset = pd.read_csv('E:/AIML_Projects/Netflix/combined_data_1.txt', header=None, names=['Cust_Id', 'Rating'],usecols=[0, 1])
# not named---->we are naming the columns

# Convert Ratings column to a float
dataset['Rating'] = dataset['Rating'].astype(float)
dataset.dtypes            # Add Print to display dataset
dataset.shape            # Add Print to display dataset
print(dataset.head())            # Add Print to display dataset

# to find the distribution of different ratings in the dataset
p = dataset.groupby('Rating')['Rating'].agg(['count'])
# 3 ----> 1st row
# 3 ---->10th row
# 3   merge it
print(p)

# get movie_count
movie_count = dataset.isnull().sum()[1]
print('Total count of movies: ',movie_count)

# get customer count
cust_count = dataset['Cust_Id'].nunique()-movie_count
print('Total count of custoemrs: ',cust_count)

# get rating count
rating_count = dataset['Cust_Id'].count() - movie_count
print('Total count of ratings: ',rating_count)

# To plot the distribution of the ratings in as a bar plot
ax = p.plot(kind= 'barh', legend= False, figsize=(15,10))
ax.set_xlabel('Number of Ratings')
ax.set_ylabel('Rating')
ax.set_title('Distribution of Ratings', fontsize=20)
plt.show()
print('Plot displayed successfully')

# To create a numpya array containing movie ids corresponding to the rows in the 'ratings' dataset

# To count all the 'nan' values in the ratings column in the 'ratings' dataset
df_nan = pd.DataFrame(pd.isnull(dataset.Rating))
print(df_nan.head())

# To store the index of all the rows containing 'nan' values
df_nan = df_nan[df_nan['Rating'] == True]
print(df_nan.shape)

# To rest the index of the dataframe
df_nan = df_nan.reset_index()
print(df_nan.head())

# To create a numpy array containing movie ids according to the 'ratings' dataset
movie_np = []
movie_id = 1
#into tuple
for i,j in zip(df_nan['index'][1:],df_nan['index'][:-1]):
    # numpy Approach
    temp = np.full((1,i-j-1), movie_id)
    movie_np = np.append(movie_np,temp)
    movie_id += 1

# Account for last record and corresponding length
# numpy approach
last_record = np.full((1,len(dataset) - df_nan.iloc[-1,0] - 1), movie_id)
movie_np = np.append(movie_np, last_record)
print(f'Movie Numpy: {movie_np}')
print(f'Length: {len(movie_np)}')

x= zip(df_nan['index'][1:],df_nan['index'][:-1])
temp = np.full((1,547),1)
print(temp)
tuple(x)

# to append the above created array to the dataset after removing the 'nan' rows
dataset = dataset[pd.notnull(dataset['Rating'])]  #movies count so nan
dataset['Movie_Id'] = movie_np.astype(int)
dataset['Cust_Id'] = dataset['Cust_Id'].astype(int)
print('-Dataset Examples-')
print(dataset.head())

print(dataset.shape)

# Data cleaning:
f = ['count','mean']
#count and mean 
#To create a list of all the movies rated less often(only include top 30% rated movies)
dataset_movie_summary = dataset.groupby('Movie_Id')['Rating'].agg(f)
dataset_movie_summary.index = dataset_movie_summary.index.map(int)
movie_benchmark = round(dataset_movie_summary['count'].quantile(0.7),0)
drop_movie_list = dataset_movie_summary[dataset_movie_summary['count'] < movie_benchmark].index
print('Movie minimum times of review: {}'.format(movie_benchmark))

#To create a list of all the inactive users(users who rate less often)
dataset_cust_summary = dataset.groupby('Cust_Id')['Rating'].agg(f) #mean and count of customer id
dataset_cust_summary.index = dataset_cust_summary.index.map(int) #integer format
cust_benchmark = round(dataset_cust_summary['count'].quantile(0.7),0)
drop_cust_list = dataset_cust_summary[dataset_cust_summary['count'] < cust_benchmark].index
#                                           6                       <  5
#drop_cust_list[3,4]
print(f'Customer minimum times of review: {cust_benchmark}')

print(f'Original Shape: {dataset.shape}')

dataset = dataset[~dataset['Movie_Id'].isin(drop_movie_list)]
#in dataset --->in all movie ids------>if the whic ever  movies we put under drop list are present or not---
#if present drop /remove them
dataset = dataset[~dataset['Cust_Id'].isin(drop_cust_list)]
#removing all inactive users from our dataset
print('After Trim Shape: {}'.format(dataset.shape))

print('-Data Examples-')
dataset.head()

# create ratings matrix for 'ratings' matrix with rows = userld, columns = movield
df_p = pd.pivot_table(dataset,values='Rating',index='Cust_Id',columns='Movie_Id')

print(df_p.shape)

print(df_p.head())

# To load the movie_titles dataset
df_title = pd.read_csv('movie_titles.csv', encoding = "ISO-8859-1", header = None, names = ['Movie_Id', 'Year', 'Name'])

df_title.set_index('Movie_Id', inplace = True)

print (df_title.head(10))


import math

import matplotlib.pyplot as plt

from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

# Load Reader library
reader = Reader()

# get just top 100K rows for faster run time
data = Dataset.load_from_df(dataset[['Cust_Id', 'Movie_Id', 'Rating']][:100000], reader)

# Use the SVD algorithm.
svd = SVD()

# Compute the RMSE of the SVD algorithm
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)

print(dataset.head())

dataset_712664 = dataset[(dataset['Cust_Id'] == 712664) & (dataset['Rating'] == 5)]
dataset_712664 = dataset_712664.set_index('Movie_Id')
dataset_712664 = dataset_712664.join(df_title)['Name']
dataset_712664.head(10)

# Create a shallow copy for the movies dataset
user_712664 = df_title.copy()

user_712664 = user_712664.reset_index()

#To remove all the movies rated less often 
user_712664 = user_712664[~user_712664['Movie_Id'].isin(drop_movie_list)]


# getting full dataset
data = Dataset.load_from_df(dataset[['Cust_Id', 'Movie_Id', 'Rating']], reader)

#create a training set for svd
trainset = data.build_full_trainset()
svd.fit(trainset)

#Predict the ratings for user_712664
user_712664['Estimate_Score'] = user_712664['Movie_Id'].apply(lambda x: svd.predict(712664, x).est)

#Drop extra columns from the user_712664 data frame
user_712664 = user_712664.drop('Movie_Id', axis = 1)

# Sort predicted ratings for user_712664 in descending order
user_712664 = user_712664.sort_values('Estimate_Score', ascending=False)

#Print top 10 recommendations
print(user_712664.head(10))

# Load movie_titles dataset
df_title = pd.read_csv('E:/AIML_Projects/Netflix/movie_titles.csv', encoding="ISO-8859-1", header=None, names=['Movie_Id', 'Year', 'Name'])
df_title.set_index('Movie_Id', inplace=True)
print(df_title.head(10))

# Model training
reader = Reader()
data = Dataset.load_from_df(dataset[['Cust_Id', 'Movie_Id', 'Rating']], reader)
svd = SVD()
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)

# Recommend movies for a specific user
user_id = 712664
user_data = dataset[(dataset['Cust_Id'] == user_id) & (dataset['Rating'] == 5)]
user_data = user_data.set_index('Movie_Id')
user_data = user_data.join(df_title)['Name']
print(user_data.head(10))

# Final recommendations
trainset = data.build_full_trainset()
svd.fit(trainset)
user_movies = df_title.copy().reset_index()
user_movies = user_movies[~user_movies['Movie_Id'].isin(drop_movie_list)]
user_movies['Estimate_Score'] = user_movies['Movie_Id'].apply(lambda x: svd.predict(user_id, x).est)
user_movies = user_movies.drop('Movie_Id', axis=1)
user_movies = user_movies.sort_values('Estimate_Score', ascending=False)
print(user_movies.head(10))