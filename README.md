Problem Statement:

1) Customer Behaviour and it’s prediction lies at the core of every Business Model. From Stock Exchange, e-Commerce and Automobile to even Presidential Elections, 
predictions serve a great purpose. Most of these predictions are based on the data available about a person’s activity either online or in-person.

2) Recommendation Engines are the much needed manifestations of the desired Predictability of User Activity. Recommendation Engines move one step further and
not only give information but put forth strategies to further increase users interaction with the platform.

3) In today’s world OTT platform and Streaming Services have taken up a big chunk in the Retail and Entertainment industry. Organizations like Netflix, Amazon etc. 
analyse User Activity Pattern’s and suggest products that better suit the user needs and choices.

4) For the purpose of this Project we will be creating one such Recommendation Engine from the ground-up, where every single user, based on there area of interest and ratings, 
would be recommended a list of movies that are best suited for them.

Database Information:

1. ID–Contains the separate keys for customer and movies. 
2. Rating– A section contains the user ratings for all the movies. 
3. Genre–Highlights the category of the movie.
4. Movie Name–Name of the movie with respect to the movie id.

Objective:

1. Find out the list of most popular and liked genre
2. Create Model that finds the best suited Movie for one user in every genre.
3. Find what Genre Movies have received the best and worst ratings based on User Rating.


Implementation of code:

Netflix Movie Recommendation System:

This project implements a recommendation system for Netflix movies using collaborative filtering techniques. Collaborative filtering is a method used by recommender systems to make predictions about the interests of a user by collecting preferences from many users.

Table of Contents
    Introduction
    Dataset
    Data Preprocessing
    Model Training
    Evaluation
    Recommendations
    Dependencies
    Usage
    Contributing
    License
    
Introduction:

Netflix has a vast collection of movies and TV shows, making it challenging for users to find something to watch. To address this issue, we developed a recommendation system that suggests movies to users based on their historical ratings and the ratings of similar users. The system employs Singular Value Decomposition (SVD) algorithm, a collaborative filtering technique, to predict user ratings for movies.

Dataset:

The dataset used in this project is a subset of the Netflix Prize dataset, containing ratings given by users to different movies. It consists of three columns: Customer ID, Movie ID, and Rating.

Data Preprocessing:

Before training the model, we performed several data preprocessing steps, including handling missing values, removing inactive users and less frequently rated movies, and creating a ratings matrix.

Model Training:

We trained the recommendation model using the Surprise library, which offers various collaborative filtering algorithms. We chose the SVD algorithm for its effectiveness in handling sparse matrices and its ability to capture latent factors underlying user preferences.

Evaluation:

To evaluate the performance of our model, we used cross-validation and computed the Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE). These metrics provide insights into how well the model predicts user ratings compared to the actual ratings.

Recommendations:

After training the model, we generated recommendations for users based on their historical ratings. The recommendations are sorted in descending order of predicted ratings, and the top recommendations are displayed for each user.

Dependencies:

The following Python libraries are required to run the code:

pandas
numpy
matplotlib
scikit-surprise
You can install the required dependencies using pip:
  pip install pandas numpy matplotlib scikit-surprise

  
Usage:

To use the recommendation system:

Clone the repository to your local machine.
Open the Jupyter Notebook Netflix_Recommendation.ipynb.
Run each cell in the notebook sequentially to preprocess the data, train the model, and generate recommendations.

Contributing:

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.
