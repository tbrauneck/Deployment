import streamlit as st
import pickle
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Load the saved model
with open("knn_recommender.sav", "rb") as file:
    model = pickle.load(file)

# define number of predictions, k
k=5

# Define Genre as an empty list; it will be filled by user input
Genre = []

# Prepare a disctionay for the genre information of the movie from which we are building the prediction
watched_movie_data = pd.DataFrame({
     'Biography': [0],
     'Drama': [0],
     'Thriller': [0],
     'Comedy': [0],
     'Crime': [0],
     'Mystery': [0],
     'History': [0]
})

df = pd.read_csv(r"https://github.com/ArinB/MSBA-CA-Data/raw/main/CA05/movies_recommendation_data.csv")
model_data = df.sort_values(by='IMDB Rating', ascending=False) # sort the data by rating, highest first
y = model_data.pop('Movie Name')
titles = y.tolist()

# Custom function to return the recommended movies based on the KNN model

def recommend_movies(watched_movie_data, n_recommendations=k):
    
    # Find nearest neighbors for the new movie
    distances, indices = model.kneighbors(watched_movie_data, n_neighbors=k)
    
    # Get recommendations with their IMDb ratings
    recommendations = []
    for i in range(len(distances.flatten())): # loop for each recommendation, i.e., k times
        movie_index = indices.flatten()[i] # Get the index of the movie
        movie_title = y.iloc[movie_index] # Find the index in the y dataframe to get the movie title
        imdb_rating = model_data.iloc[movie_index]['IMDB Rating']
        recommendations.append((movie_title, f"{imdb_rating:.1f}"))
    
    # Sort recommendations by IMDb rating in descending order; we will show the most highly rated movies first
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    # Convert output into a formatted dataframe
    recommendations_df = pd.DataFrame(recommendations, columns=['Movie Title', 'IMDb Rating'])
    
    return recommendations_df


# Streamlit app

st.title("Movie Recommender")
st.write("Name a movie you enjoyed.")

# Input fields
Title = st.text_input('Title')
default_genre = []
genres = []
if Title in titles:
    row = titles.index(Title)
    genres = model_data.iloc[row].loc[['Biography','Drama','Thriller','Comedy','Crime','Mystery','History']]
    default_genre = genres[genres == 1].index.tolist()
    k=6 # increase number of predictions, k, if title is in the training set; the title entered will be returned in the results and will be filtered out
else:
    k=5

Genre = st.multiselect('What genre(s) is this movie?', list(watched_movie_data.keys()), default=default_genre)
    


if st.button("Get my recommendations"):
 
    # Update genre to 1 if the genre was selected
    for item in watched_movie_data:
        if item in Genre:
            watched_movie_data[item] = 1
        else:
            watched_movie_data[item] = 0


    #result = display(recommend_movies(watched_movie_data,k).style.hide(axis="index"))
    result = recommend_movies(watched_movie_data,k)
    result = result[result['Movie Title']!= Title]

    # Renumber the index
    result = result.reset_index(drop=True)
    result.index = result.index + 1
    st.write(f"Based on your selection, you may like:")
    st.dataframe(result)
    
