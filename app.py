from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import chromadb

# Specify the directory where you want the ChromaDB database files to be stored
db_directory = r"C:\Users\lavak\Search Engine Project"

# Connect to ChromaDB
client = chromadb.PersistentClient(path=db_directory)

# Specify the name of the collection
collection_name = "document_embeddings"

# Create or get the collection
collection = client.get_or_create_collection(
    name=collection_name,
    metadata={"hnsw:space": "cosine"}  # Assuming cosine similarity
    )


# Create Flask app
app = Flask(__name__)

# Load your DataFrame data here
data = pd.read_csv(r"C:\Users\lavak\Search Engine Project\cleaned_subtitle_data.csv")

# Initialize CountVectorizer
vectorizer = CountVectorizer(max_features=1000)
X = vectorizer.fit_transform(data['cleaned_file_content'])

# Function to search subtitles
def search_subtitles(query, vectorizer, X):
    query_vector = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, X)
    top_indices = np.argsort(cosine_similarities[0])[::-1]
    top_matching_movies = data.iloc[top_indices]['name'].tolist()
    return top_matching_movies[:5]

# Route for the home page
@app.route('/')
def home():
    return render_template('index1.html')

# Route for handling the search query
@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    top_matching_movies = search_subtitles(query, vectorizer, X)
    return render_template('results1.html', movies=top_matching_movies)

# Route for accessing subtitles of a specific movie
@app.route('/subtitle/<movie_name>')
def get_subtitle(movie_name):
    movie_row = data[data['name'] == movie_name].iloc[0]
    subtitle = movie_row['cleaned_content']
    return subtitle

if __name__ == '__main__':
    app.run(debug=True)