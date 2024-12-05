# Import Libraries
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel

# Load Data
ratings = pd.read_csv('ratings_small.csv')
credits = pd.read_csv('credits.csv')
metadata = pd.read_csv('movies_metadata.csv', low_memory=False)
links = pd.read_csv('links.csv')
keywords = pd.read_csv('keywords.csv')

# Display dataset summaries
print("Metadata Info:")
print(metadata.info())
print("\nCredits Info:")
print(credits.info())
print("\nKeywords Info:")
print(keywords.info())
print("\nRatings Info:")
print(ratings.info())

# Check initial missing values
datasets = {'Metadata': metadata, 'Credits': credits, 'Keywords': keywords, 'Links': links, 'Ratings': ratings}
for name, df in datasets.items():
    print(f"\n{name} Missing Values:")
    print(df.isnull().sum())

# Merge Datasets
# Ensure 'id' is of string type for consistent merging
metadata['id'] = metadata['id'].astype(str)
credits['id'] = credits['id'].astype(str)
keywords['id'] = keywords['id'].astype(str)

# Merge credits and keywords into metadata
metadata = metadata.merge(credits, on='id', how='left')
metadata = metadata.merge(keywords, on='id', how='left')

print("\nColumns after merging:")
print(metadata.columns)

# Handle Missing Values
missing_data = metadata.isnull().sum()
print("\nMissing Data After Merging:")
print(missing_data[missing_data > 0])

# Drop columns with >50% missing values
missing_percentage = metadata.isnull().sum() / len(metadata) * 100
columns_to_drop = missing_percentage[missing_percentage > 50].index
metadata.drop(columns=columns_to_drop, inplace=True)

# Fill missing values
metadata.fillna({
    'revenue': 0,
    'runtime': metadata['runtime'].mean(),
    'budget': 0,
    'popularity': 0
}, inplace=True)

# Check shape after cleaning
print("\nShape after cleaning:", metadata.shape)

# Handle Data Types
# Convert boolean-like columns
if 'adult' in metadata.columns:
    metadata['adult'] = metadata['adult'].map({'True': 1, 'False': 0}).fillna(0).astype(int)

# Convert numeric columns
numeric_columns = ['budget', 'revenue', 'popularity']
for col in numeric_columns:
    if col in metadata.columns:
        metadata[col] = pd.to_numeric(metadata[col], errors='coerce').fillna(0)

# Convert release_date to datetime
if 'release_date' in metadata.columns:
    metadata['release_date'] = pd.to_datetime(metadata['release_date'], errors='coerce')

# Drop rows with critical missing values (like title or id)
metadata.dropna(subset=['id', 'title'], inplace=True)

# Check correlations among numeric columns
numeric_metadata = metadata.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_metadata.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Final Overview
print("\nMetadata Info After Cleaning:")
print(metadata.info())

# Missing percentage for the final dataset
missing_percentage_final = metadata.isnull().sum() / len(metadata) * 100
print("\nFinal Missing Percentage:")
print(missing_percentage_final)

# Display a preview of the cleaned dataset
print("\nCleaned Dataset Preview:")
print(metadata.head())

# Combine Features
metadata['combined_features'] = (
    metadata['genres'].fillna('') + ' ' +
    metadata['keywords'].fillna('') + ' ' +
    metadata['cast'].fillna('') + ' ' +
    metadata['crew'].fillna('')
)

# Limit TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
tfidf_matrix = vectorizer.fit_transform(metadata['combined_features'])

# Merge with Links to create IMDb URL
# Convert 'movieId' to string to avoid type mismatch during merge
links['movieId'] = links['movieId'].astype(str)

# Now merge with Links to create IMDb URL
metadata = metadata.merge(links, left_on='id', right_on='movieId', how='left')

# Create IMDb URL
metadata['imdb_url'] = 'https://www.imdb.com/title/tt' + metadata['imdbId'].astype(str).str.zfill(7)

# Fill missing IMDb URLs
metadata['imdb_url'] = metadata['imdb_url'].fillna('Unavailable')

# Efficient similarity calculation
def recommend_movies(title, metadata, tfidf_matrix, top_n=10):
    indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()
    if title not in indices:
        return f"'{title}' not found in the dataset."
    
    idx = indices[title]
    similarity_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_indices = np.argpartition(similarity_scores, -top_n)[-top_n:]
    similar_indices = similar_indices[np.argsort(similarity_scores[similar_indices])[::-1]]
    
    return metadata.iloc[similar_indices][['title', 'vote_average', 'vote_count', 'imdb_url']]

# Test recommendation system
movie_title = "Avatar"
recommendations = recommend_movies(movie_title, metadata, tfidf_matrix, top_n=10)
print(f"Recommendations for '{movie_title}':")
print(recommendations)
