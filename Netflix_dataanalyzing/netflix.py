import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


#load data
df=pd.read_csv('/Users/jayraj/Desktop/study/python/UnifiedMentor_projects/Netflix_dataanalyzing/netflix1.csv')
print(df.columns,df.head())

#review data

print(df.shape,df.info())

#clean and formate data

# Check for missing values
print(df.isnull().sum())
# Drop duplicates if any
df.drop_duplicates(inplace=True)
# Drop rows with missing critical information data.dropna(subset=['director', 'cast', 'country'], inplace=True)

df['date_added'] = pd.to_datetime(df['date_added'])
df['duration_value'] = df['duration'].str.extract('(\d+)').astype(float)
df['duration_unit'] = df['duration'].str.extract('([a-zA-Z]+)')
print(df['director'].value_counts())
print(df['country'].value_counts())
print(df['type'].value_counts())

# Count and sort release years
release_year_counts = df['release_year'].value_counts().sort_index()
plt.figure(figsize=(15, 5)) 
plt.bar(release_year_counts.index, release_year_counts.values, color='skyblue')
plt.title('Number of Titles Released Each Year', fontsize=16)
plt.xlabel('Release Year', fontsize=14)
plt.ylabel('Number of Titles', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#Analyze popular genres:
# Split genres and count occurrences
genre_counts = df['listed_in'].str.split(',').explode().str.strip().value_counts()
genre_counts.head(10).plot(kind='barh', figsize=(10, 6), color='lightblue', edgecolor='black')
plt.title('Top 10 Popular Genres', fontsize=16)
plt.xlabel('Count', fontsize=14)
plt.ylabel('Genres', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


#Analyze Ratings Distribution
rating_counts = df['rating'].value_counts()
rating_counts.plot(kind='bar', figsize=(10, 5), color='lightcoral', edgecolor='black')
plt.title('Distribution of Content Ratings', fontsize=16)
plt.xlabel('Ratings', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


#Analyze Content Added Over Time
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
df['year_added'] = df['date_added'].dt.year

content_added_per_year = df['year_added'].value_counts().sort_index()

content_added_per_year.plot(kind='line', marker='o', figsize=(12, 6), color='green')
plt.title('Number of Titles Added Per Year', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Number of Titles', fontsize=14)
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()

#compair movie and tv show
type_counts = df['type'].value_counts()
type_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['skyblue', 'orange'], figsize=(8, 8))
plt.title('Proportion of Movies vs. TV Shows', fontsize=16)
plt.ylabel('') 
plt.show()

#explor most frequent director

director_counts = df['director'].value_counts().head(10)

director_counts.plot(kind='barh', figsize=(10, 6), color='lightgreen', edgecolor='black')
plt.title('Top 10 Most Frequent Directors', fontsize=16)
plt.xlabel('Count', fontsize=14)
plt.ylabel('Directors', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#explor content by country

country_counts = df['country'].value_counts().head(10)

country_counts.plot(kind='barh', figsize=(10, 6), color='lightblue', edgecolor='black')
plt.title('Top 10 Countries with Most Content on Netflix', fontsize=16)
plt.xlabel('Count', fontsize=14)
plt.ylabel('Countries', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#Analyze Average Duration for Movies and TV Shows

df['duration_value'] = df['duration'].str.extract('(\d+)').astype(float)

movie_duration = df[df['type'] == 'Movie']['duration_value'].mean()
tv_show_seasons = df[df['type'] == 'TV Show']['duration'].str.extract('(\d+)').astype(float).mean()
print(f"Average Movie Duration: {movie_duration} minutes")
print(f"Average TV Show Seasons: {tv_show_seasons} seasons")


#Identify the Most Popular Content by Rating
popular_content_by_rating = df.groupby('rating')['title'].count().sort_values(ascending=False)
popular_content_by_rating.plot(kind='bar', figsize=(10, 5), color='lightgreen', edgecolor='black')
plt.title('Most Popular Content by Rating', fontsize=16)
plt.xlabel('Rating', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(fontsize=12, rotation=45)
plt.tight_layout()
plt.show()

#Find Most Common Combinations of Genres
genre_combinations = df['listed_in'].str.split(',').explode().str.strip()
genre_combinations_counts = genre_combinations.value_counts().head(10)
genre_combinations_counts.plot(kind='barh', figsize=(10, 6), color='lightblue', edgecolor='black')
plt.title('Top 10 Most Common Genre Combinations', fontsize=16)
plt.xlabel('Count', fontsize=14)
plt.ylabel('Genre Combinations', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



print(release_year_counts,genre_counts,rating_counts,content_added_per_year,type_counts,director_counts,country_counts,movie_duration,tv_show_seasons,popular_content_by_rating,genre_combinations)