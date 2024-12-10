import pandas as pd
import matplotlib.pyplot as plt
# Load data
df = pd.read_csv('/Users/jayraj/Desktop/study/gitdemo/code-demo/Netflix_dataanalyzing/netflix1.csv')

# Review data
print(df.columns, df.head())
print(df.shape, df.info())

# Clean and format data
# Replace 'Not Given' with NaN and drop rows with critical missing information
df.replace({'Not Given': None}, inplace=True)
df.dropna(subset=['director', 'country', 'duration'], inplace=True)

# Check for missing values
df.isnull().sum()

# Drop duplicates
df.drop_duplicates(inplace=True)


# Convert 'date_added' to datetime
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')

# Extract 'duration_value' and 'duration_unit' using raw strings 
df['duration_value'] = df['duration'].str.extract(r'(\d+)').astype(float)
df['duration_unit'] = df['duration'].str.extract(r'([a-zA-Z]+)')



#Step 4: Exploratory Data Analysis (EDA)
#1. Content Type Distribution (Movies vs. TV Shows)

# Compare Movies and TV Shows
type_counts = df['type'].value_counts()
type_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['skyblue', 'orange'], figsize=(8, 8))
plt.title('Proportion of Movies vs. TV Shows', fontsize=16)
plt.ylabel('')
plt.show()  


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

# Analyze popular genres
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

# Analyze Ratings Distribution
rating_counts = df['rating'].value_counts()
print(rating_counts)
rating_counts.plot(kind='bar', figsize=(10, 5), color='lightcoral', edgecolor='black')
plt.title('Distribution of Content Ratings', fontsize=16)
plt.xlabel('Ratings', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
rating_counts.plot.pie(
    figsize=(10, 10), autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors,
    wedgeprops={'edgecolor': 'black'}, textprops={'fontsize': 9}
)
plt.title('Distribution of Content Ratings', fontsize=16)
plt.ylabel('')
plt.show()

#Converting date_added column to datetime.
df['date_added']=pd.to_datetime(df['date_added'])
print(df.describe())
# Explore Content by Country
country_counts = df['country'].value_counts().head(10)
print(country_counts)
country_counts.plot(kind='barh', figsize=(10, 6), color='lightblue', edgecolor='black')
plt.title('Top 10 Countries with Most Content on Netflix', fontsize=16)
plt.xlabel('Count', fontsize=14)
plt.ylabel('Countries', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



# Analyze Content Added Over Time
df['year_added'] = df['date_added'].dt.year
content_added_per_year = df['year_added'].value_counts().sort_index()
print("add  over time",content_added_per_year)
content_added_per_year.plot(kind='line', marker='o', figsize=(12, 6), color='green')
plt.title('Number of Titles Added Per Year', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Number of Titles', fontsize=14)
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()


df['year']=df['date_added'].dt.year
df['month']=df['date_added'].dt.month
df['day']=df['date_added'].dt.day

#Monthly releases of Movies and TV shows on Netflix
monthly_movie_release=df[df['type']=='Movie']['month'].value_counts().sort_index()
monthly_series_release=df[df['type']=='TVShow']['month'].value_counts().sort_index()
plt.plot(monthly_movie_release.index, monthly_movie_release.values, label='Movies')
plt.plot(monthly_series_release.index, monthly_series_release.values,label='Series')
plt.xlabel("Months")
plt.ylabel("Frequency of releases")
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend()
plt.grid(True)
plt.suptitle("Monthly releases of Movies and TV shows on Netflix")
plt.show()


#Top 10 popular movie genres
popular_movie_genre=df[df['type']=='Movie'].groupby("listed_in").size().sort_values(ascending=False)[:10]
popular_series_genre=df[df['type']=='TVShow'].groupby("listed_in").size().sort_values(ascending=False)[:10]
print(popular_movie_genre,popular_series_genre)
plt.bar(popular_movie_genre.index, popular_movie_genre.values)
plt.xticks(rotation=45, ha='right')
plt.xlabel("Genres")
plt.ylabel("Movies Frequency")
plt.suptitle("Top 10 popular genres for movies on Netflix")
plt.show()

# Explore Most Frequent Directors
director_counts = df['director'].value_counts().head(10)
print(director_counts)
director_counts.plot(kind='barh', figsize=(10, 6), color='lightgreen', edgecolor='black')
plt.title('Top 10 Most Frequent Directors', fontsize=16)
plt.xlabel('Count', fontsize=14)
plt.ylabel('Directors', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Identify the Most Popular Content by Rating
popular_content_by_rating = df.groupby('rating')['title'].count().sort_values(ascending=False)
popular_content_by_rating.plot(kind='bar', figsize=(10, 5), color='lightgreen', edgecolor='black')
plt.title('Most Popular Content by Rating', fontsize=16)
plt.xlabel('Rating', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(fontsize=12, rotation=45)
plt.tight_layout()
plt.show()

# Find Most Common Combinations of Genres
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




