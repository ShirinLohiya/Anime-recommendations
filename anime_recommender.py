import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import html

# loading the data
anime = pd.read_csv('anime.csv')
rating = pd.read_csv('rating.csv')

# Clean HTML encodings in the name column
anime['name'] = anime['name'].apply(lambda x: html.unescape(str(x)) if pd.notnull(x) else x)

# Filter out -1 ratings (which indicate missing ratings)
valid_ratings = rating[rating['rating'] != -1]
print(f"Filtered out {len(rating) - len(valid_ratings)} ratings with value -1")

# Calculate average ratings using only valid ratings
avg_ratings = valid_ratings.groupby("anime_id")["rating"].mean().reset_index().rename(columns={"rating": "avg_rating"})
print(f"Generated average ratings for {len(avg_ratings)} anime")

# merging the data 
anime_data = anime.merge(avg_ratings, on="anime_id", how="left")
print(f"Merged data has {len(anime_data)} rows")

# Handle NaN and empty values in the genre column
anime_data['genre'] = anime_data['genre'].fillna('').astype(str)

# TF-IDF Vectorization
Tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = Tfidf.fit_transform(anime_data['genre']) 

# cosine similarity 
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation function
def recommend_anime(title):
    # Normalize the title to handle potential HTML encoding
    title = html.unescape(title)
    
    # Find matching titles (case insensitive)
    matches = anime_data[anime_data['name'].str.lower() == title.lower()]
    
    if len(matches) == 0:
        print(f"Anime '{title}' not found in database. Available titles include: {anime_data['name'].iloc[:5].tolist()}...")
        return ["Anime not found. Please check the title."]
    
    idx = matches.index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:16]  # Top 15 recommendations

    # Get recommendations, but don't filter by rating since we've already filtered the bad ones
    recommended_anime = []
    for i in scores:
        anime_idx = i[0]
        recommended_anime.append({
            'name': anime_data.iloc[anime_idx]['name'],
            'genre': anime_data.iloc[anime_idx]['genre'],
            'similarity_score': round(i[1], 3)  # Round for readability
        })
    
    # Print detailed info for debugging
    print("\nDetailed recommendations:")
    for i, rec in enumerate(recommended_anime[:5]):  # Show first 5 with details
        print(f"{i+1}. {rec['name']} - Genre: {rec['genre']} - Similarity: {rec['similarity_score']}")
    
    # Return just the anime names
    return [anime['name'] for anime in recommended_anime[:15]]

