import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import html
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

def load_data():
    # Loading the data
    anime = pd.read_csv('anime.csv')
    rating = pd.read_csv('rating.csv')
    
    anime['name'] = anime['name'].apply(lambda x: html.unescape(str(x)) if pd.notnull(x) else x)
    
    # Filter out -1 ratings (which indicate missing ratings)
    valid_ratings = rating[rating['rating'] != -1]
    
    # Calculate average ratings using only valid ratings
    avg_ratings = valid_ratings.groupby("anime_id")["rating"].mean().reset_index().rename(columns={"rating": "avg_rating"})
    
    anime_data = anime.merge(avg_ratings, on="anime_id", how="left")
    
    anime_data['genre'] = anime_data['genre'].fillna('').astype(str)
    
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(anime_data['genre'])
    
    # Cosine similarity 
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return anime_data, cosine_sim

anime_data, cosine_sim = load_data()

# Get list of all anime titles for dropdown
def get_all_titles():
    return sorted(anime_data['name'].tolist())


# Recommendation function
def recommend_anime(title, num_recommendations=10):
    # Normalize the title to handle potential HTML encoding
    title = html.unescape(title)
    
    # Find matching titles (case insensitive)
    matches = anime_data[anime_data['name'].str.lower() == title.lower()]
    
    if len(matches) == 0:
        return []
    
    idx = matches.index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    
    # Get recommendations
    recommended_anime = []
    for i in scores:
        anime_idx = i[0]
        recommended_anime.append({
            'name': anime_data.iloc[anime_idx]['name'],
            'genre': anime_data.iloc[anime_idx]['genre'],
            'type': anime_data.iloc[anime_idx]['type'] if 'type' in anime_data.columns else 'N/A',
            'rating': round(float(anime_data.iloc[anime_idx]['rating']), 2) if 'rating' in anime_data.columns else 'N/A',
            'episodes': anime_data.iloc[anime_idx]['episodes'] if 'episodes' in anime_data.columns else 'N/A',
            'similarity': round(float(i[1]), 3) * 100  # Convert to percentage
        })
    
    return recommended_anime

@app.route('/')
def index():
    titles = get_all_titles()
    return render_template('index.html', titles=titles)

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    data = request.form
    title = data.get('anime_title')
    
    if not title:
        return jsonify({'error': 'No title provided'})
    
    try:
        recommendations = recommend_anime(title)
        return jsonify({'recommendations': recommendations})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/search', methods=['GET'])
def search_anime():
    query = request.args.get('q', '').lower()
    if not query:
        return jsonify([])
    
    # Find matching titles
    matches = [title for title in get_all_titles() if query in title.lower()]
    return jsonify(matches[:10])  # Return top 10 matches

if __name__ == '__main__':
    app.run(debug=True)