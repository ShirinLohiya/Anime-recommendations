# Anime-recommendations
A content-based anime recommendation system that suggests similar anime based on genre similarity using TF-IDF Vectorization and Cosine Similarity.

Features

Content-Based Filtering using anime genres

TF-IDF Vectorization to process genres

Cosine Similarity to measure anime similarity

Handles HTML-encoded anime titles

Filters out invalid ratings (-1) before computing average ratings

Technologies Used

Python

Pandas for data manipulation

Scikit-learn for TF-IDF and cosine similarity

HTML module for encoding fixes
# Machine Learning Concepts Used
The system applies Content-Based Filtering (CBF) using TF-IDF Vectorization and cosine similarity to recommend anime based on genre similarity.
Text Processing using TF-IDF (Term Frequency-Inverse Document Frequency)

Converts the genre column into numerical vectors by measuring the importance of words in each genre description.
Removes common English words (stop words) to focus on meaningful terms.
Helps in representing genre-based similarity in a numerical format.


Cosine Similarity for Recommendations

Computes the similarity between anime based on their TF-IDF vectors.
Uses cosine similarity, which measures the cosine of the angle between two vectors.
If two anime have similar genres, their vectors will be closer (high cosine similarity score).
