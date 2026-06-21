import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(__file__)
movies_path = os.path.join(BASE_DIR, "movies.csv")

if not os.path.exists(movies_path):
    raise FileNotFoundError("movies.csv missing in repo")

movies = pd.read_csv(movies_path)

# safe column check
if "title" not in movies.columns or "genres" not in movies.columns:
    raise ValueError("CSV must contain 'title' and 'genres' columns")

movies["genres"] = movies["genres"].fillna("")
movies["genres"] = movies["genres"].astype(str).str.replace("|", " ")

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres"])

cosine_sim = cosine_similarity(tfidf_matrix)

indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

def recommend(title, top_n=5):
    if title not in indices:
        return ["Movie not found"]

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    movie_indices = [i[0] for i in sim_scores[1:top_n+1]]

    return movies["title"].iloc[movie_indices].tolist()
