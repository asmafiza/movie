import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------------------------
# Load Data
# -------------------------
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

data = pd.merge(ratings, movies, on="movieId")

# ⚡ speed fix
data = data.sample(n=20000, random_state=42)

# -------------------------
# Content Based
# -------------------------
movies["genres"] = movies["genres"].fillna("")
movies["genres"] = movies["genres"].str.replace("|", " ", regex=False)

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres"])

cosine_sim = cosine_similarity(tfidf_matrix)

indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

# -------------------------
# KMeans
# -------------------------
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
movies["cluster"] = kmeans.fit_predict(tfidf_matrix)

# -------------------------
# Random Forest Only ML
# -------------------------
data["liked"] = (data["rating"] >= 4).astype(int)

X = data[["userId", "movieId"]]
y = data["liked"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

pred = rf.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, pred))

# -------------------------
# Recommendation Function
# -------------------------
def recommend(title, top_n=5):

    if title not in indices:
        return ["Movie not found"]

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    movie_indices = [i[0] for i in sim_scores[1:top_n+1]]

    return movies["title"].iloc[movie_indices].tolist()

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    movie = input("\nEnter Movie Name: ")

    print("\nRecommendations:\n")
    for m in recommend(movie):
        print(m)