import streamlit as st
from recommender import recommend, movies

st.title("🎬 Movie Recommendation System")

st.write("Select a movie to get recommendations")

movie_list = movies['title'].values
movie_name = st.selectbox("Choose Movie", movie_list)

if st.button("Recommend"):
    results = recommend(movie_name)

    st.subheader("Top Recommended Movies:")

    for r in results:
        st.write("👉", r)