"""Work out the recommendations once and save them, so the web app starts instantly."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TOP_N = 40      # how many similar movies to remember for each film
MIN_WORDS = 5   # movies described in fewer words are never recommended

print("Reading the dataset...")
df = pd.read_csv("movie_dataset.csv")

features = ["keywords", "cast", "genres", "director"]
for col in features:
    df[col] = df[col].fillna("")

# Step 5 again: one word per director, one word per genre
df["director_token"] = df["director"].str.replace(" ", "")
df["genres_token"] = (
    df["genres"].str.replace("Science Fiction", "ScienceFiction").str.replace("TV Movie", "TVMovie")
)
df["combined_features"] = (
    df["keywords"] + " " + df["cast"] + " " + df["genres_token"] + " " + df["director_token"]
)

print("Scoring every pair of movies...")
count_matrix = CountVectorizer().fit_transform(df["combined_features"])
similarity = cosine_similarity(count_matrix)

enough_info = df["combined_features"].str.split().str.len().to_numpy() >= MIN_WORDS
np.fill_diagonal(similarity, -1)    # a movie is never its own recommendation
similarity[:, ~enough_info] = -1    # thinly described movies are never recommended

print(f"Keeping the best {TOP_N} matches for each movie...")
order = np.argsort(-similarity, axis=1)[:, :TOP_N]
scores = np.take_along_axis(similarity, order, axis=1)
order[scores <= 0] = -1   # empty slot, the app skips these
scores[scores <= 0] = 0

np.savez_compressed(
    "model_neighbours.npz",
    indices=order.astype(np.int16),
    scores=scores.astype(np.float32),
)

# Everything the app shows on a card, plus the cleaned text it uses to explain matches
df["year"] = df["release_date"].str[:4].fillna("")
columns = [
    "title", "year", "genres", "director", "cast", "keywords", "overview", "tagline",
    "vote_average", "vote_count", "runtime", "popularity", "id", "genres_token",
    "director_token", "original_language",
]
df[columns].to_csv("model_movies.csv", index=False)

print(f"Saved {len(df)} movies. Recommendable: {enough_info.sum()}.")