import difflib

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("movie_dataset.csv")

# Step 2: fill blanks, then join the four columns into one line of text per movie
features = ["keywords", "cast", "genres", "director"]
for col in features:
    df[col] = df[col].fillna("")
df["combined_features"] = df["keywords"] + " " + df["cast"] + " " + df["genres"] + " " + df["director"]

# Step 3: turn each movie's text into word counts, then score every pair of movies
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])
cosine_sim = cosine_similarity(count_matrix)

# Step 4: lowercase titles for searching, and release years to tell repeats apart
df["search_title"] = df["title"].str.lower()
df["year"] = df["release_date"].str[:4].fillna("")


def label(index):
    """A movie's title with its year, like Batman (1989)."""
    title, year = df["title"][index], df["year"][index]
    return f"{title} ({year})" if year else title


def find_movie(query):
    """Turn whatever was typed into one movie's position in the table, or None."""
    query = query.strip().lower()

    # 1. The exact title, ignoring capitals. Two hits means a repeated title like Batman.
    matches = df[df["search_title"] == query]

    # 2. Nothing? Try titles containing what was typed, so "harry potter" finds all of them.
    if matches.empty:
        matches = df[df["search_title"].str.contains(query, regex=False)]

    # 3. Still nothing? Try titles spelled almost the same, to catch typos like "avatr".
    if matches.empty:
        close = difflib.get_close_matches(query, df["search_title"], n=5, cutoff=0.7)
        matches = df[df["search_title"].isin(close)]

    # Most popular first, 8 at most
    candidates = matches.sort_values("popularity", ascending=False).index[:8].tolist()

    if not candidates:
        print("Couldn't find anything like that. Try another spelling?")
        return None
    if len(candidates) == 1:
        return candidates[0]

    print("Which one did you mean?")
    for number, index in enumerate(candidates, start=1):
        print(f"  {number}. {label(index)}")
    choice = input("Type its number (or press Enter to skip): ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(candidates):
        return candidates[int(choice) - 1]
    return None


def recommend(movie_index, n=10):
    if df["combined_features"][movie_index].strip() == "":
        print(f"The dataset has nothing about {label(movie_index)} to compare, so no recommendations.")
        return

    # Best scores first, leaving out the movie itself and anything with no words in common
    scores = sorted(enumerate(cosine_sim[movie_index]), key=lambda pair: pair[1], reverse=True)
    scores = [pair for pair in scores if pair[0] != movie_index and pair[1] > 0]

    print(f"\nBecause you liked {label(movie_index)}:")
    for index, score in scores[:n]:
        print(f"  {score:.2f}  {label(index)}")


print(f"Ready: {count_matrix.shape[0]} movies, {count_matrix.shape[1]} different words")

while True:
    query = input("\nMovie you like (press Enter to quit): ")
    if query.strip() == "":
        break
    movie_index = find_movie(query)
    # Avatar sits at position 0, and "if movie_index:" would treat 0 as false, hence "is not None"
    if movie_index is not None:
        recommend(movie_index)