import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("movie_dataset.csv")

# Step 2: fill blanks, then join the four columns into one line of text per movie
features = ["keywords", "cast", "genres", "director"]
for col in features:
    df[col] = df[col].fillna("")
df["combined_features"] = df["keywords"] + " " + df["cast"] + " " + df["genres"] + " " + df["director"]

# Step 3: turn each movie's line of text into word counts
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])

# Score every movie against every other movie (a 4803 by 4803 table)
cosine_sim = cosine_similarity(count_matrix)


def recommend(title, n=10):
    matches = df.index[df["title"] == title]
    if len(matches) == 0:
        print("Couldn't find that title. For now it has to match exactly, capitals included.")
        return
    movie_index = matches[0]

    # Pair each movie's position with its score, highest score first
    scores = sorted(enumerate(cosine_sim[movie_index]), key=lambda pair: pair[1], reverse=True)

    # Every movie is a perfect match with itself, so leave it out
    scores = [pair for pair in scores if pair[0] != movie_index]

    print(f"\nBecause you liked {title}:")
    for index, score in scores[:n]:
        print(f"  {score:.2f}  {df['title'][index]}")


print(f"Ready: {count_matrix.shape[0]} movies, {count_matrix.shape[1]} different words")

while True:
    title = input("\nMovie you like (press Enter to quit): ")
    if title == "":
        break
    recommend(title)
