import pandas as pd

df = pd.read_csv("movie_dataset.csv")

# The four columns the video uses to describe each movie
features = ["keywords", "cast", "genres", "director"]

# Replace blanks with empty text so the columns can be joined
for col in features:
    df[col] = df[col].fillna("")

# Join the four columns into one line of text per movie
df["combined_features"] = df["keywords"] + " " + df["cast"] + " " + df["genres"] + " " + df["director"]

avatar = df[df["title"] == "Avatar"].iloc[0]
print("Avatar's combined text:\n" + avatar["combined_features"])

empty = (df["combined_features"].str.strip() == "").sum()
print(f"\nMovies with no information at all: {empty}")