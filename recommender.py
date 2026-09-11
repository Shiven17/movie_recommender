import pandas as pd

df = pd.read_csv("movie_dataset.csv")
print(f"Loaded {len(df)} movies with {df.shape[1]} columns\n")

# The four columns the video uses to describe each movie
features = ["keywords", "cast", "genres", "director"]

avatar = df[df["title"] == "Avatar"].iloc[0]
print("What the recommender will see for Avatar:")
for col in features:
    print(f"  {col}: {avatar[col]}")

print("\nBlank values in those columns:")
print(df[features].isna().sum())