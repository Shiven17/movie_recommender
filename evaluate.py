"""Check the recommender against films whose obvious partner we already know.

If Batman Begins is not near the top for The Dark Knight, something is wrong.
Each pair is checked both ways round, so 28 pairs make 56 checks.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PAIRS = [
    ("The Dark Knight", "Batman Begins"),
    ("Toy Story", "Toy Story 2"),
    ("Harry Potter and the Chamber of Secrets", "Harry Potter and the Philosopher's Stone"),
    ("The Godfather", "The Godfather: Part II"),
    ("Iron Man", "Iron Man 2"),
    ("Shrek", "Shrek 2"),
    ("The Matrix", "The Matrix Reloaded"),
    ("Star Wars", "The Empire Strikes Back"),
    ("Alien", "Aliens"),
    ("Jurassic Park", "The Lost World: Jurassic Park"),
    ("The Lord of the Rings: The Fellowship of the Ring", "The Lord of the Rings: The Two Towers"),
    ("Back to the Future", "Back to the Future Part II"),
    ("The Hunger Games", "The Hunger Games: Catching Fire"),
    ("Pirates of the Caribbean: The Curse of the Black Pearl", "Pirates of the Caribbean: Dead Man's Chest"),
    ("X-Men", "X2"),
    ("Spider-Man", "Spider-Man 2"),
    ("Despicable Me", "Despicable Me 2"),
    ("Kung Fu Panda", "Kung Fu Panda 2"),
    ("The Bourne Identity", "The Bourne Supremacy"),
    ("The Terminator", "Terminator 2: Judgment Day"),
    ("Die Hard", "Die Hard 2"),
    ("Mission: Impossible", "Mission: Impossible II"),
    ("Men in Black", "Men in Black II"),
    ("Ice Age", "Ice Age: The Meltdown"),
    ("The Hangover", "The Hangover Part II"),
    ("Madagascar", "Madagascar: Escape 2 Africa"),
    ("Twilight", "The Twilight Saga: New Moon"),
    ("Sherlock Holmes", "Sherlock Holmes: A Game of Shadows"),
]

MIN_WORDS = 5

df = pd.read_csv("movie_dataset.csv")
for col in ["keywords", "cast", "genres", "director"]:
    df[col] = df[col].fillna("")

where = {title: position for position, title in enumerate(df["title"])}

plain = df["keywords"] + " " + df["cast"] + " " + df["genres"] + " " + df["director"]
cleaned = (
    df["keywords"] + " " + df["cast"] + " "
    + df["genres"].str.replace("Science Fiction", "ScienceFiction").str.replace("TV Movie", "TVMovie")
    + " " + df["director"].str.replace(" ", "")
)


def measure(name, vectorizer, text, drop_thin):
    similarity = cosine_similarity(vectorizer.fit_transform(text))
    np.fill_diagonal(similarity, -1)
    if drop_thin:
        thin = text.str.split().str.len().to_numpy() < MIN_WORDS
        similarity[:, thin] = -1

    ranks = []
    for first, second in PAIRS:
        for asked, wanted in ((first, second), (second, first)):
            row = similarity[where[asked]]
            ranks.append(int((row > row[where[wanted]]).sum() + 1))

    ranks = np.array(ranks)
    print(
        f"{name:34s} first {(ranks == 1).sum():3d}/{len(ranks)}"
        f"   top 10 {(ranks <= 10).sum():3d}/{len(ranks)}"
        f"   average place {ranks.mean():.2f}"
    )
    return ranks


print(f"Checking {len(PAIRS)} pairs of films, both ways round.\n")
measure("Step 3, the video's version", CountVectorizer(), plain, drop_thin=False)
measure("Step 5, what the app uses", CountVectorizer(), cleaned, drop_thin=True)
measure("Step 5 but with TfidfVectorizer", TfidfVectorizer(), cleaned, drop_thin=True)
print("\nFirst = the partner film was the number one recommendation.")