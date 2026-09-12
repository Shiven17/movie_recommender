# Movie Finder

A content based movie recommender over 4,803 films, with a web app that explains every recommendation it makes.

**Live app:** https://movierecommender101.streamlit.app/

## What it does

Pick a film and get others assembled from similar parts. Each recommendation shows its rating, runtime, genres, plot and a match score, plus the actual overlap that put it there: shared cast, director, genres or themes. Results can be filtered by rating, vote count and release year.

## How it works

Every film is reduced to one line of text made from its keywords, cast, genres and director. CountVectorizer turns each line into word counts over a vocabulary of 15,787 words, and cosine similarity scores all 4,803 films against each other.

The full score table is 185 MB, far too large to ship or rebuild on each page load, so build_model.py keeps only each film's best 40 matches. That is a 477 KB file, and the deployed app reads it instead of computing anything.

## Does it work?

evaluate.py checks 28 pairs of films whose obvious partner is known, such as The Dark Knight and Batman Begins, in both directions for 56 checks:

| Version | Partner ranked first | Partner in top 10 | Average place |
| --- | --- | --- | --- |
| The tutorial's version | 31/56 | 56/56 | 1.73 |
| This version | 34/56 | 56/56 | 1.59 |
| This version with TfidfVectorizer | 33/56 | 54/56 | 1.93 |

## What I changed from the tutorial it started as

- **Director names are squashed into one word.** Splitting "James Cameron" into two words meant Avatar matched Jason X partly because James Isaac directed it. Same for the two genres written as two words, so sharing Science Fiction counted twice.
- **Films described in fewer than five words are never recommended.** The Dark Knight used to return Amidst the Devil's Wings third, whose entire description is "Drama Action Crime". Short descriptions score high against anything.
- **TfidfVectorizer was tried and rejected.** Weighting rare words higher sounds right, but the rarest words here are actors' names, so Toy Story filled up with films sharing one voice actor and Mission: Impossible lost its own sequel from the top 10. The table above is why it is not in the final version.
- **Every recommendation explains itself.** The cast column runs five names together as plain text with no separator, so shared actors are recovered by finding the longest stretches of words appearing in the same order in both films. A two word minimum stops a shared first name counting as a shared actor.
- **Forgiving search in the command line version**, covering capitals, partial titles, typos and the three titles that appear twice.

## Running it yourself

    pip install -r requirements.txt
    python build_model.py
    streamlit run app.py

build_model.py writes model_movies.csv and model_neighbours.npz. recommender.py is the same recommender in the command line, with no Streamlit needed.

Posters are optional. Without a TMDB key each film gets a tile coloured by its genre. To turn them on, put your key in .streamlit/secrets.toml:

    TMDB_API_KEY = "your_key_here"

## Files

| File | What it is |
| --- | --- |
| app.py | The Streamlit web app |
| build_model.py | Computes the matches once and saves them |
| recommender.py | Command line version |
| evaluate.py | The franchise pair test |
| movie_dataset.csv | The TMDB 5000 dataset |

## Credit

Data from the TMDB 5000 Movie Dataset. Posters from TMDB. This product uses the TMDB API but is not endorsed or certified by TMDB. The project started from Code Heroku's "Building a Movie Recommendation Engine" tutorial and grew from there.