"""A web page for the movie recommender. Run it with: streamlit run app.py"""

import re

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Movie Finder", page_icon="🎬", layout="wide")

TEXT_COLUMNS = [
    "title", "year", "genres", "director", "cast", "keywords",
    "overview", "tagline", "genres_token", "director_token", "original_language",
]


@st.cache_data
def load_model():
    """Read the files build_model.py saved. Cached, so this happens once."""
    movies = pd.read_csv("model_movies.csv", dtype={"year": "string"})
    for col in TEXT_COLUMNS:
        movies[col] = movies[col].fillna("")
    for col in ["vote_average", "vote_count", "runtime", "popularity"]:
        movies[col] = movies[col].fillna(0)
    saved = np.load("model_neighbours.npz")
    return movies, saved["indices"], saved["scores"]


try:
    movies, neighbours, neighbour_scores = load_model()
except FileNotFoundError:
    st.error("The saved model is missing. Run `python build_model.py` first, then reload this page.")
    st.stop()


@st.cache_data(show_spinner=False)
def poster_url(tmdb_id):
    """Look up a poster on TMDB. Returns None when there is no key or no poster."""
    try:
        key = st.secrets["TMDB_API_KEY"]
    except Exception:
        return None
    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}",
            params={"api_key": key},
            timeout=5,
        )
        path = response.json().get("poster_path")
    except Exception:
        return None
    return f"https://image.tmdb.org/t/p/w342{path}" if path else None


def show_poster(row, width):
    url = poster_url(row["id"])
    if url:
        st.image(url, width=width)
    else:
        st.markdown(
            f"<div style='width:{width}px;height:{int(width * 1.5)}px;border-radius:8px;"
            "background:#262730;display:flex;align-items:center;justify-content:center;"
            "padding:10px;text-align:center;font-size:13px;color:#9aa0a6;'>"
            f"{row['title']}</div>",
            unsafe_allow_html=True,
        )


def unsquash(token):
    """ScienceFiction becomes Science Fiction, TVMovie becomes TV Movie."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", token)


def words(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def common_runs(text_a, text_b, limit=3, least_words=1):
    """Bits of text_a that also appear, in the same order, in text_b.

    Both movies list cast members as plain text, so the longest stretches they
    share are the actors (or keyword phrases) they have in common.
    """
    a, b = str(text_a).split(), " " + " ".join(words(text_b)) + " "
    runs, start = [], 0
    while start < len(a):
        end = start
        while end < len(a) and f" {' '.join(words(' '.join(a[start:end + 1])))} " in b:
            end += 1
        if end > start:
            runs.append(" ".join(a[start:end]))
            start = end
        else:
            start += 1
    unique = list(dict.fromkeys(run for run in runs if len(run.split()) >= least_words and len(run) > 2))
    return unique[:limit]


def explain(chosen, other):
    """A short list of what the two movies have in common."""
    reasons = []

    shared_genres = set(movies["genres_token"][chosen].split()) & set(movies["genres_token"][other].split())
    if shared_genres:
        reasons.append("**Genres:** " + ", ".join(unsquash(g) for g in sorted(shared_genres)))

    if movies["director_token"][chosen] and movies["director_token"][chosen] == movies["director_token"][other]:
        reasons.append(f"**Director:** {movies['director'][chosen]}")

    # Two words at least, so sharing only a first name like "Jon" does not count as sharing an actor
    shared_cast = common_runs(movies["cast"][chosen], movies["cast"][other], least_words=2)
    if shared_cast:
        reasons.append("**Cast:** " + ", ".join(shared_cast))

    shared_keywords = common_runs(movies["keywords"][chosen], movies["keywords"][other])
    if shared_keywords:
        reasons.append("**Themes:** " + ", ".join(shared_keywords))

    return reasons


def facts(row):
    """The one line of details under a title: rating, votes, runtime, year, genres."""
    bits = []
    if row["vote_count"] > 0:
        bits.append(f"⭐ {row['vote_average']:.1f}/10 ({int(row['vote_count']):,} votes)")
    if row["runtime"] > 0:
        bits.append(f"{int(row['runtime'])} min")
    if row["year"]:
        bits.append(row["year"])
    if row["genres_token"]:
        bits.append(", ".join(unsquash(g) for g in row["genres_token"].split()))
    return " · ".join(bits)


st.title("🎬 Movie Finder")
st.caption(
    "Pick a film and get others built from similar parts. "
    "Every recommendation says why it turned up."
)

with st.sidebar:
    st.header("Filters")
    how_many = st.slider("How many recommendations", 3, 20, 8)
    min_rating = st.slider("Minimum rating", 0.0, 9.0, 0.0, step=0.5)
    min_votes = st.select_slider("Minimum number of votes", [0, 50, 200, 1000, 5000], value=0)
    year_from, year_to = st.select_slider(
        "Released between",
        options=[str(y) for y in range(1916, 2018)],
        value=("1916", "2017"),
    )
    st.divider()
    st.caption(
        "Built from each film's cast, director, genres and keywords. "
        "No ratings or viewing history are used, so an unpopular film can still "
        "be a close match."
    )

# The dropdown searches as you type, so titles never have to be typed exactly
order = movies.sort_values("popularity", ascending=False).index
chosen = st.selectbox(
    "Pick a movie you like",
    options=order,
    format_func=lambda i: f"{movies['title'][i]} ({movies['year'][i]})" if movies["year"][i] else movies["title"][i],
)

chosen_row = movies.loc[chosen]
left, right = st.columns([1, 4])
with left:
    show_poster(chosen_row, 170)
with right:
    st.subheader(f"{chosen_row['title']} ({chosen_row['year']})" if chosen_row["year"] else chosen_row["title"])
    if chosen_row["tagline"]:
        st.markdown(f"*{chosen_row['tagline']}*")
    st.markdown(facts(chosen_row))
    if chosen_row["director"]:
        st.markdown(f"**Directed by** {chosen_row['director']}")
    if chosen_row["overview"]:
        st.write(chosen_row["overview"])
    st.link_button("View on TMDB", f"https://www.themoviedb.org/movie/{int(chosen_row['id'])}")

st.divider()

# Walk the saved matches, best first, keeping the ones that pass the filters
picks = []
for other, score in zip(neighbours[chosen], neighbour_scores[chosen]):
    if other < 0 or len(picks) == how_many:
        break
    row = movies.loc[other]
    if row["vote_average"] < min_rating or row["vote_count"] < min_votes:
        continue
    if row["year"] and not (year_from <= row["year"] <= year_to):
        continue
    picks.append((int(other), float(score)))

if not picks:
    if neighbours[chosen][0] < 0:
        st.warning(
            f"The dataset barely describes {chosen_row['title']}, so there is nothing to compare it with."
        )
    else:
        st.warning("Nothing matched those filters. Try loosening them in the sidebar.")
else:
    st.subheader(f"Because you liked {chosen_row['title']}")
    if len(picks) < how_many:
        st.caption(f"Only {len(picks)} of the saved matches passed your filters.")

for other, score in picks:
    row = movies.loc[other]
    with st.container(border=True):
        left, right = st.columns([1, 5])
        with left:
            show_poster(row, 110)
        with right:
            title = f"{row['title']} ({row['year']})" if row["year"] else row["title"]
            st.markdown(f"#### [{title}](https://www.themoviedb.org/movie/{int(row['id'])})")
            st.markdown(facts(row))
            st.progress(min(score, 1.0), text=f"{score:.0%} match")
            for reason in explain(chosen, other):
                st.markdown(reason)
            if row["overview"]:
                with st.expander("What it's about"):
                    st.write(row["overview"])