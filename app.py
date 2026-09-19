import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Anime AI", page_icon="🎌", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("animes.csv")
    df = df.dropna(subset=["score"])
    for col in ["episodes", "members", "popularity"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())
    df["genre"] = df["genre"].fillna("")
    df["synopsis"] = df["synopsis"].fillna("")
    return df

df = load_data()

@st.cache_resource
def train_model(data):
    genres = data["genre"].apply(lambda x: x.split(", "))
    mlb = MultiLabelBinarizer()
    genre_data = mlb.fit_transform(genres)
    genre_df = pd.DataFrame(genre_data, columns=mlb.classes_, index=data.index)

    X = pd.concat([data[["episodes", "members", "popularity"]], genre_df], axis=1)
    y = data["score"]

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    return model, mlb, X.columns

model, mlb, feature_columns = train_model(df)

@st.cache_resource
def make_similarity_model(data):
    text = data["genre"] + " " + data["synopsis"]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
    matrix = vectorizer.fit_transform(text)
    return matrix

tfidf_matrix = make_similarity_model(df)

def predict_score(episodes, members, popularity, genres):
    new_data = pd.DataFrame(0, index=[0], columns=feature_columns)
    new_data["episodes"] = episodes
    new_data["members"] = members
    new_data["popularity"] = popularity

    for genre in genres:
        if genre in new_data.columns:
            new_data[genre] = 1

    return max(0, min(10, model.predict(new_data)[0]))

def recommend_anime(name, number):
    index_list = df.index[df["title"].str.lower() == name.lower()].tolist()

    if not index_list:
        return []

    index = index_list[0]
    similarities = cosine_similarity(tfidf_matrix[index], tfidf_matrix).flatten()
    similar_indexes = similarities.argsort()[::-1]

    results = []
    for i in similar_indexes:
        if i == index:
            continue
        results.append((df.loc[i, "title"], df.loc[i, "score"], similarities[i]))
        if len(results) == number:
            break
    return results

st.title("🎌 Anime AI")
st.write("Predict anime scores and find anime similar to one you like.")

tab1, tab2 = st.tabs(["🎯 Score Predictor", "🎬 Recommendations"])

with tab1:
    st.header("Predict Anime Score")

    col1, col2 = st.columns(2)

    with col1:
        episodes = st.number_input("Episodes", min_value=1, value=24)
        members = st.number_input("Members", min_value=0, value=100000, step=1000)

    with col2:
        popularity = st.number_input("Popularity Rank", min_value=1, value=1000)
        genres = st.multiselect("Genres", list(mlb.classes_))

    if st.button("Predict Score", type="primary"):
        score = predict_score(episodes, members, popularity, genres)
        st.success(f"Predicted Score: {score:.2f} / 10")

with tab2:
    st.header("Find Similar Anime")

    anime_name = st.selectbox("Choose an anime", df["title"].tolist())
    number = st.slider("Recommendations", 1, 10, 5)

    if st.button("Find Similar Anime", type="primary"):
        results = recommend_anime(anime_name, number)

        for i, (title, score, similarity) in enumerate(results, 1):
            st.write(f"### {i}. {title}")
            st.write(f"⭐ Score: {score:.2f}  |  Similarity: {similarity:.2f}")
            st.divider()

st.caption(f"Dataset: {len(df):,} anime")
