import streamlit as st
from model import df, mlb, predict_score, recommend_anime

# Page settings
st.set_page_config(
    page_title="Anime AI",
    page_icon="🎌",
    layout="wide"
)

# Title
st.title("🎌 Anime AI")
st.write("Predict anime scores and find anime similar to your favorite anime.")

# Create tabs
tab1, tab2 = st.tabs(["🎯 Score Predictor", "🎬 Anime Recommendations"])


# ==========================================
# SCORE PREDICTOR
# ==========================================

with tab1:
    st.header("🎯 Predict Anime Score")

    col1, col2 = st.columns(2)

    with col1:
        episodes = st.number_input(
            "Number of Episodes",
            min_value=1,
            value=24
        )

        members = st.number_input(
            "Number of Members",
            min_value=0,
            value=100000,
            step=1000
        )

    with col2:
        popularity = st.number_input(
            "Popularity Rank",
            min_value=1,
            value=1000
        )

        genres = st.multiselect(
            "Select Genres",
            list(mlb.classes_)
        )

    if st.button("Predict Score", type="primary"):

        score = predict_score(
            episodes,
            members,
            popularity,
            genres
        )

        st.success(
            f"⭐ Predicted Anime Score: **{score:.2f} / 10**"
        )


# ==========================================
# RECOMMENDATION SYSTEM
# ==========================================

with tab2:
    st.header("🎬 Find Similar Anime")

    anime_name = st.selectbox(
        "Select an Anime",
        df["title"].tolist()
    )

    number = st.slider(
        "Number of Recommendations",
        min_value=1,
        max_value=10,
        value=5
    )

    if st.button("Find Similar Anime", type="primary"):

        results = recommend_anime(
            anime_name,
            number
        )

        if results:

            st.subheader(
                f"Anime similar to **{anime_name}**"
            )

            for i, result in enumerate(results, 1):

                title = result[0]
                score = result[1]
                similarity = result[2]

                st.write(f"### {i}. {title}")

                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"⭐ Score: **{score:.2f} / 10**")

                with col2:
                    st.write(
                        f"🔗 Similarity: **{similarity:.2f}**"
                    )

                st.divider()

        else:
            st.warning("Anime not found.")


# ==========================================
# FOOTER
# ==========================================

st.caption(
    f"🎌 Anime AI | Dataset contains {len(df):,} anime"
)