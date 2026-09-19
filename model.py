import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# 1. LOAD DATA
# =====================================================

df = pd.read_csv("animes.csv")

print("Anime dataset loaded!")
print("Total anime:", len(df))


# =====================================================
# 2. CLEAN DATA
# =====================================================

# Remove anime without a score
df = df.dropna(subset=["score"])

# Fill missing values
df["episodes"] = df["episodes"].fillna(
    df["episodes"].median()
)

df["members"] = df["members"].fillna(
    df["members"].median()
)

df["popularity"] = df["popularity"].fillna(
    df["popularity"].median()
)

df["genre"] = df["genre"].fillna("")

df["synopsis"] = df["synopsis"].fillna("")


# =====================================================
# 3. CREATE GENRE FEATURES
# =====================================================

df["genre_list"] = df["genre"].apply(
    lambda x: x.split(", ")
)

mlb = MultiLabelBinarizer()

genre_data = mlb.fit_transform(
    df["genre_list"]
)

genre_df = pd.DataFrame(
    genre_data,
    columns=mlb.classes_,
    index=df.index
)


# =====================================================
# 4. CREATE FEATURES FOR SCORE PREDICTION
# =====================================================

X = pd.concat(
    [
        df[
            [
                "episodes",
                "members",
                "popularity"
            ]
        ],
        genre_df
    ],
    axis=1
)

y = df["score"]


# =====================================================
# 5. SPLIT DATA
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =====================================================
# 6. CREATE MODEL
# =====================================================

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)


# =====================================================
# 7. TRAIN MODEL
# =====================================================

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

print("Training complete!")


# =====================================================
# 8. TEST MODEL
# =====================================================

predictions = model.predict(
    X_test
)

mae = mean_absolute_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(
    "Mean Absolute Error:",
    round(mae, 3)
)

print(
    "R² Score:",
    round(r2, 3)
)

print(
    "R² Percentage:",
    round(r2 * 100, 2),
    "%"
)


# =====================================================
# 9. SCORE PREDICTION FUNCTION
# =====================================================

def predict_score(
    episodes,
    members,
    popularity,
    genres
):

    # Create empty feature row
    new_data = pd.DataFrame(
        0,
        index=[0],
        columns=X.columns
    )

    # Numeric values
    new_data["episodes"] = episodes
    new_data["members"] = members
    new_data["popularity"] = popularity

    # Add genres
    for genre in genres:

        if genre in new_data.columns:
            new_data[genre] = 1

    score = model.predict(
        new_data
    )[0]

    # Keep score between 0 and 10
    score = max(
        0,
        min(10, score)
    )

    return round(score, 2)


# =====================================================
# 10. ANIME RECOMMENDATION SYSTEM
# =====================================================

# Combine genre and synopsis
df["similarity_text"] = (
    df["genre"] + " " +
    df["synopsis"]
)

# Convert text into numbers
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000
)

tfidf_matrix = vectorizer.fit_transform(
    df["similarity_text"]
)

# Calculate similarity
similarity = cosine_similarity(
    tfidf_matrix
)


# =====================================================
# 11. RECOMMEND SIMILAR ANIME
# =====================================================

def recommend_anime(
    anime_name,
    number=5
):

    matches = df[
        df["title"].str.lower()
        == anime_name.lower()
    ]

    if matches.empty:

        print(
            "\nAnime not found."
        )

        return

    index = matches.index[0]

    scores = list(
        enumerate(
            similarity[index]
        )
    )

    # Sort by similarity
    scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    print(
        "\nAnime similar to:",
        df.loc[index, "title"]
    )

    print(
        "=============================="
    )

    count = 0

    for anime_index, score in scores:

        # Don't recommend itself
        if anime_index == index:
            continue

        title = df.loc[
            anime_index,
            "title"
        ]

        anime_score = df.loc[
            anime_index,
            "score"
        ]

        print(
            f"{count + 1}. {title}"
        )

        print(
            f"   Score: {anime_score}"
        )

        print(
            f"   Similarity: {score:.2f}"
        )

        print()

        count += 1

        if count == number:
            break


# =====================================================
# 12. CUSTOM SCORE PREDICTION
# =====================================================

print("\n")
print("==============================")
print("CUSTOM SCORE PREDICTION")
print("==============================")

episodes = float(
    input(
        "Number of episodes: "
    )
)

members = float(
    input(
        "Number of members: "
    )
)

popularity = float(
    input(
        "Popularity rank: "
    )
)

genre_input = input(
    "Genres separated by comma: "
)

genres = [
    x.strip()
    for x in genre_input.split(",")
]

predicted_score = predict_score(
    episodes,
    members,
    popularity,
    genres
)

print(
    "\nPredicted Anime Score:",
    predicted_score,
    "/ 10"
)


# =====================================================
# 13. CUSTOM ANIME RECOMMENDATION
# =====================================================

print("\n")
print("==============================")
print("ANIME RECOMMENDATION")
print("==============================")

anime_name = input(
    "Enter an anime you like: "
)

recommend_anime(
    anime_name,
    5
)