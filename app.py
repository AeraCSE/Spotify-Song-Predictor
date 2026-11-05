
import os
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Spotify | Hit Predictor",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

GREEN = "#1DB954"
BG = "#101010"
CARD = "#181818"
MUTED = "#B3B3B3"

# ---------------- CUSTOM CSS ----------------
st.markdown(f"""
<style>
.stApp {{
    background-color: {BG};
    color: #FFFFFF;
}}
[data-testid="stSidebar"] {{
    background-color: #080808;
    border-right: 1px solid #282828;
}}
.block-container {{
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}}
h1, h2, h3 {{
    color: #FFFFFF !important;
}}
p, label, .stCaption {{
    color: {MUTED};
}}
.eyebrow {{
    color: {GREEN};
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 3px;
}}
.hero {{
    background: linear-gradient(115deg, #202020, #121212);
    border: 1px solid #303030;
    border-radius: 16px;
    padding: 30px;
    margin: 12px 0 25px 0;
}}
.hero h1 {{
    font-size: 38px;
    margin: 8px 0;
}}
.hero p {{
    font-size: 15px;
    margin-bottom: 0;
}}
.metric-card {{
    background: {CARD};
    border: 1px solid #292929;
    border-radius: 12px;
    padding: 20px;
    min-height: 105px;
}}
.metric-label {{
    color: {MUTED};
    font-size: 13px;
}}
.metric-value {{
    color: {GREEN};
    font-size: 28px;
    font-weight: 700;
    margin-top: 8px;
}}
.section-label {{
    color: {GREEN};
    font-size: 12px;
    letter-spacing: 2px;
    font-weight: 700;
}}
div.stButton > button {{
    background-color: {GREEN};
    color: #000000;
    border: none;
    border-radius: 24px;
    font-weight: 700;
    min-height: 44px;
}}
div.stButton > button:hover {{
    background-color: #1ed760;
    color: #000000;
    border: none;
}}
div[data-testid="stMetric"] {{
    background: {CARD};
    border: 1px solid #292929;
    padding: 15px;
    border-radius: 12px;
}}
hr {{
    border-color: #303030;
}}
</style>
""", unsafe_allow_html=True)


# ---------------- DATA LOADING ----------------
@st.cache_data
def load_dataset():
    if not os.path.exists("dataset.csv"):
        return None
    data = pd.read_csv("dataset.csv")
    data = data.loc[:, ~data.columns.str.contains("^Unnamed")]
    return data


# ---------------- HEADER ----------------
st.markdown('<div class="eyebrow">MUSIC • DATA • MACHINE LEARNING</div>',
            unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>Spotify Song Hit Predictor</h1>
    <p>Explore music characteristics, understand popularity patterns,
    and estimate whether a track belongs to the high-popularity category.</p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "PROJECT NOTE  /  A hit is defined here as a dataset popularity score of 70 or above. "
    "This is a classification demo, not a guarantee of future success."
)

df = load_dataset()

if df is None:
    st.error("dataset.csv was not found. Place it in the project folder.")
    st.stop()


# ---------------- DATA OVERVIEW ----------------
st.markdown('<div class="section-label">01 / DATA OVERVIEW</div>',
            unsafe_allow_html=True)
st.subheader("Know the music")

if "popularity" not in df.columns:
    st.error("The dataset does not contain the required popularity column.")
    st.stop()

total_tracks = len(df)
avg_popularity = df["popularity"].mean()
unique_tracks = (
    df["track_id"].nunique()
    if "track_id" in df.columns else total_tracks
)
high_pop_count = int((df["popularity"] >= 70).sum())

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Dataset rows</div>'
        f'<div class="metric-value">{total_tracks:,}</div></div>',
        unsafe_allow_html=True
    )
with c2:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Unique tracks</div>'
        f'<div class="metric-value">{unique_tracks:,}</div></div>',
        unsafe_allow_html=True
    )
with c3:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Average popularity</div>'
        f'<div class="metric-value">{avg_popularity:.1f}</div></div>',
        unsafe_allow_html=True
    )
with c4:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Popularity ≥ 70</div>'
        f'<div class="metric-value">{high_pop_count:,}</div></div>',
        unsafe_allow_html=True
    )

st.write("")

tab1, tab2, tab3 = st.tabs([
    "Popularity distribution",
    "Audio feature analysis",
    "Model comparison"
])

with tab1:
    st.subheader("How popular are the tracks?")
    popularity_counts = pd.cut(
        df["popularity"].dropna(),
        bins=[-1, 19, 39, 59, 79, 100],
        labels=["0–19", "20–39", "40–59", "60–79", "80–100"]
    ).value_counts(sort=False)
    st.bar_chart(popularity_counts, color=GREEN)
    st.caption("Popularity is grouped into ranges to show the dataset distribution.")




with tab3:
    st.subheader("Compare trained classifiers")
    if os.path.exists("model_results.csv"):
        results = pd.read_csv("model_results.csv")
        st.dataframe(
            results.style.format({
                col: "{:.3f}"
                for col in [
                    "Accuracy", "Precision", "Recall", "F1-score"
                ] if col in results.columns
            }),
            use_container_width=True,
            hide_index=True
        )
        st.caption(
            "Because high-popularity tracks are a minority class, "
            "accuracy alone can be misleading. Compare precision, recall and F1."
        )
    else:
        st.info("Run train_model.py to generate model_results.csv.")


# ---------------- PREDICTION ----------------
st.markdown("---")
st.markdown('<div class="section-label">02 / TRY THE PREDICTOR</div>',
            unsafe_allow_html=True)
st.subheader("Explore a track")

st.write(
    "Adjust the audio characteristics below and use a saved classifier "
    "to estimate its popularity category."
)

model_paths = {
    "Logistic Regression": "models/logistic_regression.joblib",
    "Random Forest": "models/random_forest.joblib",
}
available_models = {
    name: path for name, path in model_paths.items()
    if os.path.exists(path)
}

if not available_models:
    st.error("No trained models found. Run train_model.py first.")
    st.stop()


with st.sidebar:
    st.markdown("## CREATE YOUR TRACK")
    st.caption("Tell us about the song you are planning to make.")

    genre = st.selectbox(
        "1. Which genre are you creating?",
        ["Pop", "Hip-Hop", "Rock", "Electronic", "Acoustic", "Lo-fi"]
    )

    mood = st.selectbox(
        "2. What mood should your song have?",
        ["Happy", "Sad", "Romantic", "Relaxing", "Energetic"]
    )

    energy_choice = st.selectbox(
        "3. How energetic should it feel?",
        ["Low", "Medium", "High"]
    )

    vocals = st.selectbox(
        "4. What type of vocals?",
        ["Mostly singing", "Mostly rap", "Instrumental", "Mixed"]
    )

    pace = st.selectbox(
        "5. What should the tempo feel like?",
        ["Slow", "Moderate", "Fast"]
    )

    duration_choice = st.selectbox(
        "6. Approximate song duration?",
        ["Under 2 minutes", "2–3 minutes", "3–4 minutes", "Over 4 minutes"]
    )

# Convert creative choices into approximate model features

genre_features = {
    "Pop":        (0.75, 0.10, 0.05, 0.60),
    "Hip-Hop":    (0.80, 0.05, 0.15, 0.55),
    "Rock":       (0.55, 0.05, 0.00, 0.65),
    "Electronic": (0.80, 0.00, 0.20, 0.60),
    "Acoustic":   (0.50, 0.70, 0.05, 0.55),
    "Lo-fi":      (0.55, 0.65, 0.20, 0.40)
}

danceability, acousticness, instrumentalness, valence = genre_features[genre]

energy_map = {
    "Low": (0.30, -12.0),
    "Medium": (0.60, -7.0),
    "High": (0.90, -4.0)
}
energy, loudness = energy_map[energy_choice]

mood_map = {
    "Happy": 0.85,
    "Sad": 0.15,
    "Romantic": 0.65,
    "Relaxing": 0.50,
    "Energetic": 0.80
}
valence = mood_map[mood]

vocal_map = {
    "Mostly singing": (0.04, 0.00),
    "Mostly rap": (0.75, 0.00),
    "Instrumental": (0.02, 0.90),
    "Mixed": (0.20, 0.05)
}
speechiness, instrumentalness = vocal_map[vocals]

tempo_map = {
    "Slow": 75,
    "Moderate": 110,
    "Fast": 145
}
tempo = tempo_map[pace]

duration_map = {
    "Under 2 minutes": 100000,
    "2–3 minutes": 150000,
    "3–4 minutes": 210000,
    "Over 4 minutes": 270000
}
duration_ms = duration_map[duration_choice]

liveness = 0.15
# ---------------- DYNAMIC AUDIO FEATURE GRAPH ----------------

with tab2:
    st.subheader("Explore music characteristics")

    friendly_features = {
        "Dance-friendly feel": "danceability",
        "Energy level": "energy",
        "Mood and positivity": "valence",
        "Acoustic sound": "acousticness",
        "Song speed": "tempo",
        "Sound level": "loudness"
    }

    selected_label = st.selectbox(
        "What would you like to explore?",
        list(friendly_features.keys())
    )

    selected_feature = friendly_features[selected_label]

    # Actual Spotify dataset points
    chart_data = df[[selected_feature, "popularity"]].dropna()
    chart_data = chart_data.sample(
        n=min(1500, len(chart_data)),
        random_state=42
    )

    # Current selected track value
    input_values = {
        "danceability": danceability,
        "energy": energy,
        "loudness": loudness,
        "speechiness": speechiness,
        "acousticness": acousticness,
        "instrumentalness": instrumentalness,
        "liveness": liveness,
        "valence": valence,
        "tempo": tempo,
        "duration_ms": duration_ms
    }

    selected_value = input_values[selected_feature]
    st.write("LIVE SELECTED VALUE:", selected_value)
    st.write("CURRENT ENERGY:", energy)

    # Create scatter graph
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(
        chart_data[selected_feature],
        chart_data["popularity"],
        alpha=0.35,
        s=12,
        color="#1DB954",
        label="Spotify dataset"
    )

    # Highlight selected track
    ax.axvline(
        x=selected_value,
        color="white",
        linestyle="--",
        linewidth=2,
        label=f"Your track: {selected_value:.2f}"
    )

    ax.set_xlabel(selected_label)
    ax.set_ylabel("Popularity")
    ax.set_title(f"{selected_label} vs Popularity")
    ax.legend()
    ax.grid(alpha=0.15)

    fig.patch.set_facecolor("#101010")
    ax.set_facecolor("#181818")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "The white dashed line represents your selected track's "
        "audio feature value. Green points represent actual Spotify tracks."
    )
st.caption("BPM = beats per minute")


left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.markdown("### Audio profile")
    st.write("Your selected track characteristics")

    p1, p2, p3 = st.columns(3)
    p1.metric("Danceability", f"{danceability:.2f}")
    p2.metric("Energy", f"{energy:.2f}")
    p3.metric("Tempo", f"{tempo} BPM")

    st.progress(danceability, text="Danceability")
    st.progress(energy, text="Energy")
    st.progress(valence, text="Valence")

with right:
    st.markdown("### Prediction settings")
    chosen_model = st.selectbox(
        "Classification model",
        list(available_models.keys())
    )

    st.write(
        "The classifier uses ten audio features to estimate "
        "the track's category."
    )

    predict_clicked = st.button(
        "ANALYSE TRACK",
        use_container_width=True
    )
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if predict_clicked:
    try:
        bundle = joblib.load(available_models[chosen_model])
        model = bundle["model"]
        features = bundle["features"]

        input_values = {
            "danceability": danceability,
            "energy": energy,
            "loudness": loudness,
            "speechiness": speechiness,
            "acousticness": acousticness,
            "instrumentalness": instrumentalness,
            "liveness": liveness,
            "valence": valence,
            "tempo": tempo,
            "duration_ms": duration_ms,
        }

        input_data = pd.DataFrame(
            [[input_values[f] for f in features]],
            columns=features
        )

        prediction = int(model.predict(input_data)[0])
        st.session_state.prediction_result = {
            "prediction": prediction,
            "probability": (
                float(model.predict_proba(input_data)[0][list(model.classes_).index(1)])
                if hasattr(model, "predict_proba") else None
             ),
            "inputs": input_values.copy()
        }

        st.markdown("---")
        st.markdown("### Analysis result")

        if prediction == 1:
            st.success("HIGH-POPULARITY CATEGORY")
        else:
            st.info("OTHER POPULARITY CATEGORY")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = list(model.classes_)
            positive_index = classes.index(1)
            probability = float(probabilities[positive_index])

            st.metric(
                "Estimated model probability for high popularity",
                f"{probability:.1%}"
            )

            st.progress(probability)

        st.caption(
            "This is a model estimate based on learned patterns, "
            "not a prediction of future streams or guaranteed success. "
            "Probability estimates may be poorly calibrated."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")


# ---------------- FOOTER ----------------
st.markdown("---")
st.caption(
    "SPOTIFY SONG HIT PREDICTOR  •  Predictive Analytics Project"
)
