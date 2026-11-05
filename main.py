
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.dummy import DummyClassifier

# Load and clean the dataset
df = pd.read_csv("dataset.csv")
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
df = df.drop_duplicates(subset=["track_id"])

features = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
]

df = df.dropna(subset=features + ["popularity"])
df["hit"] = (df["popularity"] >= 70).astype(int)

# Audio features only: popularity is NOT an input
X = df[features]
y = df["hit"]

# Keep the class proportions similar in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Baseline: always predict the majority class
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)

# Train the actual ML model
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(class_weight="balanced", max_iter=1000)
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\nSPOTIFY SONG HIT PREDICTOR")
print("=" * 40)

print("\nBASELINE F1:", round(
    f1_score(y_test, baseline_pred, zero_division=0), 3
))

print("\nMODEL RESULTS")
print("Accuracy :", round(accuracy_score(y_test, predictions), 3))
print("Precision:", round(precision_score(y_test, predictions, zero_division=0), 3))
print("Recall   :", round(recall_score(y_test, predictions, zero_division=0), 3))
print("F1-score :", round(f1_score(y_test, predictions, zero_division=0), 3))

print("\nClassification Report:")
print(classification_report(
    y_test, predictions,
    target_names=["Other songs", "High-popularity songs"],
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
