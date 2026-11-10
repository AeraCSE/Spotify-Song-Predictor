import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.dummy import DummyClassifier

# Load and clean data
df = pd.read_csv("dataset.csv")
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
df = df.drop_duplicates(subset=["track_id"])

features = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence",
    "tempo", "duration_ms"
]
df = df.dropna(subset=features + ["popularity"])
df["hit"] = (df["popularity"] >= 70).astype(int)

X, y = df[features], df["hit"]

# Keep a final test set untouched during threshold selection
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Split training data again to select the Logistic Regression threshold
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train, y_train, test_size=0.20, random_state=42, stratify=y_train
)

logistic = make_pipeline(
    StandardScaler(),
    LogisticRegression(class_weight="balanced", max_iter=1000)
)
logistic.fit(X_fit, y_fit)

val_probabilities = logistic.predict_proba(X_val)[:, 1]
thresholds = np.arange(0.05, 0.96, 0.01)
threshold_scores = [
    f1_score(y_val, val_probabilities >= threshold, zero_division=0)
    for threshold in thresholds
]
best_threshold = float(thresholds[int(np.argmax(threshold_scores))])

# Refit on all training data; evaluate once on the untouched test set
logistic.fit(X_train, y_train)

models = {
    "Logistic Regression": (logistic, best_threshold),
    "Random Forest": (
        RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        0.50
    )
}

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)
print("Baseline F1-score:", round(
    f1_score(y_test, baseline_pred, zero_division=0), 4
))
print("Selected Logistic Regression threshold:", round(best_threshold, 2))

os.makedirs("models", exist_ok=True)
results = []

for name, (model, threshold) in models.items():
    print(f"\nTraining/evaluating {name}...")
    if name == "Random Forest":
        model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    row = {
        "Model": name,
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1-score": f1_score(y_test, predictions, zero_division=0)
    }
    results.append(row)

    print(row)
    print(classification_report(
        y_test, predictions,
        target_names=["Other", "High popularity"],
        zero_division=0
    ))
    print("Confusion matrix:\n", confusion_matrix(y_test, predictions))

    filename = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(
        {"model": model, "features": features, "threshold": threshold},
        os.path.join("models", filename)
    )

results_df = pd.DataFrame(results).sort_values("F1-score", ascending=False)
results_df.to_csv("model_results.csv", index=False)
print("\nMODEL COMPARISON\n", results_df.to_string(index=False))
print("\nSaved models and model_results.csv.")
