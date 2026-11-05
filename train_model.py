
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.dummy import DummyClassifier

# 1. Load the original dataset
df = pd.read_csv("dataset.csv")
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
df = df.drop_duplicates(subset=["track_id"])

# 2. Select audio features
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

X = df[features]
y = df["hit"]

print("Songs:", len(df))
print("High-popularity songs:", int(y.sum()))

# 3. Split data while preserving class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# 4. Define models
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
        ),
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}

# 5. Baseline for comparison
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_predictions = baseline.predict(X_test)

print("\nBaseline F1-score:",
      round(f1_score(y_test, baseline_predictions, zero_division=0), 3))

# 6. Train, evaluate and save both models
os.makedirs("models", exist_ok=True)
results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
    })

    print("\n", name)
    print(classification_report(
        y_test,
        predictions,
        target_names=["Other", "High popularity"],
        zero_division=0,
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    filename = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(
        {"model": model, "features": features},
        os.path.join("models", filename),
    )

# 7. Compare models
results_df = pd.DataFrame(results).sort_values(
    "F1-score", ascending=False
)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

results_df.to_csv("model_results.csv", index=False)
print("\nModels saved in the models/ folder.")
print("Comparison saved to model_results.csv.")
