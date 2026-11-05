
# Spotify Song Hit Predictor 🎧

An interactive Machine Learning application that estimates whether a song belongs to the high-popularity category based on its audio characteristics and user-selected creative preferences.

## Project Overview

This project explores the relationship between Spotify audio features and song popularity using supervised machine learning.

Users can describe the type of song they want to create by selecting its genre, mood, energy, vocals, tempo and duration. These selections are converted into approximate numerical audio features, which are passed to trained classification models to estimate the song's popularity category.

## Dataset

- Original dataset: Approximately 114,000 records.
- Target variable: `hit`
- Hit (1): Popularity >= 70
- Other (0): Popularity < 70

The popularity threshold of 70 is defined for this project.

## Input Features

The models use 10 numerical audio features:

- Danceability
- Energy
- Loudness
- Speechiness
- Acousticness
- Instrumentalness
- Liveness
- Valence
- Tempo
- Duration (ms)

## User Inputs

The Streamlit application allows users to select:

- Genre: Pop, Hip-Hop, Rock, Electronic, Acoustic, Lo-fi
- Mood: Happy, Sad, Romantic, Relaxing, Energetic
- Energy: Low, Medium, High
- Vocals: Singing, Rap, Instrumental, Mixed
- Tempo: Slow, Moderate, Fast
- Duration: Under 2 minutes, 2–3 minutes, 3–4 minutes, Over 4 minutes

These creative choices are mapped to approximate numerical audio features before prediction. They are not extracted from an actual audio recording.

## Machine Learning Models

Two supervised classification algorithms are implemented:

1. Logistic Regression
2. Random Forest Classifier

Training process:
- Data cleaning and preprocessing
- Feature selection
- Binary target creation
- Stratified 80:20 train-test split
- Model training and evaluation
- Model comparison

Logistic Regression uses StandardScaler and both classifiers use class weighting to address class imbalance.

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

A Dummy Classifier is also used as a baseline.

## Application Features

- Spotify-inspired Streamlit dashboard
- Dataset overview and popularity distribution
- Interactive audio feature analysis
- User-defined song characteristics
- Audio profile visualization
- Logistic Regression and Random Forest comparison
- Hit / Other category prediction
- Estimated model probability

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Joblib

## Project Structure

```text
Spotify-Song-Predictor/
│
├── app.py
├── main.py
├── train_model.py
├── model_results.csv
├── requirements.txt
├── dataset.csv
│
└── models/
    ├── logistic_regression.joblib
    └── random_forest.joblib
```

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/AeraCSE/Spotify-Song-Predictor.git
```

2. Navigate to the project directory:

```bash
cd Spotify-Song-Predictor
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Place the required dataset file named `dataset.csv` in the project directory.

5. Train and save the models:

```bash
python train_model.py
```

6. Launch the application:

```bash
python -m streamlit run app.py
```

## Limitations

- User selections are mapped to approximate audio feature values.
- The application does not analyse uploaded audio files.
- Popularity depends on external factors such as artist recognition, marketing, trends and audience preferences.
- Model predictions are estimates and do not guarantee future popularity or streaming success.
- Class imbalance affects model performance.

## Future Scope

- Integrate Spotify API.
- Extract audio features from actual songs.
- Improve feature mapping.
- Experiment with class imbalance handling and hyperparameter tuning.
- Deploy the application online.

## Author

AeraCSE
