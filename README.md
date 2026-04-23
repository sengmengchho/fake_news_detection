# Fake News Detector (Ensemble Deep Learning)

This project is a Streamlit app that predicts whether a news article is **FAKE** or **TRUE** using an ensemble of:
- Dense Neural Network
- CNN
- BiLSTM
- Meta-learner (final decision)

## Project Structure

- `app.py` — Streamlit web app
- `models/` — trained model files (`*.h5`) and TF-IDF vectorizer (`tfidf_vectorizer.pkl`)
- `src/data/` — training/test datasets (`Fake.csv`, `True.csv`)
- `src/Notebook/` — notebook, logs, and generated outputs

## Requirements

Recommended Python: 3.10+

Main libraries used:
- streamlit
- tensorflow
- scikit-learn
- nltk
- pandas
- numpy

## Setup

1. Create and activate virtual environment
2. Install dependencies
3. Ensure model files exist in `models/`
4. Run the app:

```bash
streamlit run app.py
```

## Notes

- NLTK resources (`punkt`, `stopwords`) are downloaded automatically on first run.
- GPU is disabled in the app for model loading consistency.
- If models are missing, run your training notebook first and save outputs to `models/`.

## Output

The app shows:
- Final ensemble prediction (FAKE/TRUE)
- Confidence score
- Per-model probabilities and labels
- Optional detailed analysis and preprocessing preview
