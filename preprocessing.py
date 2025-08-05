import joblib
import os
import pandas as pd
from scipy.sparse import hstack, csr_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Aynı sırayla olmalı!
NUMERIC_COLS = [
    'word_count', 'char_count', 'sentence_count', 'avg_word_length',
    'exclamation_count', 'capital_count', 'positive_word_count',
    'negative_word_count', 'neutral_word_count', 'review_time_delta',
    'has_emoji', 'contains_url', 'sentiment_compound',
    'taste_mention', 'price_mention', 'packaging_mention',
    'shipping_mention', 'quality_mention', 'delivery_mention',
    'size_mention', 'smell_mention'
]

def load_vectorizer():
    return joblib.load(os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl"))

def load_model():
    return joblib.load(os.path.join(BASE_DIR, "models", "lightgbm_model_v2.pkl"))

def load_label_encoder():
    return joblib.load(os.path.join(BASE_DIR, "models", "label_encoder.pkl"))

def predict_sentiment(df_subset):
    vectorizer = load_vectorizer()
    model = load_model()
    label_encoder = load_label_encoder()

    # TF-IDF vektörü
    tfidf_matrix = vectorizer.transform(df_subset["Clean_Text"].astype(str))

    # Numeric sütunlar
    X_numeric = df_subset[NUMERIC_COLS].fillna(0).astype(float)
    X_numeric_sparse = csr_matrix(X_numeric)

    # Birleştir
    X_combined = hstack([tfidf_matrix, X_numeric_sparse])

    # Tahmin et
    preds = model.predict(X_combined)
    return label_encoder.inverse_transform(preds)
