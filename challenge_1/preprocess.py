import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords if needed
try:
    stopwords.words("german")
except LookupError:
    nltk.download("stopwords")

german_stopwords = stopwords.words("german")

# Categorical columns commonly used in all sets
CATEGORICAL_COLS = ['municipality', 'district', 'state', 'category', 'age_group', 'gender', 'origin']

# Text column
TEXT_COLUMN = 'description'

# Shared OneHotEncoder instance to ensure consistent encoding
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)

def preprocess(df: pd.DataFrame, tfidf: TfidfVectorizer = None, fit: bool = True):
    # Select available categorical columns
    available_cat_cols = [col for col in CATEGORICAL_COLS if col in df.columns]
    cat_data = df[available_cat_cols].fillna('missing')

    # One-hot encode categorical features
    if fit:
        cat_encoded = encoder.fit_transform(cat_data)
    else:
        cat_encoded = encoder.transform(cat_data)

    # Prepare text data
    text_data = df[TEXT_COLUMN].fillna("")

    if tfidf is None:
        tfidf = TfidfVectorizer(stop_words=german_stopwords, max_features=1000)
        X_text = tfidf.fit_transform(text_data)
    else:
        X_text = tfidf.transform(text_data)

    return cat_encoded, X_text, tfidf
