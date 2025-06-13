import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix

# Ensure NLTK stopwords are available
import nltk
try:
    from nltk.corpus import stopwords
    german_stopwords = stopwords.words('german')
except LookupError:
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    german_stopwords = stopwords.words('german')

# Cached encoder (optional but keeps it consistent between fit/transform)
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)

def preprocess(df, tfidf=None, fit=True):
    # --- 1. Text column ---
    text = df['description'].fillna('')

    if fit or tfidf is None:
        tfidf = TfidfVectorizer(stop_words=german_stopwords, max_features=10000)
        X_text = tfidf.fit_transform(text)
    else:
        X_text = tfidf.transform(text)

    # --- 2. Categorical features ---
    categorical_cols = ['category']  # or whatever exists in your CSV
    cat_data = df[categorical_cols].fillna('missing')

    if fit:
        X_cat = encoder.fit_transform(cat_data)
    else:
        X_cat = encoder.transform(cat_data)

    # Convert to sparse DataFrame for alignment
    X_cat_df = pd.DataFrame.sparse.from_spmatrix(X_cat, index=df.index)

    return X_cat_df, X_text, tfidf
