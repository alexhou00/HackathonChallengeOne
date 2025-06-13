from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def preprocess(df, tfidf=None, fit=True):
    text = df['description'].fillna('')
    if fit:
        tfidf = TfidfVectorizer(max_features=1000, stop_words='german')
        X_text = tfidf.fit_transform(text)
    else:
        X_text = tfidf.transform(text)

    cat_cols = ['category', 'municipality', 'district', 'state', 'age_group', 'gender', 'origin']
    eng_cols = [col for col in df.columns if col.startswith('has_') or col in [
        'description_length', 'description_words', 'hour', 'day_of_week',
        'month', 'quarter', 'is_weekend', 'is_business_hours',
        'is_morning', 'is_afternoon'
    ]]

    X_cat = pd.get_dummies(df[cat_cols + eng_cols])
    return X_cat, X_text, tfidf
