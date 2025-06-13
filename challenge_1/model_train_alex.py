import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from lightgbm import LGBMClassifier

# 1. Load datasets
def load_data():
    train = pd.read_csv('../data/challenge_1/train/classification_data.csv')
    val = pd.read_csv('../data/challenge_1/val/classification_data.csv')
    test = pd.read_csv('../data/challenge_1/test/classification_data.csv')
    return train, val, test

# 2. Define feature groups
NUMERIC_FEATURES = [
    'description_length', 'description_words',
    'hour', 'day_of_week', 'day_of_month', 'week_of_year', 'month', 'quarter',
    'is_weekend', 'is_business_hours', 'is_morning', 'is_afternoon'
]

BINARY_KEYWORD_FLAGS = [col for col in ['has_verkehr_keywords', 'has_bildung_keywords',
                                         'has_umwelt_keywords', 'has_gesundheit_keywords']]

CAT_FEATURES = ['category', 'municipality', 'district', 'state', 'age_group', 'gender', 'origin']
TEXT_FEATURE = 'description'

# 3. Build preprocessing pipeline

def build_preprocessor():
    # Numeric features: scale
    num_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Categorical features: one-hot
    cat_transformer = Pipeline(
        steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))]
    )

    # Text features: TF-IDF
    text_transformer = Pipeline(
        steps=[('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2)))]
    )

    # Combine all
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, NUMERIC_FEATURES),
            ('bin', 'passthrough', BINARY_KEYWORD_FLAGS),
            ('cat', cat_transformer, CAT_FEATURES),
            ('text', text_transformer, TEXT_FEATURE)
        ],
        remainder='drop'
    )
    return preprocessor

# 4. Train Model

def train_model(train, val):
    # Encode target
    le = LabelEncoder()
    y_train = le.fit_transform(train['responsible_entity_id'])
    y_val = le.transform(val['responsible_entity_id'])

    X_train = train[NUMERIC_FEATURES + BINARY_KEYWORD_FLAGS + CAT_FEATURES + [TEXT_FEATURE]]
    X_val = val[NUMERIC_FEATURES + BINARY_KEYWORD_FLAGS + CAT_FEATURES + [TEXT_FEATURE]]

    preprocessor = build_preprocessor()

    # Full pipeline
    clf = Pipeline(steps=[
        ('preproc', preprocessor),
        ('clf', LGBMClassifier(objective='multiclass', class_weight='balanced',
                               num_class=len(le.classes_), random_state=42))
    ])

    # Simple grid search
    param_grid = {
        'clf__num_leaves': [31, 63],
        'clf__n_estimators': [100, 200]
    }
    grid = GridSearchCV(clf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    preds = grid.predict(X_val)
    print("Validation Accuracy:", np.mean(preds == y_val))
    print("Sample predictions:")
    print(le.inverse_transform(preds)[:5])

    return grid.best_estimator_, le

# 5. Predict on test

def predict_and_save(model, le, test):
    X_test = test[NUMERIC_FEATURES + BINARY_KEYWORD_FLAGS + CAT_FEATURES + [TEXT_FEATURE]]
    preds = model.predict(X_test)
    test['responsible_entity_id'] = le.inverse_transform(preds)
    submission = test[['issue_id', 'responsible_entity_id']]
    submission.to_csv('challenge1_submission.csv', index=False)

# 6. Main execution

if __name__ == '__main__':
    train, val, test = load_data()
    model, le = train_model(train, val)
    predict_and_save(model, le, test)
