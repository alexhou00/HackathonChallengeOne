# challenge_1/preprocessing.py

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def build_preprocessor():
    categorical_cols = ['age_group', 'gender', 'origin']
    #categorical_cols = ['category', 'age_group', 'gender', 'origin']

    numeric_cols = [
        'description_length', 'description_words',
        'has_verkehr_keywords', 'has_bildung_keywords',
        'has_umwelt_keywords', 'has_gesundheit_keywords',
        'hour', 'day_of_week', 'day_of_month',
        'week_of_year', 'month', 'quarter',
        'is_weekend', 'is_business_hours',
        'is_morning', 'is_afternoon'
    ]

    print("🔧 Building preprocessor")
    print(f"🧾 Categorical columns: {categorical_cols}")
    print(f"🔢 Numerical columns: {numeric_cols}")

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ('cat', cat_pipeline, categorical_cols),
        ('num', num_pipeline, numeric_cols)
    ])

    return preprocessor
