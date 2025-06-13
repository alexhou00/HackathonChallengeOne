# challenge_1/model_train.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import numpy as np

# === Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'challenge_1', 'train', 'classification_data.csv')

# === Load data
print(f"📄 Using dataset: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

TARGET_COL = 'responsible_entity_level'
drop_cols = [
    'issue_id', 'timestamp',
    'responsible_entity_id', 'responsible_entity_name',
    'responsible_entity_level',  # target
    'description', 'municipality', 'district', 'state'  # ⚠️ removed due to leakage
]

#X = df.drop(columns=drop_cols, errors='ignore')
X = df.drop(columns=drop_cols + ['category'], errors='ignore')
y = df[TARGET_COL]

# === Diagnostics
print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"🎯 Target distribution:\n{y.value_counts()}")
print(f"🧪 Features used for training: {list(X.columns)}")
print("🔍 Sample row:")
print(X.iloc[0])

# === Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Import preprocessing
from preprocessing import build_preprocessor
preprocessor = build_preprocessor()

# === Build pipeline
model = LogisticRegression(max_iter=1000, random_state=42)
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# === Train model
print("🚀 Training the model...")
pipeline.fit(X_train, y_train)
print("✅ Training complete!")

# === Evaluate
y_pred = pipeline.predict(X_test)
print("\n📊 Evaluation on test set:")
print(classification_report(y_test, y_pred))

# === Sample predictions
print("🔮 Sample predictions:")
for i in range(min(5, len(y_test))):
    print(f"  True: {y_test.iloc[i]} | Predicted: {y_pred[i]}")

# === Training accuracy
train_score = pipeline.score(X_train, y_train)
print(f"🎯 Training accuracy: {train_score:.4f}")

# === Leakage test
print("\n🧪 Testing for data leakage with shuffled labels...")
y_shuffled = np.random.permutation(y_train)
pipeline.fit(X_train, y_shuffled)
shuffled_score = pipeline.score(X_test, y_test)
print(f"🎭 Accuracy with shuffled labels: {shuffled_score:.4f}")
print("⚠️  If this is high (>0.5), you likely have leakage.")

# === Save model
MODEL_PATH = os.path.join(BASE_DIR, 'trained_model.joblib')
joblib.dump(pipeline, MODEL_PATH)
print(f"\n💾 Model saved to: {MODEL_PATH}")

# === Feature importance diagnostics ===
from sklearn.ensemble import RandomForestClassifier

print("\n🌲 Training RandomForest to check feature importance...")
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])
rf_pipeline.fit(X_train, y_train)

# Get feature names after preprocessing
feature_names = rf_pipeline.named_steps['preprocessor'].get_feature_names_out()
importances = rf_pipeline.named_steps['classifier'].feature_importances_

# Combine and sort
import pandas as pd
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

print("\n🔥 Top 20 most important features:")
print(importance_df.head(20).to_string(index=False))
