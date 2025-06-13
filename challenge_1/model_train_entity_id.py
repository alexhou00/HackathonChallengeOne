import pandas as pd
import numpy as np
import json
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack
from preprocess import preprocess

# Load entity catalog to understand the relationships
with open('../data/shared/entity_catalog.json', 'r', encoding='utf-8') as f:
    catalog = json.load(f)

# Load data
train = pd.read_csv('../data/challenge_1/train/classification_data.csv')
val = pd.read_csv('../data/challenge_1/val/classification_data.csv')
test = pd.read_csv('../data/challenge_1/test/classification_data.csv')

print(f"Train data shape: {train.shape}")
print(f"Validation data shape: {val.shape}")
print(f"Test data shape: {test.shape}")

# Analyze target distribution
print("\nTarget distribution in training data:")
print(train['responsible_entity_id'].value_counts().head(10))
print(f"Total unique entities in training data: {train['responsible_entity_id'].nunique()}")

# Add category-entity mapping as a feature
category_entity_map = catalog['category_entity_map']

# Function to create additional features
def create_features(df):
    # Create a copy to avoid modifying the original dataframe
    df_copy = df.copy()

    # Add state code feature (extract from entity_id if available)
    if 'responsible_entity_id' in df_copy.columns:
        df_copy['entity_state_code'] = df_copy['responsible_entity_id'].apply(
            lambda x: x.split('_')[1] if 'LAND_' in x else 'BUND' if 'BUND_' in x else 'OTHER'
        )

    # Add ministry type feature (extract from entity_id if available)
    if 'responsible_entity_id' in df_copy.columns:
        df_copy['entity_ministry_type'] = df_copy['responsible_entity_id'].apply(
            lambda x: x.split('_')[-1] if '_' in x else 'OTHER'
        )

    # Add feature for whether the entity is in the same state as the issue
    if 'responsible_entity_id' in df_copy.columns and 'state' in df_copy.columns:
        # Map state names to state codes (simplified mapping)
        state_to_code = {
            'Baden-Württemberg': '08',
            'Bayern': '09',
            'Berlin': '11',
            'Brandenburg': '12',
            'Bremen': '04',
            'Hamburg': '02',
            'Hessen': '06',
            'Mecklenburg-Vorpommern': '13',
            'Niedersachsen': '03',
            'Nordrhein-Westfalen': '05',
            'Rheinland-Pfalz': '07',
            'Saarland': '10',
            'Sachsen': '14',
            'Sachsen-Anhalt': '15',
            'Schleswig-Holstein': '01',
            'Thüringen': '16'
        }

        df_copy['state_code'] = df_copy['state'].map(state_to_code)
        df_copy['is_same_state'] = df_copy.apply(
            lambda row: 'LAND_' + row['state_code'] in row['responsible_entity_id'] 
            if 'LAND_' in str(row['responsible_entity_id']) and pd.notnull(row['state_code']) 
            else False, 
            axis=1
        )

    # Add feature for possible entities based on category
    if 'category' in df_copy.columns:
        df_copy['possible_entities'] = df_copy['category'].apply(
            lambda cat: len(category_entity_map.get(cat, []))
        )

        # Add binary features for each category
        for category in category_entity_map.keys():
            df_copy[f'is_{category.lower()}'] = (df_copy['category'] == category).astype(int)

    return df_copy

# Create additional features
train_enhanced = create_features(train)
val_enhanced = create_features(val)
test_enhanced = create_features(test)

# Select numerical features
numerical_features = [
    'description_length', 'description_words', 
    'has_verkehr_keywords', 'has_bildung_keywords', 
    'has_umwelt_keywords', 'has_gesundheit_keywords',
    'hour', 'day_of_week', 'day_of_month', 'week_of_year', 
    'month', 'quarter', 'is_weekend', 'is_business_hours', 
    'is_morning', 'is_afternoon', 'possible_entities'
]

# Add category binary features
for category in category_entity_map.keys():
    if f'is_{category.lower()}' in train_enhanced.columns:
        numerical_features.append(f'is_{category.lower()}')

# Target: Predict the responsible_entity_id
le = LabelEncoder()
train_enhanced['target'] = le.fit_transform(train_enhanced['responsible_entity_id'])
val_enhanced['target'] = le.transform(val_enhanced['responsible_entity_id'])

# Feature preprocessing
X_train_cat, X_train_text, tfidf = preprocess(train_enhanced)
X_val_cat, X_val_text, _ = preprocess(val_enhanced, tfidf=tfidf, fit=False)
X_test_cat, X_test_text, _ = preprocess(test_enhanced, tfidf=tfidf, fit=False)

# Convert numerical features to sparse matrix
from scipy.sparse import csr_matrix

# Ensure all numerical features are of the correct data type
for col in numerical_features:
    if col in train_enhanced.columns:
        # Convert boolean to int
        if train_enhanced[col].dtype == bool:
            train_enhanced[col] = train_enhanced[col].astype(int)
            val_enhanced[col] = val_enhanced[col].astype(int)
            test_enhanced[col] = test_enhanced[col].astype(int)
        # Convert object to float if possible
        elif train_enhanced[col].dtype == object:
            train_enhanced[col] = pd.to_numeric(train_enhanced[col], errors='coerce')
            val_enhanced[col] = pd.to_numeric(val_enhanced[col], errors='coerce')
            test_enhanced[col] = pd.to_numeric(test_enhanced[col], errors='coerce')

# Convert to sparse matrix
X_train_num = csr_matrix(train_enhanced[numerical_features].fillna(0).values)
X_val_num = csr_matrix(val_enhanced[numerical_features].fillna(0).values)
X_test_num = csr_matrix(test_enhanced[numerical_features].fillna(0).values)

# Combine sparse matrices
X_train = hstack([X_train_cat, X_train_text, X_train_num])
X_val = hstack([X_val_cat, X_val_text, X_val_num])
X_test = hstack([X_test_cat, X_test_text, X_test_num])

print(f"\nTraining with {X_train.shape[1]} features")

# Train model
model = lgb.LGBMClassifier(
    objective='multiclass', 
    num_class=len(le.classes_), 
    class_weight='balanced',
    n_estimators=200,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=31,
    random_state=42
)

model.fit(X_train, train_enhanced['target'])

# Predict and evaluate
val_preds = model.predict(X_val)
print("\nValidation Results:")
print("Accuracy:", accuracy_score(val_enhanced['target'], val_preds))
print("F1 Score (macro):", f1_score(val_enhanced['target'], val_preds, average='macro'))

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(val_enhanced['target'], val_preds))

# Generate predictions for test data
test_preds = model.predict(X_test)
test_pred_entities = le.inverse_transform(test_preds)

# Create submission file
submission = pd.DataFrame({
    'issue_id': test['issue_id'],
    'responsible_entity_id': test_pred_entities
})

# Save to CSV
submission.to_csv('../challenge1_submission.csv', index=False)
print("\nSubmission file created: ../challenge1_submission.csv")

# Optional: print predicted entities for validation data
val_enhanced['predicted_entity_id'] = le.inverse_transform(val_preds)
print("\nSample predictions:")
print(val_enhanced[['issue_id', 'responsible_entity_id', 'predicted_entity_id']].head(10))

# Feature importance
feature_importance = model.feature_importances_
print("\nTop 20 feature importances:")
# Since we're using sparse matrices, we don't have feature names directly
# We can print the indices of the most important features
top_indices = np.argsort(feature_importance)[-20:][::-1]
for idx in top_indices:
    print(f"Feature {idx}: {feature_importance[idx]}")
