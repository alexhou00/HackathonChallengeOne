import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack
from preprocess import preprocess

# Load data
train = pd.read_csv('../data/challenge_1/train/classification_data.csv')
val = pd.read_csv('../data/challenge_1/val/classification_data.csv')

# Target: Predict the responsible_entity_level (Kommune, Land, Bund)
le = LabelEncoder()
train['target'] = le.fit_transform(train['responsible_entity_level'])
val['target'] = le.transform(val['responsible_entity_level'])

# Feature preprocessing
X_train_cat, X_train_text, tfidf = preprocess(train)
X_val_cat, X_val_text, _ = preprocess(val, tfidf=tfidf, fit=False)

# Combine sparse matrices
X_train = hstack([X_train_cat, X_train_text])
X_val = hstack([X_val_cat, X_val_text])

# Train model
model = lgb.LGBMClassifier(objective='multiclass', num_class=len(le.classes_), class_weight='balanced')
model.fit(X_train, train['target'])

# Predict and evaluate
preds = model.predict(X_val)
print("Accuracy:", accuracy_score(val['target'], preds))
print("F1 Score:", f1_score(val['target'], preds, average='macro'))

# Optional: print predicted classes
val['predicted_level'] = le.inverse_transform(preds)
print(val[['issue_id', 'responsible_entity_level', 'predicted_level']].head())
