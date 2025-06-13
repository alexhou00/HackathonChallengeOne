import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack
from preprocess import preprocess
import pandas as pd

# Load data
train = pd.read_csv('../data/challenge_1/train/classification_data.csv')
val = pd.read_csv('../data/challenge_1/val/classification_data.csv')

# Encode target
le = LabelEncoder()
train['responsible_entity_int'] = le.fit_transform(train['responsible_entity_id'])
val['responsible_entity_int'] = le.transform(val['responsible_entity_id'])

# Preprocess features
X_train_cat, X_train_text, tfidf = preprocess(train)
X_val_cat, X_val_text, _ = preprocess(val, tfidf=tfidf, fit=False)

# Combine categorical + text features
X_train = hstack([X_train_cat.values, X_train_text])
X_val = hstack([X_val_cat.values, X_val_text])

# Train LightGBM model
model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=len(le.classes_),
    class_weight='balanced'
)
model.fit(X_train, train['target'])

# Predict and evaluate
preds = model.predict(X_val)
print("Accuracy:", accuracy_score(val['target'], preds))
print("F1 Score:", f1_score(val['target'], preds, average='macro'))
