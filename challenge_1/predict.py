import pandas as pd
from preprocess import preprocess
from model_train import model, tfidf, le  # assume imported or load from .pkl
from scipy.sparse import hstack

test = pd.read_csv('../data/challenge_1/test/classification_data.csv')
X_test_cat, X_test_text, _ = preprocess(test, tfidf=tfidf, fit=False)
X_test = hstack([X_test_cat.values, X_test_text])
preds = model.predict(X_test)

submission = pd.DataFrame({
    'issue_id': test['issue_id'],
    'responsible_entity_id': le.inverse_transform(preds)
})
submission.to_csv('./outputs/submission/challenge1_submission.csv', index=False)
