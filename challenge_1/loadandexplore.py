import pandas as pd
import json

# Load training and validation data
train_df = pd.read_csv('../data/challenge_1/train/classification_data.csv')
val_df = pd.read_csv('../data/challenge_1/val/classification_data.csv')

# Load metadata
with open('../data/shared/entity_catalog.json') as f:
    entity_catalog = json.load(f)

with open('../data/shared/categories.json') as f:
    categories_info = json.load(f)

# Explore columns
print(train_df.columns)
train_df.head()

print(train_df['responsible_entity_id'].value_counts().head(10))
print(train_df.groupby('category')['responsible_entity_id'].nunique())
print(pd.crosstab(train_df['has_verkehr_keywords'], train_df['category']))
