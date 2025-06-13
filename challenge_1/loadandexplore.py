import pandas as pd

train = pd.read_csv('../data/challenge_1/train/classification_data.csv')
val = pd.read_csv('../data/challenge_1/val/classification_data.csv')

print(train.head())
print(train['responsible_entity_level'].value_counts())
print(train['category'].value_counts())
