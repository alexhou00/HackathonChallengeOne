# Challenge 1: Wer ist zuständig? – Politisches Mapping von Bürgeranliegen

## Task
Develop a system that automatically determines political responsibilities for citizen concerns. Given a list of problems with assigned categories (e.g., "Verkehr", "Bildung", "Migration"), develop a mapping mechanism that assigns these categories to appropriate political levels and actors – e.g., a member of the Bundestag, a state parliament, or a municipal office.

## Goal
A functioning assignment logic (e.g., simple rule engine or ML model) that directly assigns the responsible political offices to a concern.

## Data Structure

All data for this challenge is in `../data/challenge_1/`:

### Training Data
- `train/classification_data.csv` - Training dataset with labeled examples
- `val/classification_data.csv` - Validation dataset for tuning
- `test/classification_data.csv` - Test dataset (**WITHOUT LABELS** - you must predict them!)

### Shared Resources
- `../data/shared/entity_catalog.json` - Complete catalog of all 179 government entities
- `../data/shared/categories.json` - List of issue categories with descriptions

### Data Fields

#### Input Features
- `issue_id`: Unique identifier
- `timestamp`: When the issue was submitted
- `description`: Citizen complaint text (German, 1-2 sentences)
- `category`: Problem category (Verkehr, Bildung, Umwelt, Gesundheit, etc.)
- `municipality`, `district`, `state`: Geographic location
- `age_group`, `gender`, `origin`: Demographic data
- Additional engineered features:
  - Text features: `description_length`, `description_words`, `has_*_keywords`
  - Geographic features: `dist_to_*` (distances to major cities), regional indicators
  - Temporal features: `hour`, `day_of_week`, `month`, `is_weekend`, etc.

#### Target Variables (in train/val only, NOT in test)
- `responsible_entity_id`: The ID of the responsible authority (TARGET - you must predict this)
- `responsible_entity_name`: Human-readable name
- `responsible_entity_level`: Government level (Bund, Land, Kreis, Kommune)

## Understanding the Entity Catalog

The entity catalog contains detailed information about each government entity:

```python
import json

with open('../data/shared/entity_catalog.json', 'r') as f:
    catalog = json.load(f)

# Explore structure
print(f"Total entities: {len(catalog['entities'])}")
print(f"Entities by level: {catalog['metadata']['entities_by_level']}")

# Find entities for a specific category
verkehr_entities = catalog['category_entity_map']['Verkehr']
print(f"Entities handling Verkehr issues: {verkehr_entities}")

# Get entity details
entity = catalog['entities']['BUND_BMDV']
print(f"Name: {entity['name']}")
print(f"Level: {entity['level']}")
print(f"Competencies: {entity['competencies']}")
```

## Test Data & Submission Format

### ⚠️ Important: Test Data Has No Labels!
The test dataset (`data/challenge_1/test/classification_data.csv`) does **NOT** contain the target columns:
- ❌ `responsible_entity_id` (removed - you must predict this!)
- ❌ `responsible_entity_name` (removed)
- ❌ `responsible_entity_level` (removed)

### Submission Format
Your predictions must be submitted as a CSV file with exactly two columns:
```csv
issue_id,responsible_entity_id
ISS_20240612_0001,LAND_BE_VM
ISS_20240612_0002,BUND_BMDV
ISS_20240612_0003,KOMMUNE_HAMBURG
...
```

### How to Create Your Submission
```python
# Load test data
test_df = pd.read_csv('../data/challenge_1/test/classification_data.csv')

# Make predictions with your model
predictions = your_model.predict(test_df)

# Create submission file
submission = pd.DataFrame({
    'issue_id': test_df['issue_id'],
    'responsible_entity_id': predictions
})

# Save to CSV
submission.to_csv('challenge1_submission.csv', index=False)
```

### Submission Process
1. Train your model using the training data
2. Validate using the validation set
3. Generate predictions for the test set
4. Create your submission CSV file
5. Upload your solution to GitHub (or similar)
6. Submit the link to your repository on our hackathon website

Your repository should include:
- Your code (Jupyter notebooks, Python scripts, etc.)
- Your final predictions (`challenge1_submission.csv`)
- A brief README explaining your approach
- Any requirements files needed to run your code

## Evaluation Metrics
- **Primary**: Accuracy - Percentage of correctly classified entities
- **Secondary**: F1-Score (macro) - Balanced performance across all classes
- **Note**: We will evaluate your predictions against the hidden ground truth

## Tips
- Start with a simple rule-based approach to understand the data
- The German federal structure is hierarchical: some issues clearly belong to specific levels
- Text analysis is important - German keywords can indicate jurisdiction
- Geographic location matters: some entities are state-specific
- Consider the competencies listed in the entity catalog
- Some categories have predictable patterns (e.g., Education → usually State level)
- Ensemble methods often work well for this type of multi-class problem

## Common Pitfalls to Avoid
- Don't ignore the government level (`responsible_entity_level`) in training - it's a strong signal
- Remember that entity IDs follow patterns: `BUND_*`, `LAND_*_*`, `KOMMUNE_*`
- Some entities handle multiple categories - check the category_entity_map
- Test data is temporally separated - ensure your model generalizes over time
- **Important**: The test data does NOT have the target columns - don't try to access them!