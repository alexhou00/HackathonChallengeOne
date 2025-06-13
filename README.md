# PrecisionPiranhas: Challenge 1

## Wer ist zuständig? – Politisches Mapping von Bürgeranliegen

## Task
Develop a system that automatically determines political responsibilities for citizen concerns. Given a list of problems with assigned categories (e.g., "Verkehr", "Bildung", "Migration"), develop a mapping mechanism that assigns these categories to appropriate political levels and actors – e.g., a member of the Bundestag, a state parliament, or a municipal office

The classification system processes German text descriptions and additional keyword features to predict which government entity (`responsible_entity_id`) should handle specific issues. The model handles multiple classes including federal ministries (BUND) and various state-level departments (LAND) across different German states. In the future the model could be expanded to KOMMUNEN as well.

## Project Structure

```
HackathonChallengeOne/
├── challenge_1/
│   ├── basic_model_explore.ipynb
│   ├── basic_model_explore_alex_17.ipynb
├── data/
│   ├── challenge_1/
│   │   ├── test/
│   │   │   └── classification_data.csv
│   │   ├── train/
│   │   │   └── classification_data.csv
│   │   ├── val/
│   │   │   └── classification_data.csv
│   │   └── metadata.json
│   └── shared/
├── outputs/
│   ├── logs/
│   │   ├── challenge1_YYYY-MM-DD_HH-MM-SS-xxx-alex_17.log
│   │   └── challenge1_YYYY-MM-DD_HH-MM-SS-xxx.log
│   ├── models/
│   │   ├── challenge1_model.pkl
│   │   └── challenge1_model-alex_17.pkl
│   └── submission/
│       ├── challenge1_submission-YYYY-MM-DD_HH-MM-SS-xxx-alex_17.csv
│       ├── challenge1_submission-YYYY-MM-DD_HH-MM-SS-xxx.csv
│       ├── test_with_predictions-YYYY-MM-DD_HH-MM-SS-xxx-alex_17.csv
│       └── test_with_predictions-YYYY-MM-DD_HH-MM-SS-xxx.csv


```

## Features

### Text Features
- **Primary Feature**: `description` - German text descriptions of issues, state - where the complaint is filed from,
category - under which category this complaint fell under
- **Preprocessing**: TF-IDF vectorization with German stopwords removal
- **Dimensionality Reduction**: TruncatedSVD to 300 components

### Additional Features
- `has_verkehr_keywords` - Traffic/transportation keywords indicator
- `has_bildung_keywords` - Education keywords indicator
- `has_umwelt_keywords` - Environment keywords indicator
- `has_gesundheit_keywords` - Health keywords indicator

## Model Architecture
The model is a suprervised machine learning piepline designed to classify German-language complaint descriptions into over 100 categories. 
1. **Input**: Raw German text (e.g., customers complaint). 
2. **Preprocessing**: 
   - Lowercasing 
   - Removal of German stopwords (using NLTK)
   - Bigrams are included
   - Limits feature size to 10,000 most informative terms. 
3. **ColumnTransformer**:
   - Combines the transformed text with other features (e.g., keyword-based features).
4. **Classifier**:
   - A Decision Tree or Logistic Regression model (depending on what’s used in `clf`).
   - Trained to predict the correct complaint category from over 100 classes.



### Preprocessing Pipeline
1. **Text Processing**:
   - TF-IDF Vectorization
     - Max features: 10,000
     - N-gram range: (1, 2)
     - Min document frequency: 3
     - Max document frequency: 0.9
     - German stopwords removal
   - Dimensionality reduction with TruncatedSVD (300 components)

2. **Feature Combination**:
   - Text features (300 dimensions from SVD)
   - Binary keyword indicators (4 features)

### Classification Model
- **Algorithm**: Random Forest Classifier
- **Parameters**:
  - 100 estimators
  - Balanced class weights
  - Random state: 42
  - Parallel processing enabled

1. basic_model_explore.ipynb
- Selects the 5 most frequently mentioned classes of responsible entities
- After computing the 5 most frequently mentioned class, the model goes through the list of
data to decide which of the 5 most frequently mentioned class it belongs to 
- The preprocessing pipeline also reduces the number of features in the data to 300 components 
- loads test data and applies the pipeline to predict responsible entities 
- saves submission CSV with predictions to outputs/submission


2. intended method if time would permit (basic_model_explore_alex_17.ipynb)
- create a log directory if it doesnt exist 
- process data by splitting and filtering

                                    🌳 Start: All Complaints
                                        │
                           ┌────────────┴────────────┐
                           │                         │
               🏛️ Contains Land keywords?       🏛️ Contains Bund keywords?
                     (has_land_keywords)             (has_bund_keywords)
                           │                         │
               ┌───────────┴───────────┐   ┌─────────┴─────────┐
               │                       │   │                   │
       🚦 Verkehr-related?     🎓 Bildung-related?    🌳 Umwelt-related?     🏥 Gesundheit-related?
    (has_verkehr_keywords)   (has_bildung_keywords) (has_umwelt_keywords) (has_gesundheit_keywords)
               │                       │                   │                      │
          [Route to Entity A]   [Route to Entity B]   [Entity C]            [Entity D]







## Installation

### Requirements
```bash
pip install pandas numpy scikit-learn nltk
```

### NLTK Setup
The German stopwords are automatically downloaded if not available:
```python
import nltk
nltk.download("stopwords")
```

## Usage

### Training and Evaluation
```python
# Load the notebook and run all cells
#notebook basic_model_explore.ipynb
```

### Key Steps:
1. **Data Loading**: Loads training and validation datasets
2. **Feature Engineering**: Combines text and keyword features
3. **Model Training**: Fits Random Forest on preprocessed features
4. **Evaluation**: Generates predictions and performance metrics
5. **Submission**: Creates final predictions for test data

## Performance

### Current Results
- **Validation Accuracy**: ~15.4%
- **Challenge**: High class imbalance with 90+ different government entities
- **Best Performing Classes**: 
  - Federal Ministry for Digital and Transport: 51% precision, 100% recall
  - Some state-level departments show limited success  


  **The training data is not sufficient to increase the accuracy of descriptions with less frequency**  
  
- **Validation Accuracy with most frequent asked issues of bias N5**: ~84.4%
- However, the other classes will be ignored

## Output Files

### Submission File
- **Location**: `../outputs/submission/challenge1_submission.csv`
- **Format**: 
  ```csv
  issue_id,responsible_entity_id
  1,BUND_BUNDESMINISTERIUM_FÜR_DIGITALES_UND_VERKEHR
  2,LAND_01_SM
  ...
  ```

### Debug File
- **Location**: `../outputs/submission/test_with_predictions.csv`
- **Content**: Full test dataset with predicted entities for analysis

## Government Entity Categories

### Federal Level (BUND)
- `BUND_BUNDESMINISTERIUM_FÜR_DIGITALES_UND_VERKEHR` - Federal Ministry for Digital and Transport

### State Level (LAND_XX)
States numbered 01-16 with departments:
- **BM**: Building/Construction Ministry
- **FM**: Finance Ministry  
- **GM**: Health Ministry
- **IM**: Interior Ministry
- **MW**: Economic Ministry
- **SM**: Social Ministry
- **UM**: Environment Ministry
- **VM**: Transport Ministry

## Potential Improvements

### Model Enhancements
1. **Address Class Imbalance**:
   - SMOTE oversampling
   - Focal loss implementation
   - Hierarchical classification

2. **Feature Engineering**:
   - Advanced text preprocessing (lemmatization, stemming)
   - Domain-specific word embeddings
   - Additional keyword categories

3. **Model Architecture**:
   - Ensemble methods (XGBoost, CatBoost)
   - Neural networks (BERT for German text)
   - Multi-level classification (federal vs state, then specific department)

### Data Analysis
1. **Class Distribution Analysis**: Understand entity frequency patterns
2. **Text Analysis**: Identify discriminative terms per entity
3. **Cross-Validation**: Implement stratified k-fold validation

## Troubleshooting

### Common Issues
1. **Memory Errors**: Reduce `max_features` in TfidfVectorizer
2. **Long Training Time**: Decrease `n_estimators` or use `n_jobs=-1`
3. **Poor Performance**: Consider feature selection or dimensionality reduction

### Performance Optimization
- Use sparse matrices for TF-IDF representation
- Implement early stopping for tree-based models
- Consider feature hashing for very large vocabularies

## Contributing

When improving the model:
1. Maintain reproducibility with fixed random seeds
2. Document parameter changes and their impact
3. Preserve the submission file format
4. Add validation metrics for new approaches

## License

This project is part of a classification challenge for German government entity prediction.