# Credit Card Fraud Detection using Machine Learning

**Repository:** Credit card fraud detection using multiple ML classifiers with comprehensive evaluation and visualization.  
**Author:** Nikhil  
**Last updated:** 2025-11-29

---

## Table of Contents

* [Project Overview](#project-overview)
* [Features](#features)
* [Files & Structure](#files--structure)
* [Requirements](#requirements)
* [Quick Start / Setup](#quick-start--setup)
* [Dataset Description](#dataset-description)
* [Data Preprocessing](#data-preprocessing)
* [Model Training](#model-training)
* [Model Performance](#model-performance)
* [Evaluation Metrics & Visualizations](#evaluation-metrics--visualizations)
* [Results & Insights](#results--insights)
* [Tips, Troubleshooting & FAQ](#tips-troubleshooting--faq)
* [Future Improvements](#future-improvements)
* [Contributing](#contributing)
* [License](#license)

---

## Project Overview

This project implements a **credit card fraud detection system** using multiple machine learning algorithms to classify transactions as fraudulent or legitimate. The system addresses the challenge of highly imbalanced datasets (fraudulent transactions are rare) through:

* Advanced preprocessing and feature engineering (distance calculations, demographic features, temporal patterns)
* Class imbalance handling using SMOTE (Synthetic Minority Over-sampling Technique)
* Multiple model comparison (Logistic Regression, Random Forest, Decision Tree, KNN, XGBoost)
* Comprehensive evaluation with ROC curves, Precision-Recall curves, and confusion matrices

**Use-cases:** Financial fraud prevention, transaction monitoring systems, research demo, portfolio project.

---

## Features

* **Multi-Model Comparison:** Train and evaluate 5 different classifiers simultaneously
* **Advanced Feature Engineering:** Geographic distance calculations, age derivation, temporal features, profession categorization
* **Class Imbalance Handling:** SMOTE oversampling for minority class
* **Comprehensive Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC
* **Rich Visualizations:** Confusion matrices, ROC curves, Precision-Recall curves for all models
* **Automated Output:** All metrics and plots saved to organized output directory
* **Scalable Preprocessing:** Handles large datasets with robust data cleaning and categorical encoding

---

## Files & Structure

```
credit-card-fraud-detection/
├─ Credit_Card_Fraud_Detection.ipynb  # Main Jupyter notebook with full pipeline
├─ outputs/
│  ├─ model_metrics_summary.csv       # Performance metrics for all models
│  ├─ confusion_matrix_*.png          # Confusion matrix for each model
│  ├─ roc_curves_all_models.png       # Combined ROC curves
│  └─ pr_curves_all_models.png        # Combined Precision-Recall curves
├─ requirements.txt                   # Python dependencies
└─ README.md                          # This file
```

---

## Requirements

* Python 3.8+ (3.10 recommended)
* Jupyter Notebook or Google Colab
* Recommended: 8GB+ RAM for processing full dataset

### Core Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
imbalanced-learn>=0.8.0
xgboost>=1.5.0
```

---

## Quick Start / Setup

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Prepare Dataset

Place your fraud dataset (`fraud_data.csv`) in an accessible directory. Update the `INPUT_PATH` variable in the notebook:

```python
INPUT_PATH = "/path/to/your/fraud_data.csv"
OUTPUT_DIR = "/path/to/outputs"
```

### 5. Run the Notebook

```bash
jupyter notebook Credit_Card_Fraud_Detection.ipynb
```

Or upload to Google Colab and run all cells.

---

## Dataset Description

The dataset contains credit card transaction records with the following key features:

### Core Features
* **Transaction Details:** `amt` (transaction amount), `trans_date_trans_time` (timestamp)
* **Location Data:** `lat`, `long` (customer location), `merch_lat`, `merch_long` (merchant location)
* **Merchant Info:** `merchant` (merchant name), `category` (transaction category)
* **Customer Demographics:** `dob` (date of birth), `job` (profession), `city_pop` (city population)
* **Target Variable:** `is_fraud` (0 = legitimate, 1 = fraudulent)

### Engineered Features
* **age:** Customer age in years (derived from DOB and transaction date)
* **distance_km:** Haversine distance between customer and merchant locations
* **hour:** Hour of day (0-23) when transaction occurred
* **day_of_week:** Day name (Monday-Sunday)
* **professions:** Grouped profession categories (Education, Healthcare, STEM, Business, Creative, Construction, PublicSector, Pilot, Other)

### Dataset Characteristics
* **Size:** ~14,000 transactions
* **Class Distribution:** Highly imbalanced (~12,600 legitimate vs ~1,800 fraudulent)
* **Imbalance Ratio:** ~7:1 (legitimate:fraudulent)

---

## Data Preprocessing

### 1. Data Cleaning
* Remove duplicates and drop unnecessary columns (`trans_num`)
* Fix corrupted target labels (extract leading 0/1 digit)
* Handle missing values with appropriate defaults

### 2. Temporal Feature Extraction
```python
# Parse datetime and extract features
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], format="%d-%m-%Y %H:%M")
df['hour'] = df['trans_date_trans_time'].dt.hour
df['day_of_week'] = df['trans_date_trans_time'].dt.day_name()
```

### 3. Geographic Distance Calculation
```python
# Haversine formula for distance in kilometers
def haversine(lat1, lon1, lat2, lon2):
    # Earth radius: 6371 km
    # Returns distance between two geographic coordinates
```

### 4. Profession Categorization
* Groups 400+ unique job titles into 9 broad categories
* Reduces dimensionality while preserving semantic information

### 5. Categorical Encoding
* **Merchant Cardinality Reduction:** Keep top 50 merchants, group others as 'OTHER'
* **One-Hot Encoding:** Convert all categorical features to binary columns
* **Final Feature Count:** 82 features after encoding

### 6. Class Imbalance Handling
* **SMOTE (Synthetic Minority Over-sampling Technique):**
  - Generates synthetic fraudulent samples
  - Balances training set to 1:1 ratio
  - Applied only to training data (not test set)

### 7. Feature Scaling
* StandardScaler applied to all features
* Ensures models like Logistic Regression and KNN perform optimally

---

## Model Training

### Models Implemented

1. **Logistic Regression**
   - Linear baseline model
   - Max iterations: 1000
   - Good for interpretability

2. **Random Forest**
   - Ensemble of 200 decision trees
   - Class-weighted for imbalance
   - Robust to overfitting

3. **Decision Tree**
   - Single tree classifier
   - Class-weighted
   - Interpretable splits

4. **K-Nearest Neighbors (KNN)**
   - K=5 neighbors
   - Distance-based classification
   - Requires scaled features

5. **XGBoost**
   - Gradient boosting framework
   - State-of-the-art performance
   - Handles imbalance well

### Training Configuration

```python
# Common settings
RANDOM_STATE = 42
TEST_SIZE = 0.3  # 70-30 train-test split
PROB_THRESHOLD = 0.5  # Classification threshold
```

### Training Process

```python
# Pseudocode workflow
1. Load and preprocess data
2. Split into train/test (stratified by fraud label)
3. Apply SMOTE to training set
4. Scale features using StandardScaler
5. Train each model on resampled training data
6. Predict on test set
7. Evaluate and save metrics
```

---

## Model Performance

### Summary Table

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|-------|----------|-----------|--------|----------|---------|--------|
| **XGBoost** | **97.91%** | **90.68%** | **92.71%** | **91.68%** | **99.30%** | **97.14%** |
| **Random Forest** | 97.27% | 90.33% | 87.29% | 88.78% | 98.82% | 95.62% |
| **Decision Tree** | 96.55% | 83.05% | 90.65% | 86.68% | 94.02% | 76.45% |
| **Logistic Regression** | 94.23% | 82.06% | 68.41% | 74.62% | 89.80% | 71.83% |
| **KNN** | 90.20% | 63.73% | 48.60% | 55.14% | 83.06% | 52.25% |

### Key Insights

**🏆 Best Overall Model: XGBoost**
* Highest across all metrics
* Exceptional ROC-AUC (99.30%) indicates excellent class separation
* Best PR-AUC (97.14%) shows strong performance on imbalanced data
* Balanced precision-recall tradeoff

**🥈 Runner-up: Random Forest**
* Close performance to XGBoost
* Slightly lower recall but maintains high precision
* More interpretable through feature importance

**📊 Metric Interpretation:**
* **Precision:** Of predicted frauds, what % are actually fraudulent? (Higher = fewer false alarms)
* **Recall:** Of actual frauds, what % did we catch? (Higher = fewer missed frauds)
* **F1-Score:** Harmonic mean of precision and recall
* **ROC-AUC:** Overall ability to discriminate between classes
* **PR-AUC:** Performance on imbalanced data (more important than ROC-AUC here)

---

## Evaluation Metrics & Visualizations

### Generated Outputs

#### 1. Confusion Matrices
Individual confusion matrix for each model showing:
* True Negatives (correct legitimate predictions)
* False Positives (false fraud alarms)
* False Negatives (missed frauds) ⚠️ Most critical error
* True Positives (correctly caught frauds)

**Saved as:** `confusion_matrix_<ModelName>.png`

#### 2. ROC Curves (Receiver Operating Characteristic)
* Plots True Positive Rate vs False Positive Rate
* Shows model performance across all classification thresholds
* Diagonal line = random classifier baseline
* Area Under Curve (AUC) quantifies overall performance

**Saved as:** `roc_curves_all_models.png`

#### 3. Precision-Recall Curves
* More informative than ROC for imbalanced datasets
* Shows tradeoff between precision and recall
* Higher area under curve = better performance

**Saved as:** `pr_curves_all_models.png`

#### 4. Metrics Summary CSV
Complete metrics table for all models with sortable columns.

**Saved as:** `model_metrics_summary.csv`

---

## Results & Insights

### Why XGBoost Performs Best

1. **Gradient Boosting:** Iteratively corrects errors from previous trees
2. **Regularization:** Built-in L1/L2 regularization prevents overfitting
3. **Handles Imbalance:** Naturally adapts to class weights
4. **Feature Interactions:** Captures complex non-linear patterns

### Feature Importance (Top Features)

Based on Random Forest and XGBoost analysis, key fraud indicators include:
* **Transaction Amount:** Unusually high or low amounts
* **Distance:** Geographic distance between customer and merchant
* **Hour of Day:** Late night transactions more suspicious
* **Merchant Category:** Certain categories have higher fraud rates
* **Customer Age:** Demographic patterns in fraud attempts

### Business Impact

Using XGBoost model in production:
* **92.71% Fraud Detection Rate:** Catches 9 out of 10 fraudulent transactions
* **90.68% Precision:** Only ~9% false positive rate
* **Cost Savings:** Prevents losses while minimizing customer friction from false alarms

### Model Selection Guidance

| Use Case | Recommended Model | Reason |
|----------|-------------------|--------|
| Production System | **XGBoost** | Best overall performance |
| Real-time API | **Random Forest** | Fast inference, nearly as good |
| Interpretability Needed | **Decision Tree** | Clear decision rules |
| Baseline/Prototype | **Logistic Regression** | Simple, fast training |
| Not Recommended | KNN | Poor performance, slow inference |

---

## Tips, Troubleshooting & FAQ

### Common Issues

**Q: "Model predicts everything as non-fraud"**
* Check class weights are enabled (`class_weight='balanced'`)
* Verify SMOTE is applied correctly
* Confirm target variable is properly encoded (0/1)

**Q: "SMOTE import fails"**
* Install: `pip install imbalanced-learn`
* Or disable SMOTE and use original training set (performance will degrade)

**Q: "XGBoost not found"**
* Install: `pip install xgboost`
* Fallback to GradientBoostingClassifier is automatic

**Q: "Memory error during training"**
* Reduce dataset size for testing
* Use `RandomForestClassifier(n_estimators=100)` instead of 200
* Consider using Google Colab with GPU runtime

**Q: "Feature shape mismatch"**
* Ensure train and test sets have same features after encoding
* Check for missing values causing NaN columns
* Verify categorical encoding is consistent

### Performance Optimization

**Speed up training:**
```python
# Reduce estimators
RandomForestClassifier(n_estimators=100)  # instead of 200

# Use fewer neighbors
KNeighborsClassifier(n_neighbors=3)  # instead of 5

# Limit max depth
DecisionTreeClassifier(max_depth=10)
```

**Reduce memory usage:**
```python
# Sample dataset
df_sample = df.sample(frac=0.5, random_state=42)

# Reduce merchant cardinality
TOP_N_MERCHANTS = 20  # instead of 50
```

### Best Practices

1. **Always use stratified split** to maintain class distribution
2. **Never apply SMOTE to test set** (causes data leakage)
3. **Scale features** before training KNN and Logistic Regression
4. **Monitor PR-AUC** more than accuracy for imbalanced data
5. **Tune threshold** based on business cost of false positives vs false negatives

---

## Future Improvements

### Model Enhancements
- [ ] Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- [ ] Ensemble methods (stacking multiple models)
- [ ] Neural network approaches (LSTM for temporal patterns)
- [ ] Cost-sensitive learning with custom loss functions

### Feature Engineering
- [ ] Transaction velocity features (frequency in last hour/day)
- [ ] Historical fraud rate per merchant/category
- [ ] Time since last transaction
- [ ] Customer transaction profile (average amount, typical categories)
- [ ] Seasonal and day-of-week patterns

### Deployment
- [ ] Model serialization (save/load trained models)
- [ ] REST API for real-time predictions
- [ ] Batch processing pipeline
- [ ] Model monitoring and retraining workflow
- [ ] A/B testing framework

### Evaluation
- [ ] Cross-validation instead of single train-test split
- [ ] Cost-benefit analysis (fraud cost vs false alarm cost)
- [ ] Fairness metrics across demographic groups
- [ ] Temporal validation (train on old data, test on recent)

---

## Contributing

Contributions welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes with clear commit messages
4. Add tests if applicable
5. Update documentation
6. Submit a pull request with detailed description

**Areas for contribution:**
* Additional models (LightGBM, CatBoost, Neural Networks)
* Feature engineering ideas
* Visualization improvements
* Performance optimization
* Documentation enhancements

---

## License

MIT License

Copyright (c) 2025 Nikhil

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## Acknowledgments

* Dataset preprocessing inspired by common fraud detection practices
* SMOTE implementation from `imbalanced-learn` library
* XGBoost framework for gradient boosting
* Scikit-learn for classical ML algorithms

---

## Contact

For questions, suggestions, or collaboration:
* **GitHub:** [yourusername](https://github.com/yourusername)
* **Email:** your.email@example.com

---

**⭐ If you found this project helpful, please star the repository!**
