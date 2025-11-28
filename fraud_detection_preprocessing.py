# fraud_detection_extended.py
# Extended training script: adds DecisionTree, KNN, XGBoost (or GradientBoosting fallback)
# and computes/saves metrics + ROC/PR/Confusion Matrix plots.

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings("ignore")

# Optional libs
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except Exception:
    SMOTE = None
    IMBLEARN_AVAILABLE = False

# ---------------- CONFIG ----------------
INPUT_PATH = "./fraud_data.csv"  # keep same as your notebook
OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.3
TOP_N_MERCHANTS = 50   # keep top N merchants; others -> 'OTHER' to limit dummies
PROB_THRESHOLD = 0.4   # your logistic threshold was 0.4
# ----------------------------------------

# ---------------- helper functions ----------------
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371.0
    return r * c

def clean_target(df, target_col='is_fraud'):
    # convert to string then extract single leading digit (0/1) if corrupted
    df[target_col] = df[target_col].astype(str).str.extract(r'([01])')[0]
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce').fillna(0).astype(int)
    return df

def top_n_group(series, n=TOP_N_MERCHANTS, other_label='OTHER'):
    top = series.value_counts().nlargest(n).index
    return series.where(series.isin(top), other_label)

def save_confusion_matrix(cm, labels, outpath, title="Confusion Matrix"):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=labels, yticklabels=labels, annot_kws={"size":12})
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

# ---------------- Load & preprocess (based on your original notebook) ----------------
print("Loading:", INPUT_PATH)
df = pd.read_csv(INPUT_PATH)

# Fix corrupted target values
if 'is_fraud' in df.columns:
    df = clean_target(df, 'is_fraud')
else:
    raise ValueError("Expected is_fraud column in dataset.")

# Drop any obvious duplicates and useless cols
df = df.drop_duplicates().reset_index(drop=True)

if 'trans_num' in df.columns:
    df = df.drop(columns=['trans_num'])

# create age
# NOTE: original notebook used "%d-%m-%Y %H:%M" for trans_date_trans_time and "%d-%m-%Y" for dob
# Attempt parse robustly with errors='coerce' then compute age (years)
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], format="%d-%m-%Y %H:%M", errors='coerce')
df['dob'] = pd.to_datetime(df['dob'], format="%d-%m-%Y", errors='coerce')
df['age'] = ((df['trans_date_trans_time'] - df['dob']).dt.days // 365).fillna(-1).astype(int)

# time features
df['time_of_day'] = df['trans_date_trans_time'].dt.strftime('%H:%M')
df['hour'] = df['trans_date_trans_time'].dt.hour.fillna(-1).astype(int)
df['day_of_week'] = df['trans_date_trans_time'].dt.day_name().fillna('Unknown')

# distance feature
if {'lat','long','merch_lat','merch_long'}.issubset(set(df.columns)):
    df['distance_km'] = df.apply(
        lambda r: haversine(r['lat'], r['long'], r['merch_lat'], r['merch_long']) 
                  if pd.notnull(r['lat']) and pd.notnull(r['merch_lat']) else 0.0,
        axis=1
    )
else:
    df['distance_km'] = 0.0

# job → grouped professions (use your function mapping)
def job_categories(profession):
    if pd.isna(profession):
        return 'Other'
    profession_lower = str(profession).lower()
    # collapsed mapping for brevity (keeps same logic as your earlier function)
    if any(k in profession_lower for k in ['educ', 'teacher', 'lectur', 'professor', 'research']):
        return 'Education'
    if any(k in profession_lower for k in ['nurs', 'therap', 'psych', 'health', 'medical', 'clinic', 'pharm']):
        return 'Healthcare'
    if any(k in profession_lower for k in ['engineer', 'scientist', 'developer', 'geoscientist','technolog']):
        return 'STEM'
    if any(k in profession_lower for k in ['account', 'tax', 'finance', 'bank', 'manager', 'sales']):
        return 'Business'
    if any(k in profession_lower for k in ['artist', 'designer', 'media', 'journalist', 'musician']):
        return 'Creative'
    if any(k in profession_lower for k in ['architect', 'construction', 'surveyor', 'civil']):
        return 'Construction'
    if any(k in profession_lower for k in ['police','fire','armed','civil service','government']):
        return 'PublicSector'
    if any(k in profession_lower for k in ['pilot']):
        return 'Pilot'
    return 'Other'

df['professions'] = df['job'].apply(job_categories)

# cast some columns
for c in ['merchant','category','city','state','day_of_week','professions']:
    if c in df.columns:
        df[c] = df[c].astype(str)

# Reduce merchant cardinality
if 'merchant' in df.columns:
    df['merchant'] = top_n_group(df['merchant'], n=TOP_N_MERCHANTS)

# choose feature columns similar to your earlier dftree
features = []
for c in ['merchant','category','amt','city_pop','age','hour','day_of_week','distance_km','professions']:
    if c in df.columns:
        features.append(c)

print("Using features:", features)
print("Target distribution:\n", df['is_fraud'].value_counts())

# ---------------- Encode features (one-hot for categorical) ----------------
cat_cols = [c for c in features if df[c].dtype == 'object' or df[c].dtype.name == 'category']
num_cols = [c for c in features if c not in cat_cols]

# One-hot encode categorical columns (safe for all models)
X = pd.get_dummies(df[features], columns=cat_cols, drop_first=True)
y = df['is_fraud'].astype(int)

print("After get_dummies, feature count:", X.shape[1])

# train/test split (stratify)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
print("Train/test shapes:", X_train.shape, X_test.shape)

# Optionally resample using SMOTE if available
if IMBLEARN_AVAILABLE:
    print("SMOTE available: performing oversampling on training set.")
    sm = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
else:
    print("SMOTE not available: using original training set (or you can enable undersampling).")
    X_train_res, y_train_res = X_train.copy(), y_train.copy()

# For LR and KNN we scale numeric columns. We'll scale the entire matrix to be consistent.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# ---------------- Define models ----------------
models = {}
models['LogisticRegression'] = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
models['RandomForest'] = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=RANDOM_STATE)
models['DecisionTree'] = DecisionTreeClassifier(class_weight='balanced', random_state=RANDOM_STATE)
models['KNN'] = KNeighborsClassifier(n_neighbors=5)

if XGBOOST_AVAILABLE:
    models['XGBoost'] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE)
else:
    models['GradientBoosting'] = GradientBoostingClassifier(random_state=RANDOM_STATE)

print("Models to train:", list(models.keys()))

# ---------------- Train, predict, evaluate ----------------
metrics = []
roc_curves = {}
pr_curves = {}

for name, model in models.items():
    print("\n--- Training:", name)
    # For tree-based models we can fit on unscaled (X_train_res), but scaled works for all; use scaled for consistency
    try:
        model.fit(X_train_scaled, y_train_res)
    except Exception as e:
        # fallback to unscaled if model fails on sparse dense shape mismatch
        print(f"Fit failed on scaled data for {name}: {e}; trying unscaled features.")
        model.fit(X_train_res, y_train_res)

    # predict probabilities/scores
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test_scaled)[:, 1]
    else:
        # decision_function or predict
        if hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test_scaled)
        else:
            y_score = model.predict(X_test_scaled).astype(float)

    # predictions with same threshold logic (for LR you used 0.4 earlier, but for consistent comparison use 0.5)
    y_pred = (y_score >= 0.5).astype(int)

    # metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_test, y_score)
    except Exception:
        roc_auc = np.nan
    try:
        pr_auc = average_precision_score(y_test, y_score)
    except Exception:
        pr_auc = np.nan

    print(f"{name} metrics -> Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")
    print("Classification report:\n", classification_report(y_test, y_pred, digits=4, zero_division=0))

    # confusion matrix plot saved
    cm = confusion_matrix(y_test, y_pred)
    cm_path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{name}.png")
    save_confusion_matrix(cm, labels=["Not Fraud","Fraud"], outpath=cm_path, title=f"{name} Confusion Matrix")
    print("Saved confusion matrix to:", cm_path)

    # store metrics
    metrics.append({
        'model': name,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    })

    # curves
    try:
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_curves[name] = (fpr, tpr)
    except Exception:
        pass
    try:
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        pr_curves[name] = (precision, recall)
    except Exception:
        pass

# ---------------- Save metrics dataframe ----------------
metrics_df = pd.DataFrame(metrics).sort_values(by='pr_auc', ascending=False).reset_index(drop=True)
metrics_csv = os.path.join(OUTPUT_DIR, "model_metrics_summary.csv")
metrics_df.to_csv(metrics_csv, index=False)
print("\nSaved metrics summary to:", metrics_csv)
print(metrics_df)

# ---------------- Plot combined ROC ----------------
if roc_curves:
    plt.figure(figsize=(8,6))
    for name, (fpr, tpr) in roc_curves.items():
        auc_val = roc_auc_score(y_test, models[name].predict_proba(X_test_scaled)[:,1]) if hasattr(models[name], 'predict_proba') else np.nan
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={metrics_df.loc[metrics_df['model']==name,'roc_auc'].values[0]:.3f})")
    plt.plot([0,1],[0,1], color='gray', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - All Models")
    plt.legend(loc="lower right")
    plt.grid(True)
    roc_png = os.path.join(OUTPUT_DIR, "roc_curves_all_models.png")
    plt.tight_layout()
    plt.savefig(roc_png)
    plt.close()
    print("Saved ROC curves to:", roc_png)
else:
    print("No ROC curves to plot.")

# ---------------- Plot combined Precision-Recall ----------------
if pr_curves:
    plt.figure(figsize=(8,6))
    for name, (precision, recall) in pr_curves.items():
        # compute approx pr auc for label
        try:
            auc_pr = metrics_df.loc[metrics_df['model']==name,'pr_auc'].values[0]
        except Exception:
            auc_pr = np.nan
        plt.plot(recall, precision, lw=2, label=f"{name} (PR-AUC={auc_pr:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves - All Models")
    plt.legend(loc="lower left")
    plt.grid(True)
    pr_png = os.path.join(OUTPUT_DIR, "pr_curves_all_models.png")
    plt.tight_layout()
    plt.savefig(pr_png)
    plt.close()
    print("Saved PR curves to:", pr_png)
else:
    print("No PR curves to plot.")

print("\nAll done. Outputs saved to folder:", OUTPUT_DIR)
print("Model metrics:\n", metrics_df.to_string(index=False))
