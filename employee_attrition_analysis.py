
"""
EMPLOYEE ATTRITION PREDICTION - COMPLETE ANALYSIS
Author: Student
Date: April 2026
This script performs employee attrition prediction using the IBM HR dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("EMPLOYEE ATTRITION PREDICTION SYSTEM")
print("=" * 60)

# Load data
df = pd.read_csv("HR-Employee-Attrition-Dataset.csv")
print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")

# Prepare data
def prepare_data(df):
    data = df.copy()
    data['Attrition'] = data['Attrition'].map({'Yes': 1, 'No': 0})
    
    # Drop constant columns
    for col in ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']:
        if col in data.columns:
            data = data.drop(columns=[col])
    
    # Encode categorical variables
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        data[col] = LabelEncoder().fit_transform(data[col])
    
    X = data.drop('Attrition', axis=1)
    y = data['Attrition']
    return X, y

X, y = prepare_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"Attrition rate: {y.mean()*100:.1f}%")

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest (100 trees)': RandomForestClassifier(n_estimators=100, random_state=42),
    'Random Forest (200 trees)': RandomForestClassifier(n_estimators=200, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'KNN (7 neighbors)': KNeighborsClassifier(n_neighbors=7),
}

# Train and evaluate
results = []
print("\n" + "=" * 60)
print("TRAINING MODELS")
print("=" * 60)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results.append({
        'model_name': name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    })
    
    print(f"\n{name}:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

# Results summary
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('accuracy', ascending=False)

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)
best = results_df.iloc[0]
print(f"Model: {best['model_name']}")
print(f"Accuracy: {best['accuracy']:.4f}")
print(f"F1-Score: {best['f1_score']:.4f}")

# Feature importance (Random Forest)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False).head(10)

print("\n" + "=" * 60)
print("TOP 10 IMPORTANT FEATURES")
print("=" * 60)
for idx, row in feature_importance.iterrows():
    print(f"{row['feature']}: {row['importance']:.4f}")

print("\n✅ Analysis complete!")
