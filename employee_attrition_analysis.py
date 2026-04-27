"""
EMPLOYEE ATTRITION PREDICTION - AUTO LEADERBOARD
Updated version that works locally and auto-pushes to GitHub
"""

import pandas as pd
import numpy as np
import os
import subprocess
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("🎯 EMPLOYEE ATTRITION PREDICTION SYSTEM")
print("=" * 60)

# ============================================================
# STEP 1: GET USER NAME
# ============================================================
user_name = input("\n📝 Enter your name for the leaderboard: ")

# ============================================================
# STEP 2: LOAD DATA
# ============================================================
print("\n📂 Loading data...")

# Try multiple possible locations
possible_paths = [
    "HR-Employee-Attrition-Dataset.csv",
    "data/HR-Employee-Attrition-Dataset.csv",
    "../HR-Employee-Attrition-Dataset.csv"
]

df = None
for path in possible_paths:
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"✅ Found data at: {path}")
        break

if df is None:
    print("❌ Error: Could not find dataset")
    print("Please make sure HR-Employee-Attrition-Dataset.csv is in this folder")
    exit()

print(f"📊 Dataset: {len(df)} rows, {len(df.columns)} columns")

# ============================================================
# STEP 3: PREPARE DATA
# ============================================================
print("\n🔧 Preparing data...")

def prepare_data(df):
    data = df.copy()
    data['Attrition'] = data['Attrition'].map({'Yes': 1, 'No': 0})
    
    for col in ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']:
        if col in data.columns:
            data = data.drop(columns=[col])
    
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        data[col] = LabelEncoder().fit_transform(data[col])
    
    X = data.drop('Attrition', axis=1)
    y = data['Attrition']
    return X, y

X, y = prepare_data(df)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Training: {len(X_train)} samples")
print(f"✅ Testing: {len(X_test)} samples")

# ============================================================
# STEP 4: TRAIN MODELS
# ============================================================
print("\n" + "=" * 60)
print("🤖 TRAINING MODELS")
print("=" * 60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=7),
}

best_accuracy = 0
best_model_name = ""
best_precision = 0
best_recall = 0
best_f1 = 0

for name, model in models.items():
    print(f"\n🔄 Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"   ✅ Accuracy: {accuracy:.4f}")
    print(f"   ✅ F1-Score: {f1:.4f}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = name
        best_precision = precision
        best_recall = recall
        best_f1 = f1

print("\n" + "=" * 60)
print("🏆 BEST MODEL")
print("=" * 60)
print(f"Model: {best_model_name}")
print(f"Accuracy: {best_accuracy:.4f}")

# ============================================================
# STEP 5: UPDATE LEADERBOARD
# ============================================================
print("\n" + "=" * 60)
print("📊 UPDATING LEADERBOARD")
print("=" * 60)

leaderboard_file = "leaderboard.csv"

if os.path.exists(leaderboard_file):
    leaderboard = pd.read_csv(leaderboard_file)
    print("✅ Loaded existing leaderboard")
else:
    leaderboard = pd.DataFrame(columns=[
        'user_id', 'model_name', 'accuracy', 'precision', 
        'recall', 'f1_score', 'submitted_at'
    ])
    print("✅ Created new leaderboard")

# Remove old entry if exists
leaderboard = leaderboard[leaderboard['user_id'] != user_name]

# Add new entry
new_entry = pd.DataFrame([{
    'user_id': user_name,
    'model_name': best_model_name,
    'accuracy': round(best_accuracy, 4),
    'precision': round(best_precision, 4),
    'recall': round(best_recall, 4),
    'f1_score': round(best_f1, 4),
    'submitted_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}])

leaderboard = pd.concat([leaderboard, new_entry], ignore_index=True)
leaderboard = leaderboard.sort_values('accuracy', ascending=False)
leaderboard.to_csv(leaderboard_file, index=False)

print("\n📊 UPDATED LEADERBOARD:")
print("=" * 60)
print(leaderboard[['user_id', 'model_name', 'accuracy']].to_string(index=False))

# ============================================================
# STEP 6: GIT PUSH (FIXED VERSION)
# ============================================================
print("\n" + "=" * 60)
print("📤 PUSHING TO GITHUB...")
print("=" * 60)

# Configure git if not already configured
try:
    # Check if git is configured
    subprocess.run(["git", "config", "user.name"], check=True, capture_output=True)
except:
    print("⚙️ Setting up git configuration...")
    subprocess.run(["git", "config", "user.name", "Student"], check=False)
    subprocess.run(["git", "config", "user.email", "student@example.com"], check=False)

try:
    # Add the file
    subprocess.run(["git", "add", "leaderboard.csv"], check=True, capture_output=True)
    
    # Commit
    commit_msg = f"Add {user_name}'s submission with accuracy {best_accuracy:.4f}"
    subprocess.run(["git", "commit", "-m", commit_msg], check=True, capture_output=True)
    
    # Push
    result = subprocess.run(["git", "push"], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("\n✅ SUCCESS! Results pushed to GitHub!")
        print("🔗 https://github.com/neetasharma05/employee-attrition-leaderboard")
    else:
        print("\n⚠️ Push failed. You may need to:")
        print("   1. Run: git push")
        print("   2. Or share the leaderboard.csv file manually")
        
except subprocess.CalledProcessError as e:
    print("\n⚠️ Could not auto-push. The leaderboard was saved locally.")
    print("📁 leaderboard.csv has been updated in your folder")
    print("\n📌 To manually push to GitHub, run these commands:")
    print("   git add leaderboard.csv")
    print("   git commit -m 'Update leaderboard'")
    print("   git push")
    
print("\n" + "=" * 60)
print("🎉 COMPLETE!")
print("=" * 60)
