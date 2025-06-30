# -*- coding: utf-8 -*-
# %% [markdown]
"""
Homework Solution:
Predicting Titanic survival using SVM, KNN, and Random Forest classifiers.
"""

# %%
# Load and preprocess data
import pandas as pd
data = pd.read_csv(r'D:\course source\aiSummerCamp2025-master\day1\assignment\data\train.csv')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv(r'D:\course source\aiSummerCamp2025-master\day1\assignment\data\train.csv')
df = data.copy()
print("Original data shape:", df.shape)
df.sample(5)

# %%
# Feature engineering
# Drop unnecessary columns
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

# Handle missing values
print("\nMissing values before handling:")
print(df.isnull().sum())

# Fill Age with median and drop the 2 missing Embarked rows
df['Age'].fillna(df['Age'].median(), inplace=True)
df.dropna(subset=['Embarked'], inplace=True)

print("\nMissing values after handling:")
print(df.isnull().sum())

# %%
# Convert categorical data into numerical data
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
print("\nData after one-hot encoding:")
df.sample(5)

# %%
# Separate features and labels
X = df.drop(columns=['Survived'])
y = df['Survived']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
# Build and evaluate models
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Train and evaluate a model, returning metrics and predictions."""
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    return {
        'model': model.__class__.__name__,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report
    }

# Initialize models
models = [
    SVC(random_state=42),
    KNeighborsClassifier(),
    RandomForestClassifier(random_state=42)
]

# Evaluate each model
results = []
for model in models:
    result = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
    results.append(result)

# %%
# Display results
for result in results:
    print(f"\n=== {result['model']} ===")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall: {result['recall']:.4f}")
    print(f"F1 Score: {result['f1']:.4f}")
    print("\nConfusion Matrix:")
    print(result['confusion_matrix'])
    print("\nClassification Report:")
    print(result['classification_report'])

# %%
# Feature importance for Random Forest
import matplotlib.pyplot as plt
import numpy as np

rf_model = models[2]  # Get the Random Forest model
importances = rf_model.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances (Random Forest)")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45)
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.show()