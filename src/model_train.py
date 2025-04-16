import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve
import matplotlib.pyplot as plt
import joblib
import os

def split_data(data: pd.DataFrame, target_col: str = 'Churn'):
    """Split the data into train/test sets and encode the target."""
    X = data.drop(columns=[target_col])
    y = data[target_col].map({'Yes': 1, 'No': 0})  # Adjusted for PR curve (positive = churned)
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

def train_random_forest(X_train, y_train):
    """Train a Random Forest with basic hyperparameter tuning."""
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }
    grid = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    print("Best Parameters:", grid.best_params_)
    return grid.best_estimator_

def evaluate_precision_recall_threshold(model, X_test, y_test):
    """Evaluate model with Precision-Recall curve and find best threshold."""
    y_probs = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    print(f"\n[INFO] Best Threshold: {best_threshold:.2f}")
    print(f"[INFO] Precision: {precision[best_idx]:.2f}")
    print(f"[INFO] Recall: {recall[best_idx]:.2f}")
    print(f"[INFO] F1-Score: {f1_scores[best_idx]:.2f}\n")

    # Classification report with best threshold
    y_pred = (y_probs >= best_threshold).astype(int)
    print("[INFO] Classification Report (Best Threshold):")
    print(classification_report(y_test, y_pred))

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')
    plt.scatter(recall[best_idx], precision[best_idx], marker='o', color='red', label='Best Threshold')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_threshold

def save_model(model, path="models/random_forest_model.pkl"):
    """Save the trained model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"[INFO] Model saved to {path}")
