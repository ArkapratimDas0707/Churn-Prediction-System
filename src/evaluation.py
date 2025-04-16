
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import joblib
import os


def evaluate_model(model, X_test, y_test):
    """Evaluate the model with classification metrics and AUC."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

    # Optional threshold optimization
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    return precision, recall, thresholds
