import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.model_train import split_data, train_random_forest, evaluate_precision_recall_threshold, save_model
from src.evaluation import evaluate_model


def run_pipeline():
    # Load the final preprocessed dataset
    file_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'processed', 'telco_final.csv')
    file_path = os.path.abspath(file_path)
    data = pd.read_csv(file_path)
    print(f"[INFO] Loaded data shape: {data.shape}")

    # Split data
    X_train, X_test, y_train, y_test = split_data(data)
    print("[INFO] Data split complete.")

    # Train model
    rf_model = train_random_forest(X_train, y_train)
    print("[INFO] Model training complete.")

    # Evaluate model
    evaluate_model(rf_model, X_test, y_test)
    print("[INFO] Evaluation complete.")

    evaluate_precision_recall_threshold(rf_model, X_test, y_test)
    # Save model
    save_model(rf_model)
    print("[INFO] Pipeline execution finished.")

if __name__ == "__main__":
    run_pipeline()
