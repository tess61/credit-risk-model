# src/train.py
"""Model training and tracking for credit risk model."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_processed_data(file_path):
    """Load processed dataset."""
    abs_path = os.path.abspath(file_path)
    logger.info(f"Attempting to load file: {abs_path}")
    try:
        df = pd.read_csv(abs_path)
        logger.info(f"Loaded dataset with shape {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File {abs_path} not found")
        raise
    except pd.errors.ParserError:
        logger.error(f"Unable to parse {abs_path}. Check file format.")
        raise

def train_and_evaluate(model, param_grid, X_train, X_test, y_train, y_test, model_name):
    """Train model with GridSearchCV and log to MLflow."""
    with mlflow.start_run(run_name=model_name):
        # Perform hyperparameter tuning
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        test_accuracy = accuracy_score(y_test, y_pred)
        test_roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Infer model signature
        signature = infer_signature(X_train, y_pred)
        
        # Create input example
        input_example = X_train.iloc[:5]
        
        # Log parameters, metrics, and model
        mlflow.log_params(best_params)
        mlflow.log_metric("train_roc_auc", best_score)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_roc_auc", test_roc_auc)
        mlflow.sklearn.log_model(
            sk_model=best_model,
            name=f"{model_name}_model",
            signature=signature,
            input_example=input_example
        )
        
        logger.info(f"{model_name} - Best Params: {best_params}")
        logger.info(f"{model_name} - Train ROC-AUC: {best_score:.4f}")
        logger.info(f"{model_name} - Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"{model_name} - Test ROC-AUC: {test_roc_auc:.4f}")
        
        return best_model, test_roc_auc

def main(input_path, target_col='is_high_risk'):
    """Main function to train and evaluate models."""
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")
    logger.info("MLflow tracking URI set to http://localhost:5000")
    
    # Load processed data
    df = load_processed_data(input_path)
    
    # Prepare features and target
    X = df.drop(columns=[target_col, 'CustomerId'])
    y = df[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    
    # Define models and parameter grids
    models = [
        (
            LogisticRegression(max_iter=1000),
            {'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']},
            "LogisticRegression"
        ),
        (
            GradientBoostingClassifier(random_state=42),
            {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]},
            "GradientBoosting"
        )
    ]
    
    # Train and evaluate models
    best_model = None
    best_roc_auc = 0
    for model, param_grid, model_name in models:
        model, roc_auc = train_and_evaluate(
            model, param_grid, X_train, X_test, y_train, y_test, model_name
        )
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_model = model
            best_model_name = model_name
    
    # Register the best model
    with mlflow.start_run(run_name=f"Best_Model_{best_model_name}"):
        signature = infer_signature(X_train, best_model.predict(X_train))
        input_example = X_train.iloc[:5]
        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="best_model",
            signature=signature,
            input_example=input_example
        )
        mlflow.set_tag("best_model", best_model_name)
        logger.info(f"Registered best model: {best_model_name} with Test ROC-AUC: {best_roc_auc:.4f}")
    
    return best_model

if __name__ == "__main__":
    input_path = r"C:\Users\tes's\Desktop\K_AIM\Week 5\credit-risk-model\data\processed\processed_data.csv"
    main(input_path)