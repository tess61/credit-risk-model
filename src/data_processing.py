# src/data_processing.py
"""Feature engineering script for credit risk model."""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from xverse.transformer import WOE
import woe.feature_process as woe
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path):
    """Load dataset from CSV file."""
    # Log current working directory for debugging
    logger.info(f"Current working directory: {os.getcwd()}")
    # Convert to absolute path for robustness
    abs_path = os.path.abspath(file_path)
    logger.info(f"Attempting to load file: {abs_path}")
    try:
        df = pd.read_csv(abs_path)
        logger.info(f"Loaded dataset from {abs_path} with shape {df.shape}")
        # Log column names for verification
        logger.info(f"Columns in dataset: {list(df.columns)}")
        return df
    except FileNotFoundError:
        logger.error(f"File {abs_path} not found")
        raise
    except pd.errors.ParserError:
        logger.error(f"Unable to parse {abs_path}. Check file format or encoding.")
        raise

def create_aggregate_features(df):
    """Create aggregate features per CustomerId."""
    try:
        agg_features = df.groupby('CustomerId').agg({
            'Amount': ['sum', 'mean', 'count', 'std'],
            'TransactionId': 'nunique'  # Unique transactions
        }).reset_index()
        
        # Log aggregated columns for debugging
        logger.info(f"Aggregated columns: {list(agg_features.columns)}")
        
        # Flatten column names
        agg_features.columns = [
            'CustomerId', 'total_amount', 'avg_amount', 'transaction_count', 
            'std_amount', 'unique_transactions'
        ]
        
        # Fill NaN in std_amount (for customers with single transaction)
        agg_features['std_amount'] = agg_features['std_amount'].fillna(0)
        
        logger.info("Created aggregate features")
        return agg_features
    except KeyError as e:
        logger.error(f"KeyError in create_aggregate_features: {e}")
        raise

def extract_temporal_features(df):
    """Extract temporal features from TransactionStartTime."""
    try:
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        df['transaction_hour'] = df['TransactionStartTime'].dt.hour
        df['transaction_day'] = df['TransactionStartTime'].dt.day
        df['transaction_month'] = df['TransactionStartTime'].dt.month
        df['transaction_year'] = df['TransactionStartTime'].dt.year
        logger.info("Extracted temporal features")
        return df
    except KeyError:
        logger.error("Column 'TransactionStartTime' not found in dataset")
        raise
    except ValueError:
        logger.error("Invalid format in 'TransactionStartTime'. Ensure it's a valid datetime.")
        raise

def build_preprocessing_pipeline(numerical_cols, categorical_cols):
    """Build sklearn pipeline for preprocessing."""
    # Numerical pipeline: Impute missing values with median, then standardize
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline: Impute missing values with mode, then one-hot encode
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine pipelines
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])
    
    logger.info("Built preprocessing pipeline")
    return preprocessor

def apply_woe_iv(df, features, target):
    """Apply Weight of Evidence (WoE) and Information Value (IV) transformation."""
    woe_transformer = WOE()
    df_woe = woe_transformer.fit_transform(df[features + [target]], df[target])
    
    # Get IV values
    iv_values = woe_transformer.iv_df
    logger.info(f"Information Values:\n{iv_values}")
    
    return df_woe, iv_values

def process_data(input_path, output_path, target_col=None):
    """Main function to process data and save output."""
    # Load data
    df = load_data(input_path)
    
    # Create aggregate features
    agg_df = create_aggregate_features(df)
    
    # Merge aggregates back to original data
    df = df.merge(agg_df, on='CustomerId', how='left')
    
    # Extract temporal features
    df = extract_temporal_features(df)
    
    # Define feature columns
    numerical_cols = [
        'Amount', 'Value', 'total_amount', 'avg_amount', 'transaction_count', 
        'std_amount', 'transaction_hour', 'transaction_day', 'transaction_month'
    ]
    categorical_cols = [
        'CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId', 'ProductCategory',
        'ChannelId', 'PricingStrategy', 'FraudResult'
    ]
    
    # Build and apply preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(numerical_cols, categorical_cols)
    pipeline = Pipeline([
        ('preprocessor', preprocessor)
    ])
    
    # Fit and transform data
    transformed_data = pipeline.fit_transform(df)
    
    # Convert back to DataFrame
    cat_feature_names = pipeline.named_steps['preprocessor']\
        .named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols)
    feature_names = numerical_cols + list(cat_feature_names)
    transformed_df = pd.DataFrame(transformed_data, columns=feature_names)
    
    # Add CustomerId and target (if provided)
    transformed_df['CustomerId'] = df['CustomerId'].reset_index(drop=True)
    if target_col:
        transformed_df[target_col] = df[target_col].reset_index(drop=True)
    
    # Apply WoE/IV if target is provided (for feature selection)
    if target_col:
        woe_features = numerical_cols + categorical_cols
        transformed_df, iv_values = apply_woe_iv(df, woe_features, target_col)
        transformed_df = transformed_df.merge(
            df[['CustomerId', target_col]], on='CustomerId', how='left'
        )
    
    # Save processed data
    transformed_df.to_csv(output_path, index=False)
    logger.info(f"Saved processed data to {output_path}")
    
    return transformed_df, pipeline

if __name__ == "__main__":
    # Use absolute path for Windows
    input_path = r"C:\Users\tes's\Desktop\K_AIM\Week 5\credit-risk-model\data\raw\data.csv"
    output_path = r"C:\Users\tes's\Desktop\K_AIM\Week 5\credit-risk-model\data\processed\processed_data.csv"
    target_col = None  # Will be 'is_high_risk' after Task 4
    processed_df, pipeline = process_data(input_path, output_path, target_col)