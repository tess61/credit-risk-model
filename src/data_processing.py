# src/data_processing.py
"""Feature engineering and proxy target variable creation for credit risk model."""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path):
    """Load dataset from CSV file."""
    logger.info(f"Current working directory: {os.getcwd()}")
    abs_path = os.path.abspath(file_path)
    logger.info(f"Attempting to load file: {abs_path}")
    try:
        df = pd.read_csv(abs_path)
        logger.info(f"Loaded dataset from {abs_path} with shape {df.shape}")
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
        
        logger.info(f"Aggregated columns: {list(agg_features.columns)}")
        
        agg_features.columns = [
            'CustomerId', 'total_amount', 'avg_amount', 'transaction_count', 
            'std_amount', 'unique_transactions'
        ]
        
        agg_features['std_amount'] = agg_features['std_amount'].fillna(0)
        
        logger.info("Created aggregate features")
        return agg_features
    except KeyError as e:
        logger.error(f"KeyError in create_aggregate_features: {e}")
        raise

def create_rfm_features(df):
    """Calculate RFM metrics and create is_high_risk label using K-Means clustering."""
    try:
        # Convert TransactionStartTime to datetime
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        
        # Calculate RFM metrics per CustomerId
        current_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
        rfm_df = df.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (current_date - x.max()).days,  # Recency
            'TransactionId': 'count',  # Frequency
            'Amount': 'sum'  # Monetary
        }).reset_index()
        
        rfm_df.columns = ['CustomerId', 'recency', 'frequency', 'monetary']
        
        # Normalize RFM features for clustering
        rfm_features = ['recency', 'frequency', 'monetary']
        scaler = MinMaxScaler()
        rfm_scaled = scaler.fit_transform(rfm_df[rfm_features])
        
        # Apply K-Means clustering (3 clusters: low, medium, high risk)
        kmeans = KMeans(n_clusters=3, random_state=42)
        rfm_df['cluster'] = kmeans.fit_predict(rfm_scaled)
        
        # Define high-risk cluster (high recency, low frequency, low monetary)
        cluster_summary = rfm_df.groupby('cluster')[rfm_features].mean().reset_index()
        cluster_summary['risk_score'] = (
            cluster_summary['recency'] - 
            (cluster_summary['frequency'] + cluster_summary['monetary']) / 2
        )
        high_risk_cluster = cluster_summary['cluster'][cluster_summary['risk_score'].idxmax()]
        
        # Assign is_high_risk label (1 for high-risk cluster, 0 otherwise)
        rfm_df['is_high_risk'] = (rfm_df['cluster'] == high_risk_cluster).astype(int)
        
        logger.info(f"RFM features created. High-risk cluster: {high_risk_cluster}")
        logger.info(f"Cluster summary:\n{cluster_summary}")
        
        return rfm_df[['CustomerId', 'recency', 'frequency', 'monetary', 'is_high_risk']]
    except KeyError as e:
        logger.error(f"KeyError in create_rfm_features: {e}")
        raise
    except ValueError as e:
        logger.error(f"ValueError in create_rfm_features: {e}")
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
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])
    
    logger.info("Built preprocessing pipeline")
    return preprocessor

def process_data(input_path, output_path, target_col='is_high_risk'):
    """Main function to process data and save output."""
    # Load data
    df = load_data(input_path)
    
    # Create aggregate features
    agg_df = create_aggregate_features(df)
    
    # Create RFM features and is_high_risk label
    rfm_df = create_rfm_features(df)
    
    # Merge aggregates and RFM features
    df = df.merge(agg_df, on='CustomerId', how='left')
    df = df.merge(rfm_df, on='CustomerId', how='left')
    
    # Extract temporal features
    df = extract_temporal_features(df)
    
    # Define feature columns
    numerical_cols = [
        'Amount', 'Value', 'total_amount', 'avg_amount', 'transaction_count', 
        'std_amount', 'transaction_hour', 'transaction_day', 'transaction_month',
        'recency', 'frequency', 'monetary'
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
    
    # Fit and transform data (exclude is_high_risk for transformation)
    transformed_data = pipeline.fit_transform(df.drop(columns=[target_col], errors='ignore'))
    
    # Convert back to DataFrame
    cat_feature_names = pipeline.named_steps['preprocessor']\
        .named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols)
    feature_names = numerical_cols + list(cat_feature_names)
    transformed_df = pd.DataFrame(transformed_data, columns=feature_names)
    
    # Add CustomerId and is_high_risk
    transformed_df['CustomerId'] = df['CustomerId'].reset_index(drop=True)
    transformed_df[target_col] = df[target_col].reset_index(drop=True)
    
    # Skip WoE/IV transformation due to compatibility issues
    # woe_features = numerical_cols + categorical_cols
    # transformed_df, iv_values = apply_woe_iv(df, woe_features, target_col)
    # transformed_df = transformed_df.merge(
    #     df[['CustomerId', target_col]], on='CustomerId', how='left'
    # )
    
    # Save processed data
    transformed_df.to_csv(output_path, index=False)
    logger.info(f"Saved processed data to {output_path}")
    
    return transformed_df, pipeline

if __name__ == "__main__":
    # Use absolute path for Windows
    input_path = r"C:\Users\tes's\Desktop\K_AIM\Week 5\credit-risk-model\data\raw\data.csv"
    output_path = r"C:\Users\tes's\Desktop\K_AIM\Week 5\credit-risk-model\data\processed\processed_data.csv"
    target_col = 'is_high_risk'
    processed_df, pipeline = process_data(input_path, output_path, target_col)
