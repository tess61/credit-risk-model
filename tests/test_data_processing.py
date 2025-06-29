
# tests/test_data_processing.py
"""Unit tests for data_processing.py."""

import sys
import os
import pytest
import pandas as pd

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import load_data, create_aggregate_features, create_rfm_features

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    data = pd.DataFrame({
        'CustomerId': ['C1', 'C1', 'C2', 'C3'],
        'TransactionId': ['T1', 'T2', 'T3', 'T4'],
        'Amount': [100, 200, 150, 300],
        'TransactionStartTime': ['2023-01-01 10:00:00', '2023-01-02 12:00:00', '2023-01-03 14:00:00', '2023-01-04 16:00:00'],
        'CurrencyCode': ['USD', 'USD', 'USD', 'USD'],
        'CountryCode': [1, 1, 1, 1],
        'ProviderId': ['P1', 'P1', 'P2', 'P3'],
        'ProductId': ['A', 'A', 'B', 'C'],
        'ProductCategory': ['cat1', 'cat1', 'cat2', 'cat3'],
        'ChannelId': ['C1', 'C1', 'C2', 'C3'],
        'Value': [100, 200, 150, 300],
        'PricingStrategy': [0, 0, 1, 1],
        'FraudResult': [0, 0, 0, 0]
    })
    return data

def test_load_data():
    """Test load_data function."""
    with pytest.raises(FileNotFoundError):
        load_data("nonexistent.csv")

def test_create_aggregate_features(sample_data):
    """Test create_aggregate_features function."""
    agg_df = create_aggregate_features(sample_data)
    assert list(agg_df.columns) == [
        'CustomerId', 'total_amount', 'avg_amount', 'transaction_count', 
        'std_amount', 'unique_transactions'
    ]
    assert agg_df.shape[0] == 3  # Three customers
    assert agg_df.loc[agg_df['CustomerId'] == 'C1', 'total_amount'].iloc[0] == 300

def test_create_rfm_features(sample_data):
    """Test create_rfm_features function."""
    rfm_df = create_rfm_features(sample_data)
    assert list(rfm_df.columns) == ['CustomerId', 'recency', 'frequency', 'monetary', 'is_high_risk']
    assert rfm_df.shape[0] == 3
    assert 'is_high_risk' in rfm_df.columns
    assert rfm_df['is_high_risk'].isin([0, 1]).all()
