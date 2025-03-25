import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from src.data.data_processor import DataProcessor

@pytest.fixture
def data_config():
    return {
        'data': {
            'train_path': 'data/train.csv',
            'test_path': 'data/test.csv',
            'target_column': 'num_sold',
            'features': ['year', 'month', 'day', 'dayofweek', 'quarter', 'is_weekend', 'country', 'store', 'product']
        },
        'features': {
            'time_features': ['year', 'month', 'day', 'dayofweek', 'quarter', 'is_weekend'],
            'categorical_features': ['country', 'store', 'product']
        },
        'training': {
            'test_size': 0.2,
            'random_state': 42
        }
    }

@pytest.fixture
def sample_data():
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    countries = ['US', 'UK', 'CA']
    stores = ['Store1', 'Store2', 'Store3']
    products = ['Product1', 'Product2', 'Product3']
    
    data = []
    for date in dates:
        for country in countries:
            for store in stores:
                for product in products:
                    data.append({
                        'id': len(data) + 1,
                        'date': date,
                        'country': country,
                        'store': store,
                        'product': product,
                        'num_sold': np.random.randint(0, 100)
                    })
    
    return pd.DataFrame(data)

def test_create_time_features(data_config, sample_data):
    processor = DataProcessor(data_config)
    df = processor.create_time_features(sample_data)
    
    assert 'year' in df.columns
    assert 'month' in df.columns
    assert 'day' in df.columns
    assert 'dayofweek' in df.columns
    assert 'quarter' in df.columns
    assert 'is_weekend' in df.columns
    
    assert df['date'].dtype == 'datetime64[ns]'
    assert df['is_weekend'].dtype == 'int64'
    assert df['year'].min() == 2023
    assert df['year'].max() == 2023
    assert df['month'].min() == 1
    assert df['month'].max() == 12
    assert df['day'].min() == 1
    assert df['day'].max() == 31
    assert df['dayofweek'].min() == 0
    assert df['dayofweek'].max() == 6
    assert df['quarter'].min() == 1
    assert df['quarter'].max() == 4

def test_preprocess_data(data_config, sample_data):
    processor = DataProcessor(data_config)
    df = processor.preprocess_data(sample_data)
    
    assert 'year' in df.columns
    assert 'month' in df.columns
    assert 'day' in df.columns
    assert 'dayofweek' in df.columns
    assert 'quarter' in df.columns
    assert 'is_weekend' in df.columns
    assert df['country'].dtype == 'category'
    assert df['store'].dtype == 'category'
    assert df['product'].dtype == 'category'

def test_split_data(data_config, sample_data):
    processor = DataProcessor(data_config)
    train_df, val_df = processor.split_data(sample_data)
    
    assert len(train_df) + len(val_df) == len(sample_data)
    assert len(train_df) > len(val_df)
    assert len(val_df) == int(len(sample_data) * data_config['training']['test_size'])

def test_prepare_data(data_config, sample_data):
    processor = DataProcessor(data_config)
    train_df, val_df, test_df = processor.prepare_data()
    
    assert len(train_df) + len(val_df) + len(test_df) == len(sample_data)
    assert len(train_df) > len(val_df)
    assert len(val_df) == int(len(sample_data) * data_config['training']['test_size'])
    assert 'year' in train_df.columns
    assert 'month' in train_df.columns
    assert 'day' in train_df.columns
    assert 'dayofweek' in train_df.columns
    assert 'quarter' in train_df.columns
    assert 'is_weekend' in train_df.columns
    assert train_df['country'].dtype == 'category'
    assert train_df['store'].dtype == 'category'
    assert train_df['product'].dtype == 'category' 