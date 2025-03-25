import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

@pytest.fixture
def sample_config() -> Dict[str, Any]:
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
        'model': {
            'name': 'lightgbm',
            'params': {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'n_estimators': 100,
                'early_stopping_rounds': 10
            },
            'save_path': 'models/model.txt'
        },
        'training': {
            'test_size': 0.2,
            'random_state': 42
        }
    }

@pytest.fixture
def sample_data() -> pd.DataFrame:
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