import pytest
import pandas as pd
import numpy as np
from src.features.feature_engineer import FeatureEngineer

@pytest.fixture
def feature_config():
    return {
        'features': {
            'time_features': ['year', 'month', 'day', 'dayofweek', 'quarter', 'is_weekend'],
            'categorical_features': ['country', 'store', 'product']
        }
    }

def test_create_time_features(feature_config, sample_data):
    engineer = FeatureEngineer(feature_config)
    df = engineer.create_time_features(sample_data.copy())
    
    for feature in feature_config['features']['time_features']:
        assert feature in df.columns
    
    assert pd.api.types.is_datetime64_any_dtype(df['date'])
    assert df['is_weekend'].isin([0, 1]).all()
    assert df['year'].between(2020, 2020).all()
    assert df['month'].between(1, 1).all()
    assert df['day'].between(1, 10).all()
    assert df['dayofweek'].between(0, 6).all()
    assert df['quarter'].between(1, 4).all()

def test_handle_categorical_features(feature_config, sample_data):
    engineer = FeatureEngineer(feature_config)
    df = engineer.handle_categorical_features(sample_data.copy())
    
    for feature in feature_config['features']['categorical_features']:
        assert pd.api.types.is_categorical_dtype(df[feature])

def test_create_features(feature_config, sample_data):
    engineer = FeatureEngineer(feature_config)
    df = engineer.create_features(sample_data.copy())
    
    for feature in feature_config['features']['time_features']:
        assert feature in df.columns
    
    for feature in feature_config['features']['categorical_features']:
        assert feature in df.columns
        assert pd.api.types.is_categorical_dtype(df[feature])

def test_get_feature_columns(feature_config):
    engineer = FeatureEngineer(feature_config)
    feature_columns = engineer.get_feature_columns()
    
    expected_features = (
        feature_config['features']['time_features'] +
        feature_config['features']['categorical_features']
    )
    assert set(feature_columns) == set(expected_features) 